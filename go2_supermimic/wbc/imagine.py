"""Differentiable WBC imagination rollouts, rewards, soft terminations, and critic input.

Mirrors the 18-DoF Go2+Piper WBC environment (``go2_supermimic/wbc/env_cfg.py``):
  - Soft contact probabilities from foot heights above local terrain replace
    hard ContactSensor booleans.
  - True world-frame finite-difference foot velocities use exact terrain height
    under the base at both steps (``ground_z0``, ``ground_z1``).
  - Analytic 6-DoF Piper FK (``piper_fk``) and yaw-invariant ``ee_world_height``
    provide smooth EE pose tracking and calibrated MuJoCo AABB self-collision
    penetration against the battery, base, and head/D415 collision bodies.
  - Deterministic 20D EE trajectory state advances via ``advance_ee_traj_state``
    and 210D actor history is rebuilt via ``wbc_obs_frame`` / ``wbc_push_history``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Literal

import mujoco
import numpy as np
import torch
import torch.nn.functional as F

from go2_supermimic.robot.go2_piper import DEFAULT_JOINT_POS, GO2_PIPER_XML
from go2_supermimic.supermimic.world_model import StateLayout
from go2_supermimic.target import terrain as tr
from go2_supermimic.wbc.commands import (
  LINK0_BODY_BOX_MAX,
  LINK0_BODY_BOX_MIN,
  TRAJ_STATE_DIM,
  advance_ee_traj_state,
)
from go2_supermimic.wbc.state import (
  FEET,
  GRAV,
  HIPS,
  KNEES,
  PIPER_COLLISION_LINK_GEOMS,
  QD,
  QD_ARM,
  QD_LEG,
  Q,
  Q_ARM,
  Q_LEG,
  V_YAW,
  W_B,
  WBC_CONTEXT_DIM,
  WBC_NUM_ARM_JOINTS,
  WBC_NUM_JOINTS,
  WBC_NUM_LEG_JOINTS,
  WBC_Q_REF_ARM_DIM,
  WBC_STATE_DIM,
  Z,
  PiperFkResult,
  WbcRobotConsts,
  _leaky_clamp_ste,
  ee_world_height,
  piper_fk,
  slew_step_arm_diff,
  wbc_apply_pose_delta,
  wbc_foot_positions_w,
  wbc_obs_frame,
  wbc_push_history,
)

WBC_STATE_LAYOUT = StateLayout(
  state_dim=WBC_STATE_DIM,
  z=Z,
  grav=GRAV,
  v_yaw=V_YAW,
  w_b=W_B,
  q=Q,
  qd=QD,
  hips=HIPS,
  knees=KNEES,
  feet=FEET,
)

# [s (82), q_ref_arm (6), cmd_vel (3), ee_pose_cmd (7), ee_rel_link0_fk (7), xi (9)]
WBC_CRITIC_BASE_DIM: int = (
  WBC_STATE_DIM + WBC_Q_REF_ARM_DIM + 3 + 7 + 7 + WBC_CONTEXT_DIM
)  # 114

_COLLISION_LINK_ORDER: tuple[str, ...] = tuple(name for name, _ in PIPER_COLLISION_LINK_GEOMS)

_BODY_COLLISION_BOX_GROUPS: tuple[tuple[str, ...], ...] = (
  ("link0_collision",),
  ("front_battery_collision_box",),
  ("base_collision",),
  ("head_collision1", "head_collision2"),
  ("d415_head_camera_collision_camera_main",),
)


@lru_cache(maxsize=4)
def _compiled_collision_proxy(
  xml_path: str = str(GO2_PIPER_XML),
) -> tuple[
  tuple[tuple[float, float, float], ...],
  tuple[tuple[float, float, float], ...],
  tuple[float, ...],
]:
  """Extract tight link0-frame body AABBs and link4..ee radii from compiled MJCF."""
  mj_model = mujoco.MjSpec.from_file(xml_path).compile()
  mj_data = mujoco.MjData(mj_model)
  mj_data.qpos[7:] = DEFAULT_JOINT_POS
  mujoco.mj_forward(mj_model, mj_data)

  l0_bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "link0")
  if l0_bid < 0:
    raise ValueError(f"Missing body 'link0' in {xml_path}")
  l0_pos = mj_data.xpos[l0_bid]
  l0_mat = mj_data.xmat[l0_bid].reshape(3, 3)

  box_mins: list[tuple[float, float, float]] = []
  box_maxs: list[tuple[float, float, float]] = []
  for geom_group in _BODY_COLLISION_BOX_GROUPS:
    grp_mins, grp_maxs = [], []
    for gname in geom_group:
      gid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, gname)
      if gid < 0:
        raise ValueError(f"Missing body collision geom '{gname}' in {xml_path}")
      p_w = mj_data.geom_xpos[gid]
      r_w = mj_data.geom_xmat[gid].reshape(3, 3)
      p_l0 = l0_mat.T @ (p_w - l0_pos)
      r_l0 = l0_mat.T @ r_w
      gtype = int(mj_model.geom_type[gid])
      if gtype == int(mujoco.mjtGeom.mjGEOM_BOX):
        ext = np.abs(r_l0) @ mj_model.geom_size[gid]
        grp_mins.append(p_l0 - ext)
        grp_maxs.append(p_l0 + ext)
      elif gtype == int(mujoco.mjtGeom.mjGEOM_MESH):
        mid = int(mj_model.geom_dataid[gid])
        adr = int(mj_model.mesh_vertadr[mid])
        num = int(mj_model.mesh_vertnum[mid])
        verts = mj_model.mesh_vert[adr : adr + num]
        pts = (verts @ r_l0.T) + p_l0
        grp_mins.append(pts.min(axis=0))
        grp_maxs.append(pts.max(axis=0))
      else:
        ext = np.full(3, float(mj_model.geom_rbound[gid]), dtype=np.float64)
        grp_mins.append(p_l0 - ext)
        grp_maxs.append(p_l0 + ext)
    bmin = np.min(grp_mins, axis=0)
    bmax = np.max(grp_maxs, axis=0)
    box_mins.append(tuple(float(v) for v in bmin))
    box_maxs.append(tuple(float(v) for v in bmax))

  # Tight principal half-extent radii (instead of inflated diagonal rbound)
  # for link4/link5, 0.035 m wrist hub radius covering link6+gripper_base, and
  # exact 0.020 m sphere radius for end_effector.
  radii: list[float] = []
  for link_name, gname in PIPER_COLLISION_LINK_GEOMS:
    gid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, gname)
    gtype = int(mj_model.geom_type[gid])
    if gtype == int(mujoco.mjtGeom.mjGEOM_SPHERE):
      radii.append(float(mj_model.geom_size[gid, 0]))
    elif link_name == "link6":
      radii.append(0.035)
    elif gtype == int(mujoco.mjtGeom.mjGEOM_MESH):
      mid = int(mj_model.geom_dataid[gid])
      adr = int(mj_model.mesh_vertadr[mid])
      num = int(mj_model.mesh_vertnum[mid])
      verts = mj_model.mesh_vert[adr : adr + num]
      half_ext = 0.5 * (verts.max(axis=0) - verts.min(axis=0))
      radii.append(float(half_ext.mean()))
    else:
      radii.append(float(mj_model.geom_rbound[gid]))
  return tuple(box_mins), tuple(box_maxs), tuple(radii)


def piper_body_aabbs_link0(
  xml_path: str = str(GO2_PIPER_XML),
) -> tuple[tuple[tuple[float, float, float], ...], tuple[tuple[float, float, float], ...]]:
  """Tight compiled MuJoCo AABBs ``(mins, maxs)`` in ``link0`` frame for battery, base, head/D415."""
  bmin, bmax, _ = _compiled_collision_proxy(xml_path)
  return bmin, bmax


def piper_tight_collision_radii(
  xml_path: str = str(GO2_PIPER_XML),
) -> dict[str, float]:
  """Tight principal collision radii for ``link4..end_effector`` calibrated from compiled MJCF."""
  _, _, rad = _compiled_collision_proxy(xml_path)
  return dict(zip(_COLLISION_LINK_ORDER, rad, strict=True))


def safe_norm(x: torch.Tensor, eps: float = 1e-6, dim: int = -1) -> torch.Tensor:
  """Smooth L2 norm ``sqrt(||x||^2 + eps)`` with finite non-NaN gradient at 0."""
  return torch.sqrt((x * x).sum(dim=dim) + eps)


def safe_abs(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
  """Smooth absolute value ``sqrt(x^2 + eps)`` with finite non-NaN gradient at 0."""
  return torch.sqrt(x * x + eps)


def quat_geodesic_angle(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
  """Sign-invariant, singularity-free WXYZ quaternion geodesic angle [rad].

  Computes ``2 * atan2(sqrt(||q_v||^2 + 1e-8), sqrt(q_w^2 + 1e-8))`` on the
  relative quaternion ``q1 * conj(q2)``, avoiding ``acos`` boundary singularities.
  """
  q2_conj = torch.cat([q2[..., :1], -q2[..., 1:]], dim=-1)
  w1, x1, y1, z1 = q1.unbind(dim=-1)
  w2, x2, y2, z2 = q2_conj.unbind(dim=-1)
  qw = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
  qx = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
  qy = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
  qz = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
  qv_norm = torch.sqrt(qx * qx + qy * qy + qz * qz + 1e-8)
  qw_abs = torch.sqrt(qw * qw + 1e-8)
  return 2.0 * torch.atan2(qv_norm, qw_abs)


def soft_contact_prob(
  z_foot: torch.Tensor,
  contact_height: float = 0.0255,
  tau: float = 0.0030,
) -> torch.Tensor:
  """Smooth foot contact probability ``sigmoid((h_contact - z_foot) / tau)``."""
  return torch.sigmoid((contact_height - z_foot) / tau)


def _make_tukey25_window(
  horizon: int,
  device: torch.device,
  dtype: torch.dtype,
) -> torch.Tensor:
  """Unit-L2-norm Tukey window (alpha=0.25) over ``horizon`` steps."""
  idx = torch.arange(horizon, device=device, dtype=dtype)
  u = (idx + 0.5) / max(horizon, 1)
  w = torch.ones_like(u)
  left = u < 0.125
  right = u > 0.875
  w[left] = 0.5 * (1.0 - torch.cos(math.pi * u[left] / 0.125))
  w[right] = 0.5 * (1.0 - torch.cos(math.pi * (1.0 - u[right]) / 0.125))
  return w / torch.linalg.norm(w).clamp_min(1e-8)


def _make_fourier_basis(
  f_min: float,
  f_max: float,
  df: float,
  horizon: int,
  dt: float,
  device: torch.device,
  dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Midpoint-quadrature Parseval real cosine/sine projection matrices ``(H, K)``."""
  n_bins = int(round((f_max - f_min) / df))
  freqs = f_min + (torch.arange(n_bins, device=device, dtype=dtype) + 0.5) * df
  t = torch.arange(horizon, device=device, dtype=dtype).unsqueeze(1) * dt
  ang = 2.0 * math.pi * t * freqs.unsqueeze(0)
  scale = math.sqrt(2.0 * dt * df)
  return scale * torch.cos(ang), scale * torch.sin(ang), freqs


def box_sdf_penetration(
  points: torch.Tensor,
  radii: torch.Tensor,
  box_min: torch.Tensor | tuple[float, ...] = LINK0_BODY_BOX_MIN,
  box_max: torch.Tensor | tuple[float, ...] = LINK0_BODY_BOX_MAX,
) -> torch.Tensor:
  """Differentiable sphere-vs-AABB(s) penetration depth ``relu(r - min_m sdf_m)``.

  Accepts either a single box ``(3,)`` or a union of ``M`` boxes ``(M, 3)``.
  Returns ``0.0`` strictly outside ``box + radius``, and increases smoothly
  inside ``box + radius`` with gradient pushing spheres away from the nearest box.
  ``points`` has shape ``(..., K, 3)`` and ``radii`` has shape ``(K,)``.
  """
  bmin = (
    box_min.to(device=points.device, dtype=points.dtype)
    if torch.is_tensor(box_min)
    else points.new_tensor(box_min)
  )
  bmax = (
    box_max.to(device=points.device, dtype=points.dtype)
    if torch.is_tensor(box_max)
    else points.new_tensor(box_max)
  )
  multi = bmin.ndim == 2
  pts = points.unsqueeze(-2) if multi else points
  center = 0.5 * (bmin + bmax)
  half = 0.5 * (bmax - bmin)
  d = torch.sqrt((pts - center).square() + 1e-8) - half
  outside = torch.sqrt(F.relu(d).square().sum(dim=-1) + 1e-12) - 1e-6
  u = F.relu(-d)
  ux, uy, uz = u.unbind(dim=-1)
  num = ux * uy * uz
  den = torch.sqrt((ux * uy).square() + (uy * uz).square() + (uz * ux).square() + 1e-16)
  inside = -(num / den)
  sdf = outside + inside
  if multi:
    sdf = sdf.amin(dim=-1)
  return F.relu(radii - sdf)


@dataclass(kw_only=True)
class WbcRewardWeights:
  """Per-term weights for differentiable WBC imagination (multiplied by dt)."""

  # Task tracking
  track_lin_vel_xy: float = 3.0
  track_ang_vel_z: float = 1.0
  ee_pos_exp: float = 3.0
  ee_ori_huber: float = -1.5
  # Base stability
  track_base_height_exp: float = 1.0
  flat_orientation_l2: float = -2.0
  lin_vel_z_l2: float = -1.0
  ang_vel_xy_l2: float = -0.05
  # Differentiable gait
  feet_slide_world_soft: float = -0.80
  feet_clearance_soft: float = -25.0
  standing_feet_air_soft: float = -1.0
  leg_joint_pos: float = -0.1
  diagonal_trot_soft: float = -0.25
  joint_mirror: float = -0.05
  contact_switch_tv: float = 0.0
  contact_switch_l2: float = 0.0
  # Route 2B window-level gait shaping
  gait_bandpass: float = 1.20
  gait_peak_lse: float = -220.0
  gait_diag_corr: float = 0.45
  # Smoothness & safety
  action_rate_leg_l2: float = -0.03
  action_rate_arm_l2: float = -0.02
  leg_joint_acc_l2: float = -2.5e-7
  joint_power: float = -2.0e-4
  joint_torques_l2: float = -2.0e-5
  dof_pos_limits: float = -1.0
  arm_vel_hinge: float = -1.0
  self_collision_penetration: float = -20.0
  # Soft termination penalty
  terminated: float = -200.0


@dataclass(kw_only=True)
class WbcImagineCfg:
  """Configuration for ``WbcImagination`` rollouts and smooth reward/termination."""

  horizon: int = 32
  gamma: float = 0.99
  lam: float = 0.95
  dt: float = 0.02
  lin_vel_kernel: Literal["smooth_l1", "exp"] = "smooth_l1"
  lin_vel_sigma: float = 0.5
  lin_vel_eps: float = 1e-3
  ang_vel_kernel: Literal["smooth_l1", "exp"] = "exp"
  ang_vel_sigma: float = 0.35
  ang_vel_eps: float = 1e-3
  ee_pos_sigma: float = 0.25
  ee_ori_delta: float = 0.5
  base_height_target: float = 0.30
  base_height_sigma: float = 0.05
  contact_height_m: float = 0.0255
  contact_tau_m: float = 0.0030
  foot_clearance_target: float = 0.065
  gait_eps_energy: float = 4.0e-5
  gait_sub_band_weight: float = 1.15
  gait_high_band_weight: float = 0.55
  gait_peak_beta: float = 180.0
  gait_peak_lme_target_m: float = 0.056
  cmd_lin_thresh: float = 0.05
  cmd_ang_thresh: float = 0.05
  arm_vel_limit: float = 2.85
  fall_angle_rad: float = 1.0
  fall_softness_rad: float = math.radians(5.0)
  base_height_term_m: float = 0.16
  base_height_term_tau: float = 0.015
  enable_collision_term: bool = True
  col_pen_margin: float = 0.015
  col_tau: float = 0.004
  action_clip_leak: float = 0.1
  obs_noise: bool = True
  sensor_grad: bool = True
  compile: bool = False
  weights: WbcRewardWeights = field(default_factory=WbcRewardWeights)


class WbcImagination:
  """Differentiable H-step WBC imagination rollout through a frozen WorldModel."""

  def __init__(
    self,
    cfg: WbcImagineCfg,
    consts: WbcRobotConsts,
    device: str | torch.device,
    *,
    scan_dim: int = 0,
    field: tr.HeightField | None = None,
  ):
    self.cfg = cfg
    self.c = consts
    self.device = torch.device(device)
    self.scan_dim = int(scan_dim)
    self.field = field
    self.critic_dim = WBC_CRITIC_BASE_DIM + self.scan_dim

    box_mins, box_maxs, radii = _compiled_collision_proxy()
    with torch.inference_mode(False):
      self._col_box_min = torch.tensor(box_mins, device=self.device, dtype=torch.float32)
      self._col_box_max = torch.tensor(box_maxs, device=self.device, dtype=torch.float32)
      self._col_radii = torch.tensor(radii, device=self.device, dtype=torch.float32)
      self._col_baseline = float(
        torch.sigmoid(torch.tensor(-cfg.col_pen_margin / cfg.col_tau)).item()
      )
      horizon = int(cfg.horizon)
      dt = float(cfg.dt)
      self._tukey_win = _make_tukey25_window(horizon, self.device, torch.float32)
      self._c_sub, self._s_sub, _ = _make_fourier_basis(
        0.00, 1.40, 0.20, horizon, dt, self.device, torch.float32
      )
      self._c_pass, self._s_pass, _ = _make_fourier_basis(
        2.00, 3.00, 0.20, horizon, dt, self.device, torch.float32
      )
      self._c_high, self._s_high, _ = _make_fourier_basis(
        4.40, 12.40, 0.40, horizon, dt, self.device, torch.float32
      )
      self._c_diag, self._s_diag, self._f_diag = _make_fourier_basis(
        0.50, 12.50, 0.25, horizon, dt, self.device, torch.float32
      )
      self._w_corr = torch.tensor(
        [
          [0.0, -1.0, -1.0, 1.0],
          [-1.0, 0.0, 1.0, -1.0],
          [-1.0, 1.0, 0.0, -1.0],
          [1.0, -1.0, -1.0, 0.0],
        ],
        device=self.device,
        dtype=torch.float32,
      ) / 12.0
    self._step_fn = torch.compile(self._step, dynamic=False) if cfg.compile else self._step

  def relief(
    self,
    pose: torch.Tensor,
    *,
    grad: bool = False,
  ) -> torch.Tensor | None:
    if self.scan_dim <= 0:
      return None
    return tr.relief(pose[..., :2], pose[..., 2], self.field, grad=grad)

  def rollout_constants(self, env_ids: torch.Tensor) -> dict[str, torch.Tensor | float | None]:
    """Gather per-rollout robot constants indexed by ``env_ids``."""
    c = self.c
    clip = c.action_clip[env_ids] if c.action_clip is not None else None
    return {
      "default_q": c.default_q[env_ids],
      "default_leg": c.default_q[env_ids, :WBC_NUM_LEG_JOINTS],
      "default_arm": c.default_arm[env_ids],
      "action_scale": c.action_scale[env_ids],
      "scale_leg": c.action_scale[env_ids, :WBC_NUM_LEG_JOINTS],
      "scale_arm": c.scale_arm[env_ids],
      "kp": c.kp,
      "kd": c.kd,
      "effort": c.effort,
      "soft_limits": c.soft_limits[env_ids],
      "arm_lower": c.arm_lower[env_ids],
      "arm_upper": c.arm_upper[env_ids],
      "max_arm_delta": c.max_arm_delta,
      "action_clip": clip,
    }

  def ee_mixed_fk(
    self,
    s: torch.Tensor,
    q_arm: torch.Tensor,
    ground_z: torch.Tensor | float | None = None,
    *,
    fk: PiperFkResult | None = None,
  ) -> tuple[torch.Tensor, PiperFkResult]:
    """Compute 7D mixed-frame FK pose ``[x_link0, y_link0, z_ee_w, qw, qx, qy, qz]``."""
    fk_res = piper_fk(q_arm) if fk is None else fk
    z_ee = ee_world_height(s[..., Z], s[..., GRAV], fk_res.p_ee_base, ground_z=ground_z)
    q_ee = fk_res.q_ee_link0_wxyz
    q_ee = torch.where(q_ee[..., :1] < 0.0, -q_ee, q_ee)
    ee_7d = torch.cat([fk_res.p_ee_link0[..., :2], z_ee.unsqueeze(-1), q_ee], dim=-1)
    return ee_7d, fk_res

  def critic_input(
    self,
    s: torch.Tensor,
    pose: torch.Tensor,
    q_ref_arm: torch.Tensor,
    cmd_vel: torch.Tensor,
    ee_pose_cmd: torch.Tensor,
    xi: torch.Tensor,
    k: dict,
    *,
    fk: PiperFkResult | None = None,
    ground_z: torch.Tensor | None = None,
    relief: torch.Tensor | None = None,
  ) -> torch.Tensor:
    """Privileged critic input: 114D on flat, +77D relief on micro-rough."""
    gz = tr.ground(pose[..., :2], self.field) if ground_z is None else ground_z
    q_arm = s[..., Q_ARM] + k["default_arm"]
    ee_rel_link0_fk, _ = self.ee_mixed_fk(s, q_arm, gz, fk=fk)
    parts = [s, q_ref_arm, cmd_vel, ee_pose_cmd, ee_rel_link0_fk, xi]
    if self.scan_dim > 0:
      rel = self.relief(pose, grad=False) if relief is None else relief
      assert rel is not None
      parts.append(rel)
    return torch.cat(parts, dim=-1)

  def soft_contact_and_foot_vel(
    self,
    s0: torch.Tensor,
    pose0: torch.Tensor,
    s1: torch.Tensor,
    pose1: torch.Tensor,
    dt: float | torch.Tensor,
    *,
    ground_z0: torch.Tensor | None = None,
    ground_z1: torch.Tensor | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute ``(p_contact, z_foot, v_foot_xy_w, v_foot_xy_norm, p_contact0)``."""
    gz0 = tr.ground(pose0[..., :2], self.field) if ground_z0 is None else ground_z0
    gz1 = tr.ground(pose1[..., :2], self.field) if ground_z1 is None else ground_z1
    p_foot_w0 = wbc_foot_positions_w(s0, pose0, ground_z=gz0)
    p_foot_w1 = wbc_foot_positions_w(s1, pose1, ground_z=gz1)
    foot_ground_z0 = tr.ground(p_foot_w0[..., :2], self.field)
    z_foot0 = p_foot_w0[..., 2] - foot_ground_z0
    p_contact0 = soft_contact_prob(z_foot0, self.cfg.contact_height_m, self.cfg.contact_tau_m)
    foot_ground_z = tr.ground(p_foot_w1[..., :2], self.field)
    z_foot = p_foot_w1[..., 2] - foot_ground_z
    p_contact = soft_contact_prob(z_foot, self.cfg.contact_height_m, self.cfg.contact_tau_m)
    dt_t = dt.unsqueeze(-1).unsqueeze(-1) if torch.is_tensor(dt) else dt
    v_foot_xy_w = (p_foot_w1[..., :2] - p_foot_w0[..., :2]) / dt_t
    v_foot_xy_norm = safe_norm(v_foot_xy_w, eps=1e-6, dim=-1)
    return p_contact, z_foot, v_foot_xy_w, v_foot_xy_norm, p_contact0

  def collision_penetration(self, fk: PiperFkResult) -> torch.Tensor:
    """Per-link tight Box-SDF penetration depths ``(..., 4)`` for ``link4..end_effector``."""
    pts = torch.stack([fk.link_pos_link0[name] for name in _COLLISION_LINK_ORDER], dim=-2)
    return box_sdf_penetration(pts, self._col_radii, self._col_box_min, self._col_box_max)

  def termination_prob(
    self,
    s: torch.Tensor,
    fk: PiperFkResult,
  ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Soft termination probability union ``d = 1 - (1-d_ori)(1-d_base)(1-d_col)``."""
    cfg = self.cfg
    grav = s[..., GRAV]
    g_hat = grav / safe_norm(grav, eps=1e-8, dim=-1).unsqueeze(-1)
    tilt = torch.acos((-g_hat[..., 2]).clamp(-1.0 + 1e-4, 1.0 - 1e-4))
    d_ori = torch.sigmoid((tilt - cfg.fall_angle_rad) / cfg.fall_softness_rad)
    d_base = torch.sigmoid((cfg.base_height_term_m - s[..., 0]) / cfg.base_height_term_tau)

    pens = self.collision_penetration(fk)
    max_pen = pens.amax(dim=-1)
    if cfg.enable_collision_term:
      raw_col = torch.sigmoid((max_pen - cfg.col_pen_margin) / cfg.col_tau)
      d_col = ((raw_col - self._col_baseline) / (1.0 - self._col_baseline)).clamp(0.0, 1.0)
    else:
      d_col = torch.zeros_like(d_ori)

    d = 1.0 - (1.0 - d_ori) * (1.0 - d_base) * (1.0 - d_col)
    return d, pens, (d_ori, d_base, d_col)

  def compute_rewards(
    self,
    s0: torch.Tensor,
    s1: torch.Tensor,
    pose0: torch.Tensor,
    pose1: torch.Tensor,
    a: torch.Tensor,
    a_prev: torch.Tensor,
    q_ref_arm: torch.Tensor,
    cmd_vel: torch.Tensor,
    ee_pose_cmd: torch.Tensor,
    xi: torch.Tensor,
    k: dict,
    *,
    fk1: PiperFkResult | None = None,
    ground_z0: torch.Tensor | None = None,
    ground_z1: torch.Tensor | None = None,
    pens1: torch.Tensor | None = None,
  ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute differentiable per-step WBC rewards ``(total * dt, terms)``."""
    total, terms, _, _ = self._compute_rewards_and_gait(
      s0,
      s1,
      pose0,
      pose1,
      a,
      a_prev,
      q_ref_arm,
      cmd_vel,
      ee_pose_cmd,
      xi,
      k,
      fk1=fk1,
      ground_z0=ground_z0,
      ground_z1=ground_z1,
      pens1=pens1,
    )
    return total, terms

  def _compute_rewards_and_gait(
    self,
    s0: torch.Tensor,
    s1: torch.Tensor,
    pose0: torch.Tensor,
    pose1: torch.Tensor,
    a: torch.Tensor,
    a_prev: torch.Tensor,
    q_ref_arm: torch.Tensor,
    cmd_vel: torch.Tensor,
    ee_pose_cmd: torch.Tensor,
    xi: torch.Tensor,
    k: dict,
    *,
    fk1: PiperFkResult | None = None,
    ground_z0: torch.Tensor | None = None,
    ground_z1: torch.Tensor | None = None,
    pens1: torch.Tensor | None = None,
  ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    cfg, w = self.cfg, self.cfg.weights
    gz0 = tr.ground(pose0[..., :2], self.field) if ground_z0 is None else ground_z0
    gz1 = tr.ground(pose1[..., :2], self.field) if ground_z1 is None else ground_z1

    q_arm1 = s1[..., Q_ARM] + k["default_arm"]
    ee_7d, fk_res = self.ee_mixed_fk(s1, q_arm1, gz1, fk=fk1)
    pens = self.collision_penetration(fk_res) if pens1 is None else pens1

    # 1. Task tracking
    v_xy = s1[..., V_YAW][..., :2]
    v_err_xy = v_xy - cmd_vel[..., :2]
    if cfg.lin_vel_kernel == "smooth_l1":
      cmd_xy_norm = safe_norm(cmd_vel[..., :2], eps=1e-6, dim=-1)
      err_xy_norm = torch.sqrt((v_err_xy * v_err_xy).sum(dim=-1) + cfg.lin_vel_eps**2)
      track_lin_vel_xy = cmd_xy_norm - err_xy_norm
    elif cfg.lin_vel_kernel == "exp":
      track_lin_vel_xy = torch.exp(
        -(v_err_xy * v_err_xy).sum(dim=-1) / (cfg.lin_vel_sigma**2)
      )
    else:
      raise ValueError(f"Unsupported lin_vel_kernel: {cfg.lin_vel_kernel}")

    wz_err = s1[..., W_B][..., 2] - cmd_vel[..., 2]
    if cfg.ang_vel_kernel == "exp":
      track_ang_vel_z = torch.exp(-(wz_err * wz_err) / (cfg.ang_vel_sigma**2))
    elif cfg.ang_vel_kernel == "smooth_l1":
      track_ang_vel_z = safe_abs(cmd_vel[..., 2], eps=1e-6) - torch.sqrt(
        wz_err * wz_err + cfg.ang_vel_eps**2
      )
    else:
      raise ValueError(f"Unsupported ang_vel_kernel: {cfg.ang_vel_kernel}")

    ee_pos_err = safe_norm(ee_7d[..., :3] - ee_pose_cmd[..., :3], eps=1e-6, dim=-1)
    ee_pos_exp = torch.exp(-ee_pos_err / cfg.ee_pos_sigma)

    ee_ori_err = quat_geodesic_angle(ee_7d[..., 3:7], ee_pose_cmd[..., 3:7])
    ee_ori_huber = torch.sqrt(1.0 + (ee_ori_err / cfg.ee_ori_delta).square()) - 1.0

    # 2. Base stability
    track_base_height_exp = torch.exp(
      -((s1[..., 0] - cfg.base_height_target).square()) / (cfg.base_height_sigma**2)
    )
    flat_orientation_l2 = s1[..., GRAV][..., :2].square().sum(dim=-1)
    lin_vel_z_l2 = s1[..., V_YAW][..., 2].square()
    ang_vel_xy_l2 = s1[..., W_B][..., :2].square().sum(dim=-1)

    # 3. Differentiable gait (world-frame FD foot velocity + B1 wz-gated moving mask)
    p_contact, z_foot, _, v_foot_xy_norm, p_contact0 = self.soft_contact_and_foot_vel(
      s0, pose0, s1, pose1, cfg.dt, ground_z0=gz0, ground_z1=gz1
    )
    lin_cmd_norm = safe_norm(cmd_vel[..., :2], eps=1e-12, dim=-1)
    moving_mask = (
      (lin_cmd_norm > cfg.cmd_lin_thresh) | (cmd_vel[..., 2].abs() > cfg.cmd_ang_thresh)
    ).to(dtype=s1.dtype)
    standing_mask = 1.0 - moving_mask
    lin_drive = F.relu(lin_cmd_norm - 0.20) / 0.30
    ang_drive = F.relu(cmd_vel[..., 2].abs() - 0.25) / 0.45
    moving_gate = moving_mask * torch.maximum(lin_drive, ang_drive).clamp(0.0, 1.0)

    dp_contact = p_contact - p_contact0
    eps_tv = 0.01
    contact_switch_tv_raw = (torch.sqrt(dp_contact.square() + eps_tv**2) - eps_tv).sum(dim=-1)
    contact_switch_tv = contact_switch_tv_raw * moving_gate
    contact_switch_l2 = dp_contact.square().sum(dim=-1) * moving_gate

    feet_slide_world_soft = (p_contact * v_foot_xy_norm).sum(dim=-1)
    feet_clearance_soft = (
      (1.0 - p_contact)
      * F.relu(cfg.foot_clearance_target - z_foot).square()
      * v_foot_xy_norm
    ).sum(dim=-1) * moving_mask
    standing_feet_air_soft = (1.0 - p_contact).sum(dim=-1) * standing_mask

    q_rel_leg = s1[..., Q_LEG]
    leg_joint_pos = (1.0 + 4.0 * standing_mask) * q_rel_leg.square().sum(dim=-1)

    p0, p1, p2, p3 = p_contact.unbind(dim=-1)
    diagonal_trot_soft = (
      (p0 - p3).square()
      + (p1 - p2).square()
      + (p0 + p1 - 1.0).square()
      + (p2 + p3 - 1.0).square()
    ) * moving_mask

    # FR[0:3], FL[3:6], RR[6:9], RL[9:12]: hips opposite sign in Q_LEG, thigh/calf same sign
    q_fr, q_fl, q_rr, q_rl = q_rel_leg.split(3, dim=-1)
    joint_mirror = (
      (q_fr[..., 0] + q_fl[..., 0]).square()
      + (q_fr[..., 1:] - q_fl[..., 1:]).square().sum(dim=-1)
      + (q_rr[..., 0] + q_rl[..., 0]).square()
      + (q_rr[..., 1:] - q_rl[..., 1:]).square().sum(dim=-1)
    )

    # 4. Smoothness & safety
    da = a - a_prev
    action_rate_leg_l2 = da[..., :WBC_NUM_LEG_JOINTS].square().sum(dim=-1)
    action_rate_arm_l2 = da[..., WBC_NUM_LEG_JOINTS:].square().sum(dim=-1)
    leg_joint_acc_l2 = ((s1[..., QD_LEG] - s0[..., QD_LEG]) / cfg.dt).square().sum(dim=-1)

    q = s1[..., Q] + k["default_q"]
    qd = s1[..., QD]
    q_ref_leg = k["default_leg"] + k["scale_leg"] * a[..., :WBC_NUM_LEG_JOINTS]
    q_ref = torch.cat([q_ref_leg, q_ref_arm], dim=-1)
    kp_scale = torch.cat(
      [
        xi[..., 5:6].expand(*xi.shape[:-1], WBC_NUM_LEG_JOINTS),
        xi[..., 7:8].expand(*xi.shape[:-1], WBC_NUM_ARM_JOINTS),
      ],
      dim=-1,
    )
    kd_scale = torch.cat(
      [
        xi[..., 6:7].expand(*xi.shape[:-1], WBC_NUM_LEG_JOINTS),
        xi[..., 8:9].expand(*xi.shape[:-1], WBC_NUM_ARM_JOINTS),
      ],
      dim=-1,
    )
    tau_raw = (k["kp"] * kp_scale) * (q_ref - q) - (k["kd"] * kd_scale) * qd
    tau = _leaky_clamp_ste(tau_raw, -k["effort"], k["effort"], leak=0.1)
    joint_torques_l2 = tau.square().sum(dim=-1)
    joint_power = torch.sqrt((tau * qd).square() + 1e-6).sum(dim=-1)

    lim = k["soft_limits"]
    dof_pos_limits = (F.relu(lim[..., 0] - q) + F.relu(q - lim[..., 1])).sum(dim=-1)

    qd_arm_abs = safe_abs(s1[..., QD_ARM], eps=1e-6)
    arm_vel_hinge = F.relu(qd_arm_abs - cfg.arm_vel_limit).square().sum(dim=-1)

    self_collision_penetration = pens.sum(dim=-1)

    terms = {
      "track_lin_vel_xy": track_lin_vel_xy,
      "track_ang_vel_z": track_ang_vel_z,
      "ee_pos_exp": ee_pos_exp,
      "ee_ori_huber": ee_ori_huber,
      "track_base_height_exp": track_base_height_exp,
      "flat_orientation_l2": flat_orientation_l2,
      "lin_vel_z_l2": lin_vel_z_l2,
      "ang_vel_xy_l2": ang_vel_xy_l2,
      "feet_slide_world_soft": feet_slide_world_soft,
      "feet_clearance_soft": feet_clearance_soft,
      "standing_feet_air_soft": standing_feet_air_soft,
      "leg_joint_pos": leg_joint_pos,
      "diagonal_trot_soft": diagonal_trot_soft,
      "joint_mirror": joint_mirror,
      "contact_switch_tv": contact_switch_tv,
      "contact_switch_l2": contact_switch_l2,
      "action_rate_leg_l2": action_rate_leg_l2,
      "action_rate_arm_l2": action_rate_arm_l2,
      "leg_joint_acc_l2": leg_joint_acc_l2,
      "joint_power": joint_power,
      "joint_torques_l2": joint_torques_l2,
      "dof_pos_limits": dof_pos_limits,
      "arm_vel_hinge": arm_vel_hinge,
      "self_collision_penetration": self_collision_penetration,
    }
    total = sum(getattr(w, name) * val for name, val in terms.items())
    with torch.no_grad():
      p_c_det = p_contact.detach()
      p_mid = ((p_c_det > 0.25) & (p_c_det < 0.75)).to(dtype=p_c_det.dtype).mean(dim=-1)
      gait_diag = torch.stack(
        [
          moving_mask.detach(),
          contact_switch_tv_raw.detach(),
          p_c_det.mean(dim=-1),
          p_mid,
          feet_slide_world_soft.detach(),
        ],
        dim=-1,
      )
    return total * cfg.dt, terms, gait_diag, z_foot

  def _step(
    self,
    s: torch.Tensor,
    pose: torch.Tensor,
    a_prev: torch.Tensor,
    q_ref_arm: torch.Tensor,
    hist: torch.Tensor,
    cmd_vel: torch.Tensor,
    ee_pose_cmd: torch.Tensor,
    traj_state: torch.Tensor,
    xi: torch.Tensor,
    enc_bias: torch.Tensor,
    k: dict,
    dt: torch.Tensor,
    actor,
    wm,
  ):
    cfg = self.cfg
    gz0 = tr.ground(pose[..., :2], self.field)
    crit_in = self.critic_input(
      s, pose, q_ref_arm, cmd_vel, ee_pose_cmd, xi, k, ground_z=gz0
    )

    a_raw = actor(hist)
    clip = k["action_clip"]
    if clip is not None:
      a = _leaky_clamp_ste(a_raw, clip[:, :, 0], clip[:, :, 1], cfg.action_clip_leak)
    else:
      a = a_raw

    q_ref_arm_next = slew_step_arm_diff(
      q_ref_arm,
      a[..., WBC_NUM_LEG_JOINTS:],
      k["default_arm"],
      k["scale_arm"],
      k["arm_lower"],
      k["arm_upper"],
      k["max_arm_delta"],
    )

    relief = self.relief(pose, grad=cfg.sensor_grad)
    s_next, dpose = wm(s, a, a_prev, xi, relief, q_ref_arm=q_ref_arm)
    pose_next = wbc_apply_pose_delta(pose, dpose)
    gz1 = tr.ground(pose_next[..., :2], self.field)

    traj_state_next, ee_pose_cmd_next = advance_ee_traj_state(
      traj_state, dt, make_quat_unique=True
    )

    fk_next = piper_fk(s_next[..., Q_ARM] + k["default_arm"])
    d, pens_next, _ = self.termination_prob(s_next, fk_next)
    r, _, gait_diag, z_foot = self._compute_rewards_and_gait(
      s,
      s_next,
      pose,
      pose_next,
      a,
      a_prev,
      q_ref_arm_next,
      cmd_vel,
      ee_pose_cmd_next,
      xi,
      k,
      fk1=fk_next,
      ground_z0=gz0,
      ground_z1=gz1,
      pens1=pens_next,
    )
    r = r + cfg.weights.terminated * d * cfg.dt

    # ``a`` is already clipped via _leaky_clamp_ste, so consts=None keeps STE grads intact.
    frame_next = wbc_obs_frame(
      s_next,
      a,
      cmd_vel,
      ee_pose_cmd_next,
      consts=None,
      add_noise=cfg.obs_noise,
      encoder_bias=enc_bias,
    )
    hist_next = wbc_push_history(hist, frame_next)
    return (
      s_next,
      pose_next,
      a,
      q_ref_arm_next,
      hist_next,
      ee_pose_cmd_next,
      traj_state_next,
      r,
      d,
      crit_in,
      gait_diag,
      z_foot,
    )

  def compute_window_gait_rewards(
    self,
    z_foot_seq: torch.Tensor,
    v_xy_seq: torch.Tensor | None,
    wz_seq: torch.Tensor | None,
    cmd_vel: torch.Tensor,
    surv_after_seq: torch.Tensor | None = None,
  ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute Route 2B window-level gait rate ``[/s]`` and diagnostics over ``[B, H, 4]``."""
    cfg = self.cfg
    w = cfg.weights
    B, H, _ = z_foot_seq.shape
    if H != int(self._tukey_win.shape[0]):
      raise ValueError(f"Expected horizon {int(self._tukey_win.shape[0])}, got {H}")
    dtype = z_foot_seq.dtype
    device = z_foot_seq.device

    lin_norm = torch.sqrt((cmd_vel[:, :2] * cmd_vel[:, :2]).sum(dim=-1) + 1e-12)
    wz_abs = cmd_vel[:, 2].abs()
    moving_mask = (
      (lin_norm > cfg.cmd_lin_thresh) | (wz_abs > cfg.cmd_ang_thresh)
    ).to(dtype=dtype)
    lin_drive = F.relu(lin_norm - cfg.cmd_lin_thresh) / 0.20
    ang_drive = F.relu(wz_abs - cfg.cmd_ang_thresh) / 0.30
    gait_gate = moving_mask * (
      0.35 + 0.65 * torch.maximum(lin_drive, ang_drive).clamp(0.0, 1.0)
    )

    if surv_after_seq is None:
      surv_det = torch.ones((B, H), device=device, dtype=dtype)
    else:
      surv_det = surv_after_seq.detach().to(device=device, dtype=dtype)

    win = self._tukey_win.to(device=device, dtype=dtype)
    w_raw = win.view(1, 1, H) * surv_det.unsqueeze(1)
    w_norm = w_raw / torch.sqrt(w_raw.square().sum(dim=-1, keepdim=True) + 1e-8)
    w_sum = w_norm.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    surv_mean = surv_det.mean(dim=-1)

    if v_xy_seq is not None and wz_seq is not None:
      s_w = surv_det / surv_det.sum(dim=-1, keepdim=True).clamp_min(1e-6)
      v_bar = (v_xy_seq * s_w.unsqueeze(-1)).sum(dim=1)
      wz_bar = (wz_seq * s_w).sum(dim=1)
      v_err_sq = (v_bar - cmd_vel[:, :2]).square().sum(dim=-1) + (
        0.24 * (wz_bar - cmd_vel[:, 2])
      ).square()
      v_cmd_eff = torch.sqrt(
        cmd_vel[:, :2].square().sum(dim=-1) + (0.24 * cmd_vel[:, 2]).square() + 1e-12
      )
      sigma_prog = 0.50 * v_cmd_eff + 0.08
      prog_gate = torch.exp(-v_err_sq / (sigma_prog * sigma_prog))
    else:
      prog_gate = torch.ones_like(gait_gate)

    zt = z_foot_seq.permute(0, 2, 1)  # [B, 4, H]
    mu_w = (zt * w_norm).sum(dim=-1, keepdim=True) / w_sum
    y = w_norm * (zt - mu_w)

    e_tot = y.square().sum(dim=-1)
    denom = e_tot + cfg.gait_eps_energy

    c_sub = self._c_sub.to(device=device, dtype=dtype)
    s_sub = self._s_sub.to(device=device, dtype=dtype)
    c_pass = self._c_pass.to(device=device, dtype=dtype)
    s_pass = self._s_pass.to(device=device, dtype=dtype)
    c_high = self._c_high.to(device=device, dtype=dtype)
    s_high = self._s_high.to(device=device, dtype=dtype)

    e_sub = (y @ c_sub).square().sum(dim=-1) + (y @ s_sub).square().sum(dim=-1)
    e_pass = (y @ c_pass).square().sum(dim=-1) + (y @ s_pass).square().sum(dim=-1)
    e_high = (y @ c_high).square().sum(dim=-1) + (y @ s_high).square().sum(dim=-1)

    rho_sub = (e_sub / denom).mean(dim=-1)
    rho_pass = (e_pass / denom).mean(dim=-1)
    rho_high = (e_high / denom).mean(dim=-1)

    gait_bandpass_raw = (
      rho_pass
      - cfg.gait_sub_band_weight * rho_sub
      - cfg.gait_high_band_weight * rho_high
    )
    gait_bandpass = (
      prog_gate * F.relu(gait_bandpass_raw) - F.relu(-gait_bandpass_raw)
    ) * gait_gate * surv_mean

    z_clamped = zt.clamp(min=-0.05, max=0.12)
    s_mask = surv_det.unsqueeze(1)
    z_for_peak = s_mask * z_clamped + (1.0 - s_mask) * 0.0205
    z_peak_lme = (
      torch.logsumexp(cfg.gait_peak_beta * z_for_peak, dim=-1) - math.log(H)
    ) / cfg.gait_peak_beta
    peak_deficit_sq = F.relu(cfg.gait_peak_lme_target_m - z_peak_lme).square().mean(dim=-1)
    gait_peak_lse = peak_deficit_sq * gait_gate * surv_mean

    y_hat = y / torch.sqrt(denom.unsqueeze(-1))
    gram = y_hat @ y_hat.transpose(-1, -2)
    w_corr = self._w_corr.to(device=device, dtype=dtype)
    diag_corr_raw = (gram * w_corr).sum(dim=(-1, -2))
    gait_diag_corr = (
      prog_gate * F.relu(diag_corr_raw) - F.relu(-diag_corr_raw)
    ) * gait_gate * surv_mean

    r_win_rate = (
      w.gait_bandpass * gait_bandpass
      + w.gait_peak_lse * gait_peak_lse
      + w.gait_diag_corr * gait_diag_corr
    )

    with torch.no_grad():
      c_diag = self._c_diag.to(device=device, dtype=dtype)
      s_diag = self._s_diag.to(device=device, dtype=dtype)
      f_diag = self._f_diag.to(device=device, dtype=dtype)
      psd = (y @ c_diag).square() + (y @ s_diag).square()
      psd_mean = psd.mean(dim=1)
      f_peak_hz = f_diag[psd_mean.argmax(dim=-1)]

    return r_win_rate, {
      "gait_bandpass": gait_bandpass,
      "gait_peak_lse": gait_peak_lse,
      "gait_diag_corr": gait_diag_corr,
      "rho_pass": rho_pass,
      "rho_sub": rho_sub,
      "rho_high": rho_high,
      "z_peak_lme": z_peak_lme.mean(dim=-1),
      "diag_corr_raw": diag_corr_raw,
      "f_peak_hz": f_peak_hz,
    }

  def rollout(
    self,
    wm,
    actor,
    critic,
    seed: dict[str, torch.Tensor],
  ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Imagine H steps from WBC replay seeds; freeze wm/critic params while keeping input grads."""
    cfg = self.cfg
    B = seed["s"].shape[0]
    env_ids = seed["env"]
    s = seed["s"]
    pose = seed["pose"]
    a_prev = seed["a_prev"]
    q_ref_arm = seed["q_ref_arm"]
    hist = seed["obs"]
    cmd_vel = seed["cmd_vel"]
    ee_pose_cmd = seed["ee_pose_cmd"]
    traj_state = seed["ee_traj_state"]
    xi = seed["xi"]
    enc_bias = seed.get(
      "enc_bias",
      torch.zeros((B, WBC_NUM_JOINTS), device=s.device, dtype=s.dtype),
    )
    if traj_state.shape[-1] != TRAJ_STATE_DIM:
      raise ValueError(
        f"Expected ee_traj_state trailing dim {TRAJ_STATE_DIM}, got {traj_state.shape[-1]}"
      )

    dt = torch.full((B,), cfg.dt, device=self.device, dtype=s.dtype)
    k = self.rollout_constants(env_ids)

    wm_params = list(wm.parameters())
    crit_params = list(critic.parameters())
    wm_req = [p.requires_grad for p in wm_params]
    crit_req = [p.requires_grad for p in crit_params]
    for p in wm_params:
      p.requires_grad_(False)
    for p in crit_params:
      p.requires_grad_(False)

    try:
      surv = torch.ones(B, device=self.device, dtype=s.dtype)
      ret = torch.zeros(B, device=self.device, dtype=s.dtype)
      disc = 1.0
      rewards, survs, conts, crit_in, gait_diags = [], [], [], [], []
      disc_survs, z_feet, v_xys, wzs, survs_after = [], [], [], [], []
      for _ in range(cfg.horizon):
        survs.append(surv)
        disc_survs.append(disc * surv)
        (
          s,
          pose,
          a_prev,
          q_ref_arm,
          hist,
          ee_pose_cmd,
          traj_state,
          r,
          d,
          x,
          gait_diag,
          z_foot,
        ) = self._step_fn(
          s,
          pose,
          a_prev,
          q_ref_arm,
          hist,
          cmd_vel,
          ee_pose_cmd,
          traj_state,
          xi,
          enc_bias,
          k,
          dt,
          actor,
          wm,
        )
        crit_in.append(x)
        gait_diags.append(gait_diag)
        z_feet.append(z_foot)
        v_xys.append(s[:, V_YAW][:, :2])
        wzs.append(s[:, W_B][:, 2])
        ret = ret + disc * surv * r
        rewards.append(r)
        c_t = 1.0 - d
        conts.append(c_t)
        surv = surv * c_t
        survs_after.append(surv)
        disc *= cfg.gamma

      r_win_rate, win_diag = self.compute_window_gait_rewards(
        torch.stack(z_feet, dim=1),
        torch.stack(v_xys, dim=1),
        torch.stack(wzs, dim=1),
        cmd_vel,
        torch.stack(survs_after, dim=1),
      )
      r_win_step = r_win_rate * cfg.dt
      w_surv = torch.stack(disc_survs, dim=1).sum(dim=1)
      ret = ret + w_surv * r_win_step
      x_H = self.critic_input(s, pose, q_ref_arm, cmd_vel, ee_pose_cmd, xi, k)
      v_H = critic(x_H).squeeze(-1)
      objective = ret + disc * surv * v_H
      rewards_stack = torch.stack(rewards, dim=1) + r_win_step.detach().unsqueeze(1)
      with torch.no_grad():
        gd = torch.stack(gait_diags, dim=1)
        mov = (gd[..., 0] > 0.5).to(dtype=gd.dtype)
        mov_count = mov.sum()
        denom = mov_count.clamp_min(1.0)
        has_mov = (mov_count > 0).to(dtype=gd.dtype)
        hz_scale = 1.0 / (8.0 * 0.85 * cfg.dt)  # ~= 7.353 at dt=0.02
        tv_moving_mean = (gd[..., 1] * mov).sum() / denom * has_mov
        f_implied_hz = tv_moving_mean * hz_scale
        p_moving_mean = (gd[..., 2] * mov).sum() / denom * has_mov
        p_mid_band_frac = (gd[..., 3] * mov).sum() / denom * has_mov
        feet_slide_raw = (gd[..., 4] * mov).sum() / denom * has_mov

        lin_cmd_norm = torch.sqrt((cmd_vel[:, :2] * cmd_vel[:, :2]).sum(dim=-1) + 1e-12)
        mov_env = (
          (lin_cmd_norm > cfg.cmd_lin_thresh) | (cmd_vel[:, 2].abs() > cfg.cmd_ang_thresh)
        ).to(dtype=gd.dtype)
        mov_env_count = mov_env.sum()
        env_denom = mov_env_count.clamp_min(1.0)
        has_mov_env = (mov_env_count > 0).to(dtype=gd.dtype)
        f_peak_hz = (win_diag["f_peak_hz"] * mov_env).sum() / env_denom * has_mov_env
        z_peak_lme_cm = (win_diag["z_peak_lme"] * 100.0 * mov_env).sum() / env_denom * has_mov_env
        diag_corr = (win_diag["diag_corr_raw"] * mov_env).sum() / env_denom * has_mov_env
        rho_pass = (win_diag["rho_pass"] * mov_env).sum() / env_denom * has_mov_env
      return objective, {
        "rewards": rewards_stack,
        "cont": torch.stack(conts, dim=1),
        "surv": torch.stack(survs + [surv], dim=1),
        "critic_in": torch.stack(crit_in + [x_H], dim=1),
        "imag_gait/f_implied_hz": f_implied_hz,
        "imag_gait/p_moving_mean": p_moving_mean,
        "imag_gait/p_mid_band_frac": p_mid_band_frac,
        "imag_gait/feet_slide_raw": feet_slide_raw,
        "imag_gait/f_peak_hz": f_peak_hz,
        "imag_gait/z_peak_lme_cm": z_peak_lme_cm,
        "imag_gait/diag_corr": diag_corr,
        "imag_gait/rho_pass": rho_pass,
      }
    finally:
      for p, req in zip(wm_params, wm_req, strict=True):
        p.requires_grad_(req)
      for p, req in zip(crit_params, crit_req, strict=True):
        p.requires_grad_(req)

  @torch.no_grad()
  def gae_targets(self, critic, data: dict[str, torch.Tensor]) -> torch.Tensor:
    """TD(lambda) returns over the imagined steps using exact per-step ``cont = 1 - d_t``."""
    cfg = self.cfg
    r = data["rewards"].detach()
    cont = data["cont"].detach()
    v = critic(data["critic_in"].detach()).squeeze(-1)
    H = r.shape[1]
    targets = torch.zeros_like(r)
    nxt = v[:, H]
    for t in reversed(range(H)):
      c_t = cont[:, t]
      nxt = r[:, t] + cfg.gamma * c_t * ((1.0 - cfg.lam) * v[:, t + 1] + cfg.lam * nxt)
      targets[:, t] = nxt
    return targets
