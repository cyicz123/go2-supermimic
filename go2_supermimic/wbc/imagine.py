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
  contact_height: float = 0.032,
  tau: float = 0.006,
) -> torch.Tensor:
  """Smooth foot contact probability ``sigmoid((h_contact - z_foot) / tau)``."""
  return torch.sigmoid((contact_height - z_foot) / tau)


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
  track_lin_vel_xy: float = 2.0
  track_ang_vel_z: float = 1.0
  ee_pos_exp: float = 3.0
  ee_ori_huber: float = -1.5
  # Base stability
  track_base_height_exp: float = 1.0
  flat_orientation_l2: float = -2.0
  lin_vel_z_l2: float = -1.0
  ang_vel_xy_l2: float = -0.05
  # Differentiable gait
  feet_slide_world_soft: float = -0.5
  feet_clearance_soft: float = -10.0
  standing_feet_air_soft: float = -1.0
  leg_joint_pos: float = -0.1
  diagonal_trot_soft: float = -0.2
  joint_mirror: float = -0.05
  # Smoothness & safety
  action_rate_leg_l2: float = -0.02
  action_rate_arm_l2: float = -0.02
  joint_power: float = -2e-4
  joint_torques_l2: float = -1e-5
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
  ang_vel_sigma: float = 0.5
  ang_vel_eps: float = 1e-3
  ee_pos_sigma: float = 0.25
  ee_ori_delta: float = 0.5
  base_height_target: float = 0.30
  base_height_sigma: float = 0.05
  contact_height_m: float = 0.032
  contact_tau_m: float = 0.006
  foot_clearance_target: float = 0.06
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
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute ``(p_contact, z_foot, v_foot_xy_w, v_foot_xy_norm)``."""
    gz0 = tr.ground(pose0[..., :2], self.field) if ground_z0 is None else ground_z0
    gz1 = tr.ground(pose1[..., :2], self.field) if ground_z1 is None else ground_z1
    p_foot_w0 = wbc_foot_positions_w(s0, pose0, ground_z=gz0)
    p_foot_w1 = wbc_foot_positions_w(s1, pose1, ground_z=gz1)
    foot_ground_z = tr.ground(p_foot_w1[..., :2], self.field)
    z_foot = p_foot_w1[..., 2] - foot_ground_z
    p_contact = soft_contact_prob(z_foot, self.cfg.contact_height_m, self.cfg.contact_tau_m)
    dt_t = dt.unsqueeze(-1).unsqueeze(-1) if torch.is_tensor(dt) else dt
    v_foot_xy_w = (p_foot_w1[..., :2] - p_foot_w0[..., :2]) / dt_t
    v_foot_xy_norm = safe_norm(v_foot_xy_w, eps=1e-6, dim=-1)
    return p_contact, z_foot, v_foot_xy_w, v_foot_xy_norm

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
    p_contact, z_foot, _, v_foot_xy_norm = self.soft_contact_and_foot_vel(
      s0, pose0, s1, pose1, cfg.dt, ground_z0=gz0, ground_z1=gz1
    )
    lin_cmd_norm = safe_norm(cmd_vel[..., :2], eps=1e-12, dim=-1)
    moving_mask = (
      (lin_cmd_norm > cfg.cmd_lin_thresh) | (cmd_vel[..., 2].abs() > cfg.cmd_ang_thresh)
    ).to(dtype=s1.dtype)
    standing_mask = 1.0 - moving_mask

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
      "action_rate_leg_l2": action_rate_leg_l2,
      "action_rate_arm_l2": action_rate_arm_l2,
      "joint_power": joint_power,
      "joint_torques_l2": joint_torques_l2,
      "dof_pos_limits": dof_pos_limits,
      "arm_vel_hinge": arm_vel_hinge,
      "self_collision_penetration": self_collision_penetration,
    }
    total = sum(getattr(w, name) * val for name, val in terms.items())
    return total * cfg.dt, terms

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
    r, _ = self.compute_rewards(
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
    )

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
      rewards, survs, conts, crit_in = [], [], [], []
      for _ in range(cfg.horizon):
        survs.append(surv)
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
        ret = ret + disc * surv * r
        rewards.append(r)
        c_t = 1.0 - d
        conts.append(c_t)
        surv = surv * c_t
        disc *= cfg.gamma

      x_H = self.critic_input(s, pose, q_ref_arm, cmd_vel, ee_pose_cmd, xi, k)
      v_H = critic(x_H).squeeze(-1)
      objective = ret + disc * surv * v_H
      return objective, {
        "rewards": torch.stack(rewards, dim=1),
        "cont": torch.stack(conts, dim=1),
        "surv": torch.stack(survs + [surv], dim=1),
        "critic_in": torch.stack(crit_in + [x_H], dim=1),
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
