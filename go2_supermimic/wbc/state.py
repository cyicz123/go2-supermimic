"""Whole-body control (WBC) 82D state, analytic differentiable 6-DoF Piper FK,
210D differentiable observation reconstruction, and 9D DR context reader.

World-model state ``s`` (82D, translation- and yaw-invariant):
  [0:1]   Z      base height above ground under base: ``base_z_w - ground_h(base_xy)`` [m]
  [1:4]   GRAV   projected gravity in body frame
  [4:7]   V_YAW  base linear velocity in the yaw frame [m/s]
  [7:10]  W_B    base angular velocity in body frame [rad/s]
  [10:28] Q      joint positions relative to default (12 leg + 6 arm canonical)
  [28:46] QD     joint velocities (12 leg + 6 arm canonical) [rad/s]
  [46:58] HIPS   FR/FL/RR/RL hip (thigh) positions relative to base in yaw frame (4x3)
  [58:70] KNEES  FR/FL/RR/RL knee (calf) positions relative to base in yaw frame (4x3)
  [70:82] FEET   FR/FL/RR/RL foot site positions relative to base in yaw frame (4x3)

Coordinate conventions (matching ``go2_supermimic.supermimic.state``):
  - ``s[..., Z]`` stores base height *above the terrain directly under the base*
    (``base_z_w - ground_h(base_xy)``). To recover true world Z on non-flat terrain,
    add ``ground_z = ground(pose[..., :2], field)``.
  - ``HIPS``, ``KNEES``, and ``FEET`` store ``p_limb_w - p_base_w`` rotated into
    the base yaw frame. Thus ``s[..., FEET][..., 2] == z_foot_w - z_base_w``.
    On non-flat terrain, pass ``ground_z`` to ``wbc_foot_positions_w``,
    ``wbc_limb_points_world``, ``wbc_feet_world_z``, and ``ee_world_height`` to
    obtain exact world-frame Z coordinates.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import mujoco
import numpy as np
import torch

from go2_supermimic.robot.go2_piper import (
  ARM_JOINT_NAMES,
  CANONICAL_JOINT_NAMES,
  FOOT_GEOM_NAMES,
  FOOT_SITE_NAMES,
  GO2_PIPER_XML,
  LEG_JOINT_NAMES,
)
from go2_supermimic.supermimic.state import apply_pose_delta, pose_delta
from go2_supermimic.target.interface import yaw_rotate_inverse
from go2_supermimic.target.terrain import ground
from go2_supermimic.wbc.env_cfg import WBC_OBS_TERMS

##
# Dimension constants and state slices.
##

WBC_NUM_LEG_JOINTS: int = len(LEG_JOINT_NAMES)  # 12
WBC_NUM_ARM_JOINTS: int = len(ARM_JOINT_NAMES)  # 6
WBC_NUM_JOINTS: int = len(CANONICAL_JOINT_NAMES)  # 18
WBC_STATE_DIM: int = 82
WBC_POSE_DELTA_DIM: int = 3
WBC_Q_REF_ARM_DIM: int = WBC_NUM_ARM_JOINTS  # 6
WBC_CONTEXT_DIM: int = 9
WBC_OBS_FRAME_DIM: int = sum(dim for _, dim in WBC_OBS_TERMS)  # 70
WBC_ACTOR_HISTORY: int = 3
WBC_ACTOR_OBS_DIM: int = WBC_OBS_FRAME_DIM * WBC_ACTOR_HISTORY  # 210

# Slices into 82D state s.
Z = slice(0, 1)
GRAV = slice(1, 4)
V_YAW = slice(4, 7)
W_B = slice(7, 10)
Q = slice(10, 28)
Q_LEG = slice(10, 22)
Q_ARM = slice(22, 28)
QD = slice(28, 46)
QD_LEG = slice(28, 40)
QD_ARM = slice(40, 46)
HIPS = slice(46, 58)
KNEES = slice(58, 70)
FEET = slice(70, 82)

LEGS: tuple[str, ...] = FOOT_SITE_NAMES
FOOT_SITES: tuple[str, ...] = LEGS
HIP_BODIES: tuple[str, ...] = tuple(f"{leg}_thigh" for leg in LEGS)
KNEE_BODIES: tuple[str, ...] = tuple(f"{leg}_calf" for leg in LEGS)
FOOT_RADIUS: float = 0.022

PIPER_COLLISION_LINK_GEOMS: tuple[tuple[str, str], ...] = (
  ("link4", "link4_collision"),
  ("link5", "link5_collision"),
  ("link6", "link6_collision"),
  ("end_effector", "end_effector_collision"),
)

# Observation noise half-widths applied BEFORE term scaling (matching mjlab
# ObservationManager.compute_group and wbc/env_cfg.py).
WBC_OBS_NOISE: dict[str, float] = {
  "base_ang_vel": 0.2,
  "projected_gravity": 0.05,
  "joint_pos": 0.01,
  "joint_vel": 1.5,
}
WBC_OBS_SCALES: dict[str, float] = {
  "base_ang_vel": 0.2,
  "projected_gravity": 1.0,
  "joint_pos": 1.0,
  "joint_vel": 0.05,
  "actions": 1.0,
  "velocity_commands": 1.0,
  "pos_commands": 1.0,
}

# Precomputed term slices in a single 70D frame.
_FRAME_TERM_SLICES: tuple[tuple[str, int, slice], ...] = tuple(
  (
    name,
    dim,
    slice(
      sum(d for _, d in WBC_OBS_TERMS[:idx]),
      sum(d for _, d in WBC_OBS_TERMS[: idx + 1]),
    ),
  )
  for idx, (name, dim) in enumerate(WBC_OBS_TERMS)
)


##
# State / pose helpers.
##


def _resolve_base_world_z(
  base_z_rel: torch.Tensor,
  ground_z: torch.Tensor | float | None,
  lead_shape: tuple[int, ...],
) -> torch.Tensor:
  """Add optional ``ground_z`` to ``base_z_rel`` ``(..., 1)`` -> ``(..., 1)``."""
  if ground_z is None:
    return base_z_rel
  gz = torch.as_tensor(ground_z, device=base_z_rel.device, dtype=base_z_rel.dtype)
  if gz.ndim == len(lead_shape):
    gz = gz.unsqueeze(-1)
  elif gz.ndim == 0:
    gz = gz.view(*([1] * len(lead_shape)), 1)
  return base_z_rel + gz


def _yaw_rel_to_world_points(
  rel_yaw: torch.Tensor,
  pose: torch.Tensor,
  base_z_w: torch.Tensor,
) -> torch.Tensor:
  """Transform yaw-frame limb offsets ``(..., 4, 3)`` into world 3D coordinates ``(..., 4, 3)``."""
  yaw = pose[..., 2]
  c = yaw.cos().unsqueeze(-1)
  si = yaw.sin().unsqueeze(-1)
  x = c * rel_yaw[..., 0] - si * rel_yaw[..., 1]
  y = si * rel_yaw[..., 0] + c * rel_yaw[..., 1]
  base = torch.cat([pose[..., :2], base_z_w], dim=-1).unsqueeze(-2)
  return base + torch.stack([x, y, rel_yaw[..., 2]], dim=-1)


def wbc_feet_world_z(
  s: torch.Tensor, ground_z: torch.Tensor | float | None = None
) -> torch.Tensor:
  """World-frame foot site heights ``(..., 4)`` from 82D state ``s``.

  ``s[..., Z]`` is base height above ground under the base; pass ``ground_z``
  (terrain height under ``pose[..., :2]``) on non-flat terrain for true world Z.
  """
  lead_shape = s.shape[:-1]
  base_z_w = _resolve_base_world_z(s[..., Z], ground_z, lead_shape)
  return base_z_w + s[..., FEET].reshape(*lead_shape, 4, 3)[..., 2]


def wbc_foot_positions_w(
  s: torch.Tensor,
  pose: torch.Tensor,
  ground_z: torch.Tensor | float | None = None,
) -> torch.Tensor:
  """Differentiable world-frame 3D positions ``(..., 4, 3)`` of the 4 feet.

  Reconstructs exact world coordinates from ``s[..., Z]``, ``s[..., FEET]``,
  planar ``pose=(x, y, yaw)``, and ``ground_z=ground(pose[..., :2], field)`` so
  finite-difference foot velocities match world-frame simulation velocities even
  over non-zero or varying terrain heights.
  """
  lead_shape = s.shape[:-1]
  base_z_w = _resolve_base_world_z(s[..., Z], ground_z, lead_shape)
  rel_feet = s[..., FEET].reshape(*lead_shape, 4, 3)
  return _yaw_rel_to_world_points(rel_feet, pose, base_z_w)


def wbc_limb_points_world(
  s: torch.Tensor,
  pose: torch.Tensor,
  ground_z: torch.Tensor | float | None = None,
) -> dict[str, torch.Tensor]:
  """World-frame hips, knees, and feet (each ``(..., 4, 3)``) from 82D state and planar pose."""
  lead_shape = s.shape[:-1]
  base_z_w = _resolve_base_world_z(s[..., Z], ground_z, lead_shape)
  out: dict[str, torch.Tensor] = {}
  for name, sl in (("hips", HIPS), ("knees", KNEES), ("feet", FEET)):
    rel = s[..., sl].reshape(*lead_shape, 4, 3)
    out[name] = _yaw_rel_to_world_points(rel, pose, base_z_w)
  return out


@dataclass
class WbcRobotConsts:
  """Per-env constants of the 18-DoF Go2+Piper robot and slew action mapping."""

  default_q: torch.Tensor  # (N, 18)
  action_scale: torch.Tensor  # (N, 18)
  kp: torch.Tensor  # (18,)
  kd: torch.Tensor  # (18,)
  effort: torch.Tensor  # (18,)
  soft_limits: torch.Tensor  # (N, 18, 2)
  joint_limits: torch.Tensor  # (N, 18, 2)
  arm_indices: torch.Tensor  # (6,)
  default_arm: torch.Tensor  # (N, 6)
  scale_arm: torch.Tensor  # (N, 6)
  arm_lower: torch.Tensor  # (N, 6)
  arm_upper: torch.Tensor  # (N, 6)
  max_arm_delta: float
  action_clip: torch.Tensor | None  # (N, 18, 2) or None


def wbc_robot_consts(env) -> WbcRobotConsts:
  """Extract per-env constants from a live 18-DoF Go2+Piper ``ManagerBasedRlEnv``."""
  robot = env.scene["robot"]
  term = env.action_manager.get_term("joint_pos")
  jids = term._target_ids
  names = term._target_names
  assert list(names) == list(CANONICAL_JOINT_NAMES), (
    f"expect canonical joint order, got {list(names)}"
  )
  assert jids.tolist() == list(range(WBC_NUM_JOINTS)), "expect actions in joint order"
  kp, kd, eff = [], [], []
  for n in names:
    for act in robot.cfg.articulation.actuators:
      if any(re.fullmatch(expr, n) for expr in act.target_names_expr):
        kp.append(float(act.stiffness))
        kd.append(float(act.damping))
        eff.append(float(act.effort_limit))
        break
    else:
      raise KeyError(f"no actuator for joint {n}")
  dev = env.device
  max_arm_delta = float(term.cfg.max_arm_vel) * float(term.cfg.policy_dt)
  with torch.inference_mode(False):
    scale = term._scale
    if not torch.is_tensor(scale):
      scale = torch.full((env.num_envs, WBC_NUM_JOINTS), float(scale), device=dev)
    else:
      scale = scale.detach().clone()
    default_q = robot.data.default_joint_pos.detach().clone()
    joint_limits = robot.data.joint_pos_limits.detach().clone()
    soft_limits = robot.data.soft_joint_pos_limits.detach().clone()
    arm_idx = torch.arange(WBC_NUM_LEG_JOINTS, WBC_NUM_JOINTS, device=dev, dtype=torch.long)
    clip = getattr(term, "_clip", None)
    clip_t = clip.detach().clone() if torch.is_tensor(clip) else None
    return WbcRobotConsts(
      default_q=default_q,
      action_scale=scale,
      kp=torch.tensor(kp, device=dev, dtype=torch.float32),
      kd=torch.tensor(kd, device=dev, dtype=torch.float32),
      effort=torch.tensor(eff, device=dev, dtype=torch.float32),
      soft_limits=soft_limits,
      joint_limits=joint_limits,
      arm_indices=arm_idx,
      default_arm=default_q[:, arm_idx],
      scale_arm=scale[:, arm_idx],
      arm_lower=joint_limits[:, arm_idx, 0],
      arm_upper=joint_limits[:, arm_idx, 1],
      max_arm_delta=max_arm_delta,
      action_clip=clip_t,
    )


_WBC_LIMB_IDS: dict[int, tuple[list[int], list[int], list[int]]] = {}


def wbc_limb_positions_w(robot) -> dict[str, torch.Tensor]:
  """World-frame hip, knee, and foot positions ``(N, 4, 3)`` ordered FR, FL, RR, RL."""
  key = id(robot)
  if key not in _WBC_LIMB_IDS:
    hips, _ = robot.find_bodies(HIP_BODIES, preserve_order=True)
    knees, _ = robot.find_bodies(KNEE_BODIES, preserve_order=True)
    feet, _ = robot.find_sites(FOOT_SITES, preserve_order=True)
    _WBC_LIMB_IDS[key] = (hips, knees, feet)
  hips, knees, feet = _WBC_LIMB_IDS[key]
  d = robot.data
  return {
    "hips": d.body_link_pos_w[:, hips],
    "knees": d.body_link_pos_w[:, knees],
    "feet": d.site_pos_w[:, feet],
  }


def wbc_robot_state(env, field=None) -> torch.Tensor:
  """Extract the 82D translation- and yaw-invariant world-model state from ``env``."""
  robot = env.scene["robot"]
  d = robot.data
  heading = d.heading_w
  v_yaw = yaw_rotate_inverse(d.root_link_lin_vel_w, heading)
  limbs = wbc_limb_positions_w(robot)
  rel = []
  for name in ("hips", "knees", "feet"):
    r = limbs[name] - d.root_link_pos_w.unsqueeze(1)
    rel.append(torch.stack([yaw_rotate_inverse(r[:, i], heading) for i in range(4)], 1))
  base_h = d.root_link_pos_w[:, 2:3] - ground(d.root_link_pos_w[:, :2], field).unsqueeze(-1)
  return torch.cat(
    [
      base_h,
      d.projected_gravity_b,
      v_yaw,
      d.root_link_ang_vel_b,
      d.joint_pos - d.default_joint_pos,
      d.joint_vel,
    ]
    + [r.reshape(-1, 12) for r in rel],
    dim=-1,
  )


def wbc_planar_pose(env) -> torch.Tensor:
  """Planar pose ``(N, 3)`` = ``[x, y, yaw]`` in world frame."""
  d = env.scene["robot"].data
  return torch.cat([d.root_link_pos_w[:, :2], d.heading_w.unsqueeze(-1)], dim=-1)


wbc_pose_delta = pose_delta
wbc_apply_pose_delta = apply_pose_delta


##
# Analytic differentiable 6-DoF Piper FK parsed directly from go2_piper.xml.
##


class PiperFkResult(NamedTuple):
  """Analytic differentiable 6-DoF Piper FK outputs."""

  p_ee_link0: torch.Tensor  # (..., 3)
  q_ee_link0_wxyz: torch.Tensor  # (..., 4)
  p_ee_base: torch.Tensor  # (..., 3)
  link_pos_link0: dict[str, torch.Tensor]  # link4, link5, link6, end_effector -> (..., 3)


@dataclass(frozen=True)
class _PiperChainData:
  base_to_link0_pos: tuple[float, float, float]
  base_to_link0_quat: tuple[float, float, float, float]
  link_pos: tuple[tuple[float, float, float], ...]  # link1..link6 (6)
  link_quat: tuple[tuple[float, float, float, float], ...]  # link1..link6 (6)
  joint_pos: tuple[tuple[float, float, float], ...]  # joint1..joint6 (6)
  joint_axis: tuple[tuple[float, float, float], ...]  # joint1..joint6 (6)
  link6_to_ee_pos: tuple[float, float, float]
  link6_to_ee_quat: tuple[float, float, float, float]
  collision_local_centers: dict[str, tuple[float, float, float]]
  collision_radii: dict[str, float]


def _parse_vec(text: str | None, default: tuple[float, ...]) -> np.ndarray:
  if text is None:
    return np.asarray(default, dtype=np.float64)
  vals = [float(x) for x in text.strip().split()]
  return np.asarray(vals, dtype=np.float64)


def _norm_quat_np(q: np.ndarray) -> np.ndarray:
  n = float(np.linalg.norm(q))
  return q / n if n > 0.0 else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def _quat_mul_np(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
  w1, x1, y1, z1 = q1
  w2, x2, y2, z2 = q2
  return np.array(
    [
      w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
      w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
      w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
      w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ],
    dtype=np.float64,
  )


def _quat_rotate_np(q: np.ndarray, v: np.ndarray) -> np.ndarray:
  w, x, y, z = q
  qv = np.array([x, y, z], dtype=np.float64)
  t = 2.0 * np.cross(qv, v)
  return v + w * t + np.cross(qv, t)


def _compose_transform_np(
  p1: np.ndarray, q1: np.ndarray, p2: np.ndarray, q2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
  return p1 + _quat_rotate_np(q1, p2), _norm_quat_np(_quat_mul_np(q1, q2))


def _invert_transform_np(p: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  q_inv = np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)
  return -_quat_rotate_np(q_inv, p), q_inv


@lru_cache(maxsize=4)
def _load_piper_chain(xml_path: str = str(GO2_PIPER_XML)) -> _PiperChainData:
  """Parse exact fixed transforms, joint axes, and compiled collision geom centers."""
  tree = ET.parse(Path(xml_path))
  root = tree.getroot()
  base_body = root.find(".//body[@name='base']")
  if base_body is None:
    raise ValueError(f"Missing 'base' body in {xml_path}")

  def find_child_body(parent: ET.Element, name: str) -> ET.Element:
    for child in parent.findall("body"):
      if child.attrib.get("name") == name:
        return child
    raise ValueError(f"Missing child body '{name}' under '{parent.attrib.get('name')}'")

  link0_body = find_child_body(base_body, "link0")
  p_l0 = _parse_vec(link0_body.attrib.get("pos"), (0.0, 0.0, 0.0))
  q_l0 = _norm_quat_np(_parse_vec(link0_body.attrib.get("quat"), (1.0, 0.0, 0.0, 0.0)))

  # Account for link0 site offset if present (0, 0, 0 & identity in go2_piper.xml).
  link0_site = link0_body.find("site[@name='link0']")
  if link0_site is not None:
    p_s0 = _parse_vec(link0_site.attrib.get("pos"), (0.0, 0.0, 0.0))
    q_s0 = _norm_quat_np(_parse_vec(link0_site.attrib.get("quat"), (1.0, 0.0, 0.0, 0.0)))
    p_base_to_l0site, q_base_to_l0site = _compose_transform_np(p_l0, q_l0, p_s0, q_s0)
    p_l0site_to_l0body, q_l0site_to_l0body = _invert_transform_np(p_s0, q_s0)
  else:
    p_base_to_l0site, q_base_to_l0site = p_l0, q_l0
    p_l0site_to_l0body = np.zeros(3, dtype=np.float64)
    q_l0site_to_l0body = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

  link_pos_list: list[tuple[float, float, float]] = []
  link_quat_list: list[tuple[float, float, float, float]] = []
  joint_pos_list: list[tuple[float, float, float]] = []
  joint_axis_list: list[tuple[float, float, float]] = []

  parent_body = link0_body
  for idx, joint_name in enumerate(ARM_JOINT_NAMES, start=1):
    link_body = find_child_body(parent_body, f"link{idx}")
    p_b = _parse_vec(link_body.attrib.get("pos"), (0.0, 0.0, 0.0))
    q_b = _norm_quat_np(_parse_vec(link_body.attrib.get("quat"), (1.0, 0.0, 0.0, 0.0)))
    if idx == 1:
      p_b, q_b = _compose_transform_np(p_l0site_to_l0body, q_l0site_to_l0body, p_b, q_b)
    j_el = link_body.find(f"joint[@name='{joint_name}']")
    if j_el is None:
      raise ValueError(f"Missing joint '{joint_name}' in body 'link{idx}'")
    j_pos = _parse_vec(j_el.attrib.get("pos"), (0.0, 0.0, 0.0))
    j_axis = _parse_vec(j_el.attrib.get("axis"), (0.0, 0.0, 1.0))
    j_axis = j_axis / np.linalg.norm(j_axis)
    link_pos_list.append(tuple(float(v) for v in p_b))
    link_quat_list.append(tuple(float(v) for v in q_b))
    joint_pos_list.append(tuple(float(v) for v in j_pos))
    joint_axis_list.append(tuple(float(v) for v in j_axis))
    parent_body = link_body

  ee_body = find_child_body(parent_body, "end_effector")
  p_ee = _parse_vec(ee_body.attrib.get("pos"), (0.0, 0.0, 0.0))
  q_ee = _norm_quat_np(_parse_vec(ee_body.attrib.get("quat"), (1.0, 0.0, 0.0, 0.0)))

  ee_site = ee_body.find("site[@name='end_effector']")
  if ee_site is not None:
    p_es = _parse_vec(ee_site.attrib.get("pos"), (0.0, 0.0, 0.0))
    q_es = _norm_quat_np(_parse_vec(ee_site.attrib.get("quat"), (1.0, 0.0, 0.0, 0.0)))
    p_l6_to_ee, q_l6_to_ee = _compose_transform_np(p_ee, q_ee, p_es, q_es)
    p_ee_site_to_body, q_ee_site_to_body = _invert_transform_np(p_es, q_es)
  else:
    p_l6_to_ee, q_l6_to_ee = p_ee, q_ee
    p_ee_site_to_body = np.zeros(3, dtype=np.float64)
    q_ee_site_to_body = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

  # Compile MJCF once to extract MuJoCo's exact mesh/primitive collision geom
  # centers (model.geom_pos) and bounding radii (model.geom_rbound) per body.
  mj_model = mujoco.MjSpec.from_file(xml_path).compile()
  collision_local_centers: dict[str, tuple[float, float, float]] = {}
  collision_radii: dict[str, float] = {}
  for link_name, geom_name in PIPER_COLLISION_LINK_GEOMS:
    gid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, link_name)
    if gid < 0 or bid < 0:
      raise ValueError(f"Missing geom '{geom_name}' or body '{link_name}' in {xml_path}")
    if int(mj_model.geom_bodyid[gid]) != bid:
      raise ValueError(f"Geom '{geom_name}' does not belong to body '{link_name}'")
    local_pos = np.asarray(mj_model.geom_pos[gid], dtype=np.float64)
    if link_name == "end_effector":
      # Transform from end_effector body frame to end_effector site frame.
      local_pos = p_ee_site_to_body + _quat_rotate_np(q_ee_site_to_body, local_pos)
    collision_local_centers[link_name] = tuple(float(v) for v in local_pos)
    if int(mj_model.geom_type[gid]) == int(mujoco.mjtGeom.mjGEOM_SPHERE):
      collision_radii[link_name] = float(mj_model.geom_size[gid, 0])
    else:
      collision_radii[link_name] = float(mj_model.geom_rbound[gid])

  return _PiperChainData(
    base_to_link0_pos=tuple(float(v) for v in p_base_to_l0site),
    base_to_link0_quat=tuple(float(v) for v in q_base_to_l0site),
    link_pos=tuple(link_pos_list),
    link_quat=tuple(link_quat_list),
    joint_pos=tuple(joint_pos_list),
    joint_axis=tuple(joint_axis_list),
    link6_to_ee_pos=tuple(float(v) for v in p_l6_to_ee),
    link6_to_ee_quat=tuple(float(v) for v in q_l6_to_ee),
    collision_local_centers=collision_local_centers,
    collision_radii=collision_radii,
  )


def piper_collision_radii(xml_path: str = str(GO2_PIPER_XML)) -> dict[str, float]:
  """Compiled MuJoCo collision bounding sphere radii for ``link4..end_effector``."""
  return dict(_load_piper_chain(xml_path).collision_radii)


def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
  """Multiply WXYZ quaternions ``(..., 4)`` smoothly with zero branches."""
  w1, x1, y1, z1 = q1.unbind(dim=-1)
  w2, x2, y2, z2 = q2.unbind(dim=-1)
  return torch.stack(
    [
      w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
      w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
      w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
      w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ],
    dim=-1,
  )


def _quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
  """Rotate vector ``v`` ``(..., 3)`` by unit WXYZ quaternion ``q`` ``(..., 4)``."""
  w = q[..., :1]
  xyz = q[..., 1:]
  t = 2.0 * torch.cross(xyz, v, dim=-1)
  return v + w * t + torch.cross(xyz, t, dim=-1)


@lru_cache(maxsize=16)
def _piper_chain_tensors(
  device_str: str, dtype: torch.dtype
) -> tuple[
  torch.Tensor,
  torch.Tensor,
  tuple[torch.Tensor, ...],
  tuple[torch.Tensor, ...],
  tuple[torch.Tensor, ...],
  tuple[torch.Tensor, ...],
  torch.Tensor,
  torch.Tensor,
  dict[str, torch.Tensor],
]:
  """Create cached FK constant tensors outside any outer ``torch.inference_mode()``."""
  chain = _load_piper_chain()
  dev = torch.device(device_str)
  # Always leave inference_mode when constructing cached constants so a first call
  # inside Trainer.collect()'s inference_mode never poisons later autograd BPTT.
  with torch.inference_mode(False), torch.no_grad():
    t = lambda seq: torch.tensor(seq, device=dev, dtype=dtype)  # noqa: E731
    return (
      t(chain.base_to_link0_pos),
      t(chain.base_to_link0_quat),
      tuple(t(p) for p in chain.link_pos),
      tuple(t(q) for q in chain.link_quat),
      tuple(t(jp) for jp in chain.joint_pos),
      tuple(t(ja) for ja in chain.joint_axis),
      t(chain.link6_to_ee_pos),
      t(chain.link6_to_ee_quat),
      {k: t(v) for k, v in chain.collision_local_centers.items()},
    )


def piper_fk(q_arm: torch.Tensor) -> PiperFkResult:
  """Analytic, C^inf differentiable 6-DoF Piper FK from ``go2_piper.xml``.

  Uses chained WXYZ quaternion multiplication (zero Shepperd branches, zero
  gimbal/antipodal singularities) and supports arbitrary leading batch
  dimensions ``(..., 6)`` including 1D ``(6,)``, 2D ``(B, 6)``, and 3D ``(T, B, 6)``.
  ``link_pos_link0`` returns the exact compiled MuJoCo collision geom centers
  of ``link4``, ``link5``, ``link6``, and ``end_effector`` in the ``link0`` frame.
  """
  if q_arm.shape[-1] != WBC_NUM_ARM_JOINTS:
    raise ValueError(f"Expected q_arm trailing dim 6, got {q_arm.shape[-1]}")

  (
    p_base_l0,
    q_base_l0,
    link_pos,
    link_quat,
    joint_pos,
    joint_axis,
    p_l6_ee,
    q_l6_ee,
    col_local_centers,
  ) = _piper_chain_tensors(str(q_arm.device), q_arm.dtype)

  lead_shape = q_arm.shape[:-1]
  p = torch.zeros((*lead_shape, 3), device=q_arm.device, dtype=q_arm.dtype)
  q = torch.zeros((*lead_shape, 4), device=q_arm.device, dtype=q_arm.dtype)
  q[..., 0] = 1.0

  half = 0.5 * q_arm
  cos_h = torch.cos(half)
  sin_h = torch.sin(half)

  link_pos_link0: dict[str, torch.Tensor] = {}
  for idx in range(WBC_NUM_ARM_JOINTS):
    t_i = link_pos[idx].expand(*lead_shape, 3)
    q_fixed = link_quat[idx].expand(*lead_shape, 4)
    j_pos_i = joint_pos[idx].expand(*lead_shape, 3)
    axis_i = joint_axis[idx].expand(*lead_shape, 3)

    p = p + _quat_rotate(q, t_i)
    q = _quat_mul(q, q_fixed)

    c_i = cos_h[..., idx : idx + 1]
    s_i = sin_h[..., idx : idx + 1]
    q_joint = torch.cat([c_i, axis_i * s_i], dim=-1)

    # Rotate around joint pivot j_pos_i in link_i frame (no-op when j_pos_i == 0).
    p = p + _quat_rotate(q, j_pos_i)
    q = _quat_mul(q, q_joint)
    p = p - _quat_rotate(q, j_pos_i)

    link_name = f"link{idx + 1}"
    if link_name in col_local_centers:
      local_c = col_local_centers[link_name].expand(*lead_shape, 3)
      link_pos_link0[link_name] = p + _quat_rotate(q, local_c)

  p_ee_link0 = p + _quat_rotate(q, p_l6_ee.expand(*lead_shape, 3))
  q_ee_link0 = _quat_mul(q, q_l6_ee.expand(*lead_shape, 4))
  q_ee_norm = torch.sqrt((q_ee_link0 * q_ee_link0).sum(dim=-1, keepdim=True) + 1e-16)
  q_ee_link0 = q_ee_link0 / q_ee_norm

  p_ee_col_local = col_local_centers["end_effector"].expand(*lead_shape, 3)
  link_pos_link0["end_effector"] = p_ee_link0 + _quat_rotate(q_ee_link0, p_ee_col_local)

  p_ee_base = p_base_l0.expand(*lead_shape, 3) + _quat_rotate(
    q_base_l0.expand(*lead_shape, 4), p_ee_link0
  )
  return PiperFkResult(
    p_ee_link0=p_ee_link0,
    q_ee_link0_wxyz=q_ee_link0,
    p_ee_base=p_ee_base,
    link_pos_link0=link_pos_link0,
  )


def ee_world_height(
  base_z: torch.Tensor,
  grav_b: torch.Tensor,
  p_ee_base: torch.Tensor,
  ground_z: torch.Tensor | float | None = None,
) -> torch.Tensor:
  """Yaw-invariant, singularity-free EE world height from ``base_z``, ``grav_b``, ``p_ee_base``.

  ``base_z`` is base height above the ground under the base (``s[..., Z]``).
  Pass ``ground_z`` (terrain height under ``pose[..., :2]``) on non-flat terrain
  so ``base_z_w = base_z + ground_z`` and the returned height is in world Z.
  Supports arbitrary leading dimensions ``(...,)`` or ``(..., 1)`` for
  ``base_z`` and ``ground_z``.
  """
  if base_z.ndim == p_ee_base.ndim and base_z.shape[-1] == 1:
    base_z = base_z.squeeze(-1)
  base_z_w = base_z
  if ground_z is not None:
    gz = torch.as_tensor(ground_z, device=p_ee_base.device, dtype=p_ee_base.dtype)
    if gz.ndim == p_ee_base.ndim and gz.shape[-1] == 1:
      gz = gz.squeeze(-1)
    base_z_w = base_z_w + gz
  g_hat = grav_b / torch.sqrt((grav_b * grav_b).sum(dim=-1, keepdim=True) + 1e-8)
  return base_z_w + ((-g_hat) * p_ee_base).sum(dim=-1)


##
# Differentiable 1-step arm slew limiter (forward-exact to wbc.actions.slew_step_arm).
##


def _leaky_clamp_ste(
  x: torch.Tensor,
  lo: torch.Tensor | float,
  hi: torch.Tensor | float,
  leak: float,
) -> torch.Tensor:
  """Exact clamp in forward pass; leaky straight-through slope outside bounds."""
  hard = torch.clamp(x, min=lo, max=hi)
  soft = (1.0 - leak) * hard + leak * x
  return hard.detach() + (soft - soft.detach())


def slew_step_arm_diff(
  q_ref_prev: torch.Tensor,
  raw_arm: torch.Tensor,
  default_arm: torch.Tensor | float,
  scale_arm: torch.Tensor | float,
  arm_lower: torch.Tensor | float | None,
  arm_upper: torch.Tensor | float | None,
  max_delta: float,
  *,
  leak: float = 0.1,
) -> torch.Tensor:
  """Differentiable 50 Hz arm slew step matching ``wbc.actions.slew_step_arm`` forward.

  Uses leaky straight-through clamping so ``raw_arm`` and ``q_ref_prev`` retain
  finite, bounded, non-zero gradients across both ``±max_delta`` slew saturation
  and joint limit clamping while keeping the forward pass exact (< 1e-7 error).
  """
  max_d = float(max_delta)
  if max_d <= 0.0:
    raise ValueError(f"max_delta must be positive, got {max_delta}")
  if (arm_lower is None) ^ (arm_upper is None):
    raise ValueError("arm_lower and arm_upper must both be provided or both be None.")

  q_arm_cmd = default_arm + scale_arm * raw_arm
  if arm_lower is not None and arm_upper is not None:
    lower_t = torch.as_tensor(arm_lower, device=q_arm_cmd.device, dtype=q_arm_cmd.dtype)
    upper_t = torch.as_tensor(arm_upper, device=q_arm_cmd.device, dtype=q_arm_cmd.dtype)
    q_arm_cmd = _leaky_clamp_ste(q_arm_cmd, lower_t, upper_t, leak)
  else:
    lower_t = None
    upper_t = None

  dq = _leaky_clamp_ste(q_arm_cmd - q_ref_prev, -max_d, max_d, leak)
  q_next = q_ref_prev + dq
  if lower_t is not None and upper_t is not None:
    q_next = _leaky_clamp_ste(q_next, lower_t, upper_t, leak)
  return q_next


##
# 210D differentiable actor observation reconstruction.
##


def _uniform_noise(x: torch.Tensor, half_width: float) -> torch.Tensor:
  return x + (torch.rand_like(x) * 2.0 - 1.0) * half_width


def wbc_obs_frame(
  s: torch.Tensor,
  a_prev_raw: torch.Tensor,
  cmd_vel: torch.Tensor,
  ee_pose_cmd: torch.Tensor,
  consts: WbcRobotConsts | None = None,
  add_noise: bool = False,
  encoder_bias: torch.Tensor | None = None,
) -> torch.Tensor:
  """Reconstruct one 70D WBC actor observation frame ``(..., 70)`` from 82D state ``s``.

  Matches ``mjlab`` ``ObservationManager.compute_group`` ordering: noise is
  applied *before* per-term scaling (``0.2`` on ``base_ang_vel``, ``0.05`` on
  ``joint_vel``).
  """
  w_b = s[..., W_B]
  grav = s[..., GRAV]
  q_rel = s[..., Q] if encoder_bias is None else s[..., Q] + encoder_bias
  qd = s[..., QD]
  actions = a_prev_raw
  if consts is not None and consts.action_clip is not None:
    clip = consts.action_clip
    c_lo = clip[0, :, 0]
    c_hi = clip[0, :, 1]
    actions = torch.clamp(actions, min=c_lo, max=c_hi)

  if add_noise:
    w_b = _uniform_noise(w_b, WBC_OBS_NOISE["base_ang_vel"])
    grav = _uniform_noise(grav, WBC_OBS_NOISE["projected_gravity"])
    q_rel = _uniform_noise(q_rel, WBC_OBS_NOISE["joint_pos"])
    qd = _uniform_noise(qd, WBC_OBS_NOISE["joint_vel"])

  w_b = w_b * WBC_OBS_SCALES["base_ang_vel"]
  qd = qd * WBC_OBS_SCALES["joint_vel"]
  return torch.cat(
    [w_b, grav, q_rel, qd, actions, cmd_vel, ee_pose_cmd],
    dim=-1,
  )


def wbc_init_history(
  frame0: torch.Tensor, history: int = WBC_ACTOR_HISTORY
) -> torch.Tensor:
  """Initialize term-major flattened history ``(..., 210)`` by repeating ``frame0``.

  Matches ``mjlab`` ``CircularBuffer.append`` / ``backfill`` behavior on reset,
  where every history slot of each observation term is filled with the first frame.
  """
  if frame0.shape[-1] != WBC_OBS_FRAME_DIM:
    raise ValueError(f"Expected frame0 trailing dim {WBC_OBS_FRAME_DIM}, got {frame0.shape[-1]}")
  blocks = [frame0[..., sl].repeat(*([1] * (frame0.ndim - 1)), history) for _, _, sl in _FRAME_TERM_SLICES]
  return torch.cat(blocks, dim=-1)


def wbc_push_history(
  obs_hist: torch.Tensor,
  frame: torch.Tensor,
  history: int = WBC_ACTOR_HISTORY,
) -> torch.Tensor:
  """Shift term-major flattened history (oldest->newest) and append ``frame``.

  Accepts arbitrary leading dimensions ``(..., 210)`` and ``(..., 70)``.
  """
  if frame.shape[-1] != WBC_OBS_FRAME_DIM:
    raise ValueError(f"Expected frame trailing dim {WBC_OBS_FRAME_DIM}, got {frame.shape[-1]}")
  expected_hist = WBC_OBS_FRAME_DIM * history
  if obs_hist.shape[-1] != expected_hist:
    raise ValueError(f"Expected obs_hist trailing dim {expected_hist}, got {obs_hist.shape[-1]}")

  out = []
  offset = 0
  for _, dim, sl in _FRAME_TERM_SLICES:
    block_len = dim * history
    block = obs_hist[..., offset : offset + block_len]
    out.append(torch.cat([block[..., dim:], frame[..., sl]], dim=-1))
    offset += block_len
  return torch.cat(out, dim=-1)


##
# 9D physical domain randomization context reader.
##


class WbcContextReader:
  """Reads the per-env 9D physical DR context ``xi (B, 9)`` from ``ManagerBasedRlEnv``.

  Layout of ``xi``:
    [0]   mean foot slide friction (``geom_friction[..., 0]`` across ``FOOT_GEOM_NAMES``)
    [1:4] ``base`` COM offset relative to un-randomized ``mj_model.body_ipos[base]``
    [4]   ``base`` mass scale ``body_mass[base] / mj_model.body_mass[base]``
    [5]   mean leg Kp scale (``stiffness / default_stiffness`` over 12 leg joints)
    [6]   mean leg Kd scale (``damping / default_damping`` over 12 leg joints)
    [7]   mean arm Kp scale (``stiffness / default_stiffness`` over 6 arm joints)
    [8]   mean arm Kd scale (``damping / default_damping`` over 6 arm joints)
  """

  def __init__(self, env) -> None:
    self.env = env
    robot = env.scene["robot"]
    local_geoms, _ = robot.find_geoms(FOOT_GEOM_NAMES, preserve_order=True)
    local_body, _ = robot.find_bodies("base")
    self.base_body = int(robot.indexing.body_ids[local_body[0]])
    mj_model = env.sim.mj_model
    self.base_mass0 = float(mj_model.body_mass[self.base_body])

    leg_set = set(LEG_JOINT_NAMES)
    arm_set = set(ARM_JOINT_NAMES)
    self._leg_actuators: list[tuple[object, torch.Tensor]] = []
    self._arm_actuators: list[tuple[object, torch.Tensor]] = []
    with torch.inference_mode(False):
      self.foot_geoms = robot.indexing.geom_ids[local_geoms].detach().to(dtype=torch.long).clone()
      self.base_ipos0 = torch.as_tensor(
        mj_model.body_ipos[self.base_body], device=env.device, dtype=torch.float32
      ).clone()
      for act in robot.actuators:
        leg_idx = [i for i, n in enumerate(act.target_names) if n in leg_set]
        arm_idx = [i for i, n in enumerate(act.target_names) if n in arm_set]
        if leg_idx:
          self._leg_actuators.append(
            (act, torch.tensor(leg_idx, device=env.device, dtype=torch.long))
          )
        if arm_idx:
          self._arm_actuators.append(
            (act, torch.tensor(arm_idx, device=env.device, dtype=torch.long))
          )
    if not self._leg_actuators or not self._arm_actuators:
      raise RuntimeError("Missing leg or arm actuators on robot entity.")

  def _base_ipos(self) -> torch.Tensor:
    ipos = self.env.sim.model.body_ipos
    if ipos.ndim == 3:
      return ipos[:, self.base_body]
    return ipos[self.base_body].unsqueeze(0).expand(self.env.num_envs, 3)

  def _base_mass_scale(self) -> torch.Tensor:
    mass = self.env.sim.model.body_mass
    m = mass[:, self.base_body] if mass.ndim == 2 else mass[self.base_body].expand(self.env.num_envs)
    return (m / self.base_mass0).unsqueeze(-1)

  def _foot_friction(self) -> torch.Tensor:
    fr = self.env.sim.model.geom_friction
    if fr.ndim == 3:
      mu = fr[:, self.foot_geoms, 0].mean(dim=-1, keepdim=True)
    else:
      mu = fr[self.foot_geoms, 0].mean().expand(self.env.num_envs, 1)
    return mu

  @staticmethod
  def _mean_gain_scales(
    act_groups: list[tuple[object, torch.Tensor]],
  ) -> tuple[torch.Tensor, torch.Tensor]:
    kp_parts, kd_parts = [], []
    for act, idx in act_groups:
      kp_parts.append(act.stiffness[:, idx] / act.default_stiffness[:, idx])
      kd_parts.append(act.damping[:, idx] / act.default_damping[:, idx])
    kp_scale = torch.cat(kp_parts, dim=-1).mean(dim=-1, keepdim=True)
    kd_scale = torch.cat(kd_parts, dim=-1).mean(dim=-1, keepdim=True)
    return kp_scale, kd_scale

  def read(self) -> torch.Tensor:
    """Return per-env 9D DR context tensor ``(num_envs, 9)``."""
    mu = self._foot_friction()
    com_offset = self._base_ipos() - self.base_ipos0
    mass_scale = self._base_mass_scale()
    leg_kp, leg_kd = self._mean_gain_scales(self._leg_actuators)
    arm_kp, arm_kd = self._mean_gain_scales(self._arm_actuators)
    return torch.cat(
      [mu, com_offset, mass_scale, leg_kp, leg_kd, arm_kp, arm_kd],
      dim=-1,
    )
