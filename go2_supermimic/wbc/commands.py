"""WBC velocity and mixed-frame end-effector pose commands with deterministic 20D trajectory rollout."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  axis_angle_from_quat,
  euler_xyz_from_quat,
  quat_apply,
  quat_apply_inverse,
  quat_conjugate,
  quat_from_euler_xyz,
  quat_mul,
  quat_unique,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

##
# Geometry & curriculum constants (WXYZ quaternion order, mixed frame).
##

STAGE1_ORI_ABS_MAX: tuple[float, float, float] = (
  math.radians(20.0),
  math.radians(20.0),
  math.radians(20.0),
)
STAGE2_ORI_ABS_MAX: tuple[float, float, float] = (
  math.radians(100.0),
  math.radians(65.0),
  math.radians(35.0),
)
ROLL_ABS_MAX = STAGE2_ORI_ABS_MAX[0]
PITCH_ABS_MAX = STAGE2_ORI_ABS_MAX[1]
YAW_ABS_MAX = STAGE2_ORI_ABS_MAX[2]

LINK0_BODY_BOX_MIN: tuple[float, float, float] = (-0.50, -0.24, -0.50)
"""Minimum corner of body + head D415 exclusion box in link0 local frame."""
LINK0_BODY_BOX_MAX: tuple[float, float, float] = (0.38, 0.24, 0.15)
"""Maximum corner of body + head D415 exclusion box in link0 local frame."""

TRAJECTORY_Z_WORLD_MIN: float = 0.05
TRAJECTORY_SAMPLE_COUNT: int = 12
_TRAJECTORY_ATTEMPTS: int = 8
_GOAL_SAMPLE_ATTEMPTS: int = 8
_R22_MIN: float = 0.05
SAFE_HOVER_POSITION: tuple[float, float, float] = (0.55, 0.0, 0.45)
_EULER_LIMIT_EPS: float = 1e-4

# 20D deterministic trajectory state layout:
#   [0:3]   p0 (start position: x_link0, y_link0, z_world)
#   [3:6]   p1 (quadratic Bezier control point)
#   [6:9]   p2 (goal position)
#   [9:13]  q0 (start quaternion relative to link0, WXYZ)
#   [13:17] q1 (goal quaternion relative to link0, WXYZ)
#   [17:18] elapsed (seconds)
#   [18:19] duration (seconds)
#   [19:20] is_traj (1.0 when active trajectory, 0.0 when static/finished)
TRAJ_STATE_DIM: int = 20


def _lerp_range(
  start: tuple[float, float], end: tuple[float, float], progress: float
) -> tuple[float, float]:
  p = max(0.0, min(1.0, float(progress)))
  return (
    (1.0 - p) * float(start[0]) + p * float(end[0]),
    (1.0 - p) * float(start[1]) + p * float(end[1]),
  )


##
# Base velocity command with B4 single-axis sampling and curriculum progress.
##


class WbcVelocityCommand(CommandTerm):
  """3D base velocity command ``[vx, vy, wz]`` with standing, single-axis, and curriculum support."""

  cfg: WbcVelocityCommandCfg

  def __init__(self, cfg: WbcVelocityCommandCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]
    self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
    self.is_standing_env = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self.is_single_axis_env = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)
    self.set_curriculum_progress(cfg.curriculum_progress)

  @property
  def command(self) -> torch.Tensor:
    return self.vel_command_b

  def set_curriculum_progress(self, progress: float) -> None:
    """Interpolate velocity bounds from ``initial_ranges`` to ``limit_ranges``."""
    p = max(0.0, min(1.0, float(progress)))
    self.cfg.curriculum_progress = p
    init_r = self.cfg.initial_ranges
    lim_r = self.cfg.limit_ranges
    self.cfg.ranges = WbcVelocityCommandCfg.Ranges(
      lin_vel_x=_lerp_range(init_r.lin_vel_x, lim_r.lin_vel_x, p),
      lin_vel_y=_lerp_range(init_r.lin_vel_y, lim_r.lin_vel_y, p),
      ang_vel_z=_lerp_range(init_r.ang_vel_z, lim_r.ang_vel_z, p),
    )

  def _update_metrics(self) -> None:
    max_time = max(float(self.cfg.resampling_time_range[1]), 1e-6)
    step_dt = float(getattr(self._env, "step_dt", 0.02))
    max_step = max(max_time / max(step_dt, 1e-6), 1.0)
    self.metrics["error_vel_xy"] += (
      torch.norm(
        self.vel_command_b[:, :2] - self.robot.data.root_link_lin_vel_b[:, :2],
        dim=-1,
      )
      / max_step
    )
    self.metrics["error_vel_yaw"] += (
      torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_link_ang_vel_b[:, 2])
      / max_step
    )

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    if len(env_ids) == 0:
      return
    ranges = self.cfg.ranges
    assert ranges is not None
    n = len(env_ids)
    r = torch.empty(n, device=self.device)
    sampled = torch.empty(n, 3, device=self.device)
    sampled[:, 0] = r.uniform_(*ranges.lin_vel_x)
    sampled[:, 1] = r.uniform_(*ranges.lin_vel_y)
    sampled[:, 2] = r.uniform_(*ranges.ang_vel_z)

    standing = torch.rand(n, device=self.device) < float(self.cfg.rel_standing_envs)
    single_axis = (~standing) & (
      torch.rand(n, device=self.device) < float(self.cfg.rel_single_axis_envs)
    )
    if bool(torch.any(single_axis)):
      sa_idx = single_axis.nonzero(as_tuple=False).flatten()
      axis = torch.randint(0, 3, (len(sa_idx),), device=self.device)
      mask = torch.zeros(len(sa_idx), 3, device=self.device)
      mask.scatter_(1, axis.unsqueeze(-1), 1.0)
      sampled[sa_idx] = sampled[sa_idx] * mask

    sampled[standing] = 0.0
    self.vel_command_b[env_ids] = sampled
    self.is_standing_env[env_ids] = standing
    self.is_single_axis_env[env_ids] = single_axis

  def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
    del env_ids
    if bool(torch.any(self.is_standing_env)):
      self.vel_command_b[self.is_standing_env] = 0.0


@dataclass(kw_only=True)
class WbcVelocityCommandCfg(CommandTermCfg):
  """Configuration for WBC 3D velocity commands."""

  @dataclass
  class Ranges:
    lin_vel_x: tuple[float, float]
    lin_vel_y: tuple[float, float]
    ang_vel_z: tuple[float, float]

  entity_name: str = "robot"
  resampling_time_range: tuple[float, float] = (4.0, 6.0)
  rel_standing_envs: float = 0.08
  rel_single_axis_envs: float = 0.20
  curriculum_progress: float = 0.0
  initial_ranges: Ranges = field(
    default_factory=lambda: WbcVelocityCommandCfg.Ranges(
      lin_vel_x=(-0.2, 0.2),
      lin_vel_y=(-0.2, 0.2),
      ang_vel_z=(-0.2, 0.2),
    )
  )
  limit_ranges: Ranges = field(
    default_factory=lambda: WbcVelocityCommandCfg.Ranges(
      lin_vel_x=(-1.0, 1.0),
      lin_vel_y=(-0.6, 0.6),
      ang_vel_z=(-1.0, 1.0),
    )
  )
  ranges: Ranges | None = None

  def __post_init__(self) -> None:
    if self.ranges is None:
      p = max(0.0, min(1.0, float(self.curriculum_progress)))
      self.ranges = WbcVelocityCommandCfg.Ranges(
        lin_vel_x=_lerp_range(self.initial_ranges.lin_vel_x, self.limit_ranges.lin_vel_x, p),
        lin_vel_y=_lerp_range(self.initial_ranges.lin_vel_y, self.limit_ranges.lin_vel_y, p),
        ang_vel_z=_lerp_range(self.initial_ranges.ang_vel_z, self.limit_ranges.ang_vel_z, p),
      )

  def build(self, env: ManagerBasedRlEnv) -> WbcVelocityCommand:
    return WbcVelocityCommand(self, env)


##
# Mixed-frame EE pose math & deterministic 20D trajectory helpers (WXYZ).
##


def clamp_wbc_euler(
  euler: torch.Tensor,
  ori_abs_max: tuple[float, float, float] = STAGE2_ORI_ABS_MAX,
) -> torch.Tensor:
  """Clamp roll/pitch/yaw per-axis to symmetric limits."""
  clamped = euler.clone()
  clamped[..., 0] = clamped[..., 0].clamp(-ori_abs_max[0], ori_abs_max[0])
  clamped[..., 1] = clamped[..., 1].clamp(-ori_abs_max[1], ori_abs_max[1])
  clamped[..., 2] = clamped[..., 2].clamp(-ori_abs_max[2], ori_abs_max[2])
  return clamped


def quintic_smoothstep(progress: torch.Tensor) -> torch.Tensor:
  """Zero-velocity, zero-acceleration quintic time profile ``10u^3 - 15u^4 + 6u^5``."""
  return progress.pow(3) * (10.0 + progress * (-15.0 + 6.0 * progress))


def quadratic_bezier(
  start: torch.Tensor,
  control: torch.Tensor,
  goal: torch.Tensor,
  curve_s: torch.Tensor,
) -> torch.Tensor:
  """Evaluate quadratic Bezier curve for arbitrary leading dimensions ``(..., 3)``.

  Supports ``curve_s`` of shape ``start.shape[:-1]`` (returning ``(..., 3)``) as
  well as ``(N, K)`` when ``start`` is ``(N, 3)`` (returning ``(N, K, 3)``).
  """
  if curve_s.shape == start.shape[:-1]:
    w = curve_s.unsqueeze(-1)
    one_minus = 1.0 - w
    return one_minus.square() * start + 2.0 * one_minus * w * control + w.square() * goal
  if curve_s.ndim == start.ndim and curve_s.shape[:-1] == start.shape[:-1]:
    w = curve_s.unsqueeze(-1)
    one_minus = 1.0 - w
    return (
      one_minus.square() * start.unsqueeze(-2)
      + 2.0 * one_minus * w * control.unsqueeze(-2)
      + w.square() * goal.unsqueeze(-2)
    )
  raise ValueError(
    f"Incompatible shapes for quadratic_bezier: start={tuple(start.shape)}, curve_s={tuple(curve_s.shape)}"
  )


def slerp_wxyz(
  start: torch.Tensor,
  goal: torch.Tensor,
  curve_s: torch.Tensor,
  *,
  make_unique: bool = True,
) -> torch.Tensor:
  """Shortest-arc SLERP on ``WXYZ`` quaternions with arbitrary leading dims and finite autograd.

  Masks near-collinear inputs (``|dot| > 1 - 1e-4``) before ``torch.acos`` so
  equal or antipodal quaternions never produce ``NaN`` gradients during backprop.
  """
  weight = curve_s if curve_s.ndim == start.ndim else curve_s.unsqueeze(-1)
  dot_raw = torch.sum(start * goal, dim=-1, keepdim=True)
  goal_aligned = torch.where(dot_raw < 0.0, -goal, goal)
  dot = dot_raw.abs().clamp(0.0, 1.0)
  nearly_parallel = dot > (1.0 - 1e-4)
  # Keep acos argument strictly away from +-1 even in the masked-out branch so
  # reverse-mode autograd never differentiates acos at |x|=1.
  safe_dot = torch.where(nearly_parallel, torch.zeros_like(dot), dot).clamp(0.0, 1.0 - 1e-4)
  theta = torch.acos(safe_dot)
  sin_theta = torch.sin(theta).clamp(min=1e-6)
  start_weight = torch.sin((1.0 - weight) * theta) / sin_theta
  goal_weight = torch.sin(weight * theta) / sin_theta
  blended = start_weight * start + goal_weight * goal_aligned
  linear = (1.0 - weight) * start + weight * goal_aligned
  output = torch.where(nearly_parallel, linear, blended)
  output = output / output.norm(dim=-1, keepdim=True).clamp(min=1e-8)
  if make_unique:
    output = quat_unique(output)
  return output


def sample_quadratic_bezier(
  start: torch.Tensor,
  control: torch.Tensor,
  goal: torch.Tensor,
  count: int = TRAJECTORY_SAMPLE_COUNT,
) -> torch.Tensor:
  """Uniformly sample ``count`` points along a quadratic Bezier curve."""
  curve_s = torch.linspace(0.0, 1.0, count, device=start.device, dtype=start.dtype)
  curve_s = curve_s.unsqueeze(0).expand(start.shape[0], count)
  return quadratic_bezier(start, control, goal, curve_s)


def link0_z_row(link0_quat_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Third row of the rotation matrix from link0 local frame to world frame (WXYZ)."""
  w, x, y, z = link0_quat_w.unbind(dim=-1)
  row_x = 2.0 * (x * z - w * y)
  row_y = 2.0 * (y * z + w * x)
  row_z = 1.0 - 2.0 * (x * x + y * y)
  return row_x, row_y, row_z


def mixed_curve_link0_points(
  points_mixed: torch.Tensor,
  link0_pos_w: torch.Tensor,
  link0_quat_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Convert mixed-frame points ``(x_link0, y_link0, z_world)`` into link0 local coords."""
  row_x, row_y, row_z = link0_z_row(link0_quat_w)
  row_x = row_x.unsqueeze(-1)
  row_y = row_y.unsqueeze(-1)
  row_z = row_z.unsqueeze(-1)
  singular = row_z.abs() < _R22_MIN
  denominator = torch.where(singular, torch.ones_like(row_z), row_z)
  z_local = (
    points_mixed[..., 2]
    - link0_pos_w[..., 2].unsqueeze(-1)
    - row_x * points_mixed[..., 0]
    - row_y * points_mixed[..., 1]
  ) / denominator
  z_local = torch.where(singular, torch.zeros_like(z_local), z_local)
  local = torch.stack((points_mixed[..., 0], points_mixed[..., 1], z_local), dim=-1)
  return local, points_mixed[..., 2], singular


def mixed_frame_world_position(
  points_mixed: torch.Tensor,
  link0_pos_w: torch.Tensor,
  link0_quat_w: torch.Tensor,
  local: torch.Tensor,
  singular: torch.Tensor,
) -> torch.Tensor:
  """Reconstruct world positions from mixed-frame points and link0 pose."""
  count = points_mixed.shape[1]
  quaternion = link0_quat_w.unsqueeze(1).expand(-1, count, -1).reshape(-1, 4)
  origin = link0_pos_w.unsqueeze(1).expand(-1, count, -1).reshape(-1, 3)
  world = (origin + quat_apply(quaternion, local.reshape(-1, 3))).reshape(
    points_mixed.shape
  )
  world[..., 2] = points_mixed[..., 2]
  singular_xyz = singular.unsqueeze(-1)
  fallback_xy = link0_pos_w.unsqueeze(1).expand(-1, count, -1)
  return torch.where(
    singular_xyz, torch.cat((fallback_xy[..., :2], world[..., 2:3]), dim=-1), world
  )


def trajectory_is_forbidden(
  points_link0: torch.Tensor,
  z_world: torch.Tensor,
  singular: torch.Tensor | None = None,
) -> torch.Tensor:
  """Check if any sampled point enters ``LINK0_BODY_BOX`` or drops below ``z_world_min``."""
  box_min = points_link0.new_tensor(LINK0_BODY_BOX_MIN)
  box_max = points_link0.new_tensor(LINK0_BODY_BOX_MAX)
  inside_box = ((points_link0 >= box_min) & (points_link0 <= box_max)).all(dim=-1)
  below_ground = z_world < TRAJECTORY_Z_WORLD_MIN
  rejected = inside_box | below_ground
  if singular is not None:
    outside_xy = (
      (points_link0[..., 0] < box_min[0])
      | (points_link0[..., 0] > box_max[0])
      | (points_link0[..., 1] < box_min[1])
      | (points_link0[..., 1] > box_max[1])
    )
    rejected = rejected | (singular & ~outside_xy)
  return rejected.any(dim=-1)


def orientation_path_exceeds_limits(
  start: torch.Tensor,
  goal: torch.Tensor,
  ori_abs_max: tuple[float, float, float] = STAGE2_ORI_ABS_MAX,
  count: int = TRAJECTORY_SAMPLE_COUNT,
) -> torch.Tensor:
  """Check whether SLERP path between WXYZ quaternions stays within per-axis Euler limits."""
  samples = torch.linspace(0.0, 1.0, count, device=start.device, dtype=start.dtype)
  eulers = []
  for idx in range(count):
    q = slerp_wxyz(start, goal, samples[idx].expand(start.shape[0]), make_unique=False)
    r, p, y = euler_xyz_from_quat(q)
    eulers.append(torch.stack((r, p, y), dim=-1))
  euler = torch.stack(eulers, dim=1)
  exceed = (
    (euler[..., 0].abs() > ori_abs_max[0] + _EULER_LIMIT_EPS)
    | (euler[..., 1].abs() > ori_abs_max[1] + _EULER_LIMIT_EPS)
    | (euler[..., 2].abs() > ori_abs_max[2] + _EULER_LIMIT_EPS)
  )
  return exceed.any(dim=-1)


def pack_ee_traj_state(
  p0: torch.Tensor,
  p1: torch.Tensor,
  p2: torch.Tensor,
  q0: torch.Tensor,
  q1: torch.Tensor,
  elapsed: torch.Tensor,
  duration: torch.Tensor,
  is_traj: torch.Tensor,
) -> torch.Tensor:
  """Pack trajectory parameters into a 20D tensor ``(..., 20)``."""
  if elapsed.ndim == p0.ndim - 1:
    elapsed = elapsed.unsqueeze(-1)
  if duration.ndim == p0.ndim - 1:
    duration = duration.unsqueeze(-1)
  if is_traj.ndim == p0.ndim - 1:
    is_traj = is_traj.unsqueeze(-1)
  return torch.cat(
    (p0, p1, p2, q0, q1, elapsed, duration, is_traj.to(dtype=p0.dtype)), dim=-1
  )


def unpack_ee_traj_state(
  traj_state: torch.Tensor,
) -> tuple[
  torch.Tensor,
  torch.Tensor,
  torch.Tensor,
  torch.Tensor,
  torch.Tensor,
  torch.Tensor,
  torch.Tensor,
  torch.Tensor,
]:
  """Unpack a 20D trajectory state tensor into ``(p0, p1, p2, q0, q1, elapsed, duration, is_traj)``."""
  if traj_state.shape[-1] != TRAJ_STATE_DIM:
    raise ValueError(
      f"Expected trailing dim {TRAJ_STATE_DIM}, got {traj_state.shape[-1]}"
    )
  return (
    traj_state[..., 0:3],
    traj_state[..., 3:6],
    traj_state[..., 6:9],
    traj_state[..., 9:13],
    traj_state[..., 13:17],
    traj_state[..., 17],
    traj_state[..., 18],
    traj_state[..., 19],
  )


def advance_ee_traj_state(
  traj_state: torch.Tensor,
  dt: float | torch.Tensor,
  *,
  make_quat_unique: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Deterministically advance 20D EE trajectory state by ``dt`` seconds.

  Supports arbitrary leading dimensions ``(..., 20)`` including 1D ``(20,)``.
  Handles ``duration <= 0`` by immediately settling at the goal without jumps,
  and preserves quaternion sign continuity across trajectory completion even
  when ``make_quat_unique=False``.
  """
  p0, p1, p2, q0, q1, elapsed, duration, is_traj = unpack_ee_traj_state(traj_state)
  dt_tensor = torch.as_tensor(dt, device=traj_state.device, dtype=traj_state.dtype)
  active = is_traj > 0.5
  nonpositive_dur = duration <= 0.0

  elapsed_next = torch.where(active, elapsed + dt_tensor, elapsed)
  safe_dur = torch.where(nonpositive_dur, torch.ones_like(duration), duration)
  raw_progress = (elapsed_next / safe_dur).clamp(0.0, 1.0)
  progress = torch.where(nonpositive_dur, torch.ones_like(raw_progress), raw_progress)
  curve_s = quintic_smoothstep(progress)

  traj_pos = quadratic_bezier(p0, p1, p2, curve_s)
  traj_quat = slerp_wxyz(q0, q1, curve_s, make_unique=make_quat_unique)

  # Align goal hemisphere to q0 so completed trajectories keep sign continuity
  # when make_quat_unique=False.
  dot_q0_q1 = torch.sum(q0 * q1, dim=-1, keepdim=True)
  q1_aligned = torch.where(dot_q0_q1 < 0.0, -q1, q1)
  q1_next = torch.where(active[..., None], q1_aligned, q1)
  static_quat = quat_unique(q1_next) if make_quat_unique else q1_next

  pos = torch.where(active[..., None], traj_pos, p2)
  quat = torch.where(active[..., None], traj_quat, static_quat)

  still_active = active & (~nonpositive_dur) & (elapsed_next < duration)
  next_is_traj = torch.where(
    still_active, torch.ones_like(is_traj), torch.zeros_like(is_traj)
  )
  next_state = pack_ee_traj_state(
    p0, p1, p2, q0, q1_next, elapsed_next, duration, next_is_traj
  )
  cmd_7d = torch.cat((pos, quat), dim=-1)
  return next_state, cmd_7d


##
# WbcEePoseCommand
##


class WbcEePoseCommand(CommandTerm):
  """Mixed-frame 7D EE pose command ``[x_link0, y_link0, z_w, qw, qx, qy, qz]`` (WXYZ).

  Defers reset-time resampling until the post-``sim.forward()`` ``compute()``
  call so ``link0`` poses and ``LINK0_BODY_BOX`` checks never observe stale
  pre-reset transforms.
  """

  cfg: WbcEePoseCommandCfg

  def __init__(self, cfg: WbcEePoseCommandCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]
    body_ids, _ = self.robot.find_bodies(cfg.body_name)
    link0_ids, _ = self.robot.find_bodies(cfg.link_name)
    if not body_ids or not link0_ids:
      raise ValueError(
        f"Missing body '{cfg.body_name}' or link0 '{cfg.link_name}' on entity '{cfg.entity_name}'."
      )
    self.body_idx = body_ids[0]
    self.link0_idx = link0_ids[0]

    self.pose_command = self._nominal_pose(self.num_envs)
    self._certified_pose = self.pose_command.clone()
    self._traj_state = torch.zeros(
      self.num_envs, TRAJ_STATE_DIM, device=self.device, dtype=self.pose_command.dtype
    )
    self._sync_static_traj_state(
      torch.arange(self.num_envs, device=self.device, dtype=torch.long),
      self.pose_command[:, :3],
      self.pose_command[:, 3:7],
    )
    self._pending_reset = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self._pending_step_dt: float | torch.Tensor | None = None
    self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)
    n_bins = int(cfg.adaptive_x_bins) * int(cfg.adaptive_z_bins)
    self._eval_env_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
    self._bin_id = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
    self._cmd_age = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
    self._bin_err = torch.full(
      (n_bins,), float(cfg.adaptive_init_err), dtype=torch.float32, device=self.device
    )
    self._bin_counts = torch.zeros(n_bins, dtype=torch.long, device=self.device)
    self.set_curriculum_progress(cfg.curriculum_progress)

  def set_eval_env_mask(self, mask: torch.Tensor | None) -> None:
    """Restrict adaptive bin error updates to non-exploration environments."""
    if mask is None:
      self._eval_env_mask.fill_(True)
      return
    mask_t = mask.to(device=self.device, dtype=torch.bool).reshape(-1)
    if mask_t.shape[0] != self.num_envs:
      raise ValueError(
        f"Expected eval env mask of length {self.num_envs}, got {mask_t.shape[0]}"
      )
    self._eval_env_mask.copy_(mask_t)

  def bin_probabilities(self) -> torch.Tensor:
    """Compute normalized 2D (x, z) bin sampling probabilities with uniform floor."""
    n_bins = int(self.cfg.adaptive_x_bins) * int(self.cfg.adaptive_z_bins)
    eps = float(self.cfg.adaptive_uniform_prob)
    if (not self.cfg.adaptive_sampling) or eps >= 1.0:
      return torch.full((n_bins,), 1.0 / float(n_bins), device=self.device, dtype=torch.float32)
    e_min = float(self.cfg.adaptive_err_clip[0])
    e_max = float(self.cfg.adaptive_err_clip[1])
    beta = float(self.cfg.adaptive_beta)
    weights = self._bin_err.clamp(min=e_min, max=e_max).pow(beta)
    w_sum = weights.sum().clamp(min=1e-12)
    return (1.0 - eps) * (weights / w_sum) + eps / float(n_bins)

  def adaptive_metrics(self) -> dict[str, float]:
    """Read-only summary metrics for TensorBoard and console logging."""
    n_x = int(self.cfg.adaptive_x_bins)
    n_z = int(self.cfg.adaptive_z_bins)
    probs_2d = self.bin_probabilities().view(n_x, n_z)
    err_2d = self._bin_err.view(n_x, n_z)
    return {
      "prob_low_z": float(probs_2d[:, 0].sum().item()),
      "prob_high_z": float(probs_2d[:, -1].sum().item()),
      "prob_max": float(probs_2d.max().item()),
      "err_low_z": float(err_2d[:, 0].mean().item()),
      "err_high_z": float(err_2d[:, -1].mean().item()),
    }

  def adaptive_state_dict(self) -> dict[str, torch.Tensor | int]:
    """Return adaptive sampling state for checkpoint persistence."""
    return {
      "x_bins": int(self.cfg.adaptive_x_bins),
      "z_bins": int(self.cfg.adaptive_z_bins),
      "bin_err": self._bin_err.detach().cpu().clone(),
      "bin_counts": self._bin_counts.detach().cpu().clone(),
    }

  def load_adaptive_state_dict(self, state: dict | None) -> None:
    """Restore adaptive sampling state when bin shape matches; ignore incompatible/missing."""
    if not isinstance(state, dict):
      return
    bin_err = state.get("bin_err")
    if not torch.is_tensor(bin_err) or bin_err.shape != self._bin_err.shape:
      return
    self._bin_err.copy_(bin_err.to(device=self.device, dtype=self._bin_err.dtype))
    bin_counts = state.get("bin_counts")
    if torch.is_tensor(bin_counts) and bin_counts.shape == self._bin_counts.shape:
      self._bin_counts.copy_(bin_counts.to(device=self.device, dtype=self._bin_counts.dtype))

  def _update_bin_errors(self, env_ids: torch.Tensor) -> None:
    """Update EMA bin steady-state position errors for eligible settled environments."""
    bins = self._bin_id[env_ids]
    errs = self.metrics["position_error"][env_ids].detach()
    valid = (bins >= 0) & (bins < self._bin_err.numel()) & torch.isfinite(errs)
    if not bool(torch.any(valid)):
      return
    bins = bins[valid]
    errs = errs[valid]
    alpha = float(self.cfg.adaptive_ema_alpha)
    for k in torch.unique(bins).tolist():
      k_int = int(k)
      mean_err = float(errs[bins == k_int].mean().item())
      self._bin_err[k_int] = (1.0 - alpha) * self._bin_err[k_int] + alpha * mean_err
      self._bin_counts[k_int] += 1

  @property
  def command(self) -> torch.Tensor:
    """Current 7D mixed-frame command with WXYZ quaternion."""
    return self.pose_command

  @property
  def traj_state(self) -> torch.Tensor:
    """Current 20D deterministic trajectory state."""
    return self._traj_state

  @property
  def _traj_active(self) -> torch.Tensor:
    return self._traj_state[:, 19] > 0.5

  def set_curriculum_progress(self, progress: float) -> None:
    """Interpolate workspace and orientation ranges from ``initial_ranges`` to ``limit_ranges``."""
    p = max(0.0, min(1.0, float(progress)))
    self.cfg.curriculum_progress = p
    init_r = self.cfg.initial_ranges
    lim_r = self.cfg.limit_ranges
    self.cfg.ranges = WbcEePoseCommandCfg.Ranges(
      pos_x=_lerp_range(init_r.pos_x, lim_r.pos_x, p),
      pos_y=_lerp_range(init_r.pos_y, lim_r.pos_y, p),
      pos_z=_lerp_range(init_r.pos_z, lim_r.pos_z, p),
      roll=_lerp_range(init_r.roll, lim_r.roll, p),
      pitch=_lerp_range(init_r.pitch, lim_r.pitch, p),
      yaw=_lerp_range(init_r.yaw, lim_r.yaw, p),
    )

  def _link0_pose_w(self, env_ids: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    if env_ids is None:
      return (
        self.robot.data.body_link_pos_w[:, self.link0_idx],
        self.robot.data.body_link_quat_w[:, self.link0_idx],
      )
    return (
      self.robot.data.body_link_pos_w[env_ids, self.link0_idx],
      self.robot.data.body_link_quat_w[env_ids, self.link0_idx],
    )

  def _update_metrics(self) -> None:
    ee_pos_w = self.robot.data.body_link_pos_w[:, self.body_idx]
    ee_quat_w = self.robot.data.body_link_quat_w[:, self.body_idx]
    link0_pos_w, link0_quat_w = self._link0_pose_w()

    ee_pos_link0 = quat_apply_inverse(link0_quat_w, ee_pos_w - link0_pos_w)
    pos_err_xy = torch.abs(ee_pos_link0[:, :2] - self.pose_command[:, :2])
    pos_err_z = torch.abs(self.pose_command[:, 2:3] - ee_pos_w[:, 2:3])
    pos_error = torch.cat((pos_err_xy, pos_err_z), dim=-1)

    des_quat_w = quat_mul(link0_quat_w, self.pose_command[:, 3:7])
    des_norm = quat_mul(des_quat_w, quat_conjugate(des_quat_w))[:, 0].clamp(min=1e-8)
    des_inv = quat_conjugate(des_quat_w) / des_norm.unsqueeze(-1)
    quat_error = quat_mul(ee_quat_w, des_inv)
    rot_error = axis_angle_from_quat(quat_error)

    self.metrics["position_error"] = torch.norm(pos_error, dim=-1)
    self.metrics["orientation_error"] = torch.norm(rot_error, dim=-1)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> dict[str, float]:
    """Clear trajectory state and defer resampling until post-forward compute()."""
    ids = self._index_env_ids(env_ids)
    extras: dict[str, float] = {}
    for metric_name, metric_value in self.metrics.items():
      extras[metric_name] = torch.mean(metric_value[ids]).item()
      metric_value[ids] = 0.0
    self.command_counter[ids] = 0
    self._bin_id[ids] = -1
    self._cmd_age[ids] = 0.0
    nominal = self._nominal_pose(ids.shape[0])
    self._commit_pose(ids, nominal[:, :3], nominal[:, 3:7])
    self._pending_reset[ids] = True
    return extras

  def compute(
    self, dt: float | torch.Tensor, env_ids: torch.Tensor | None = None
  ) -> None:
    """Flush pending post-reset resamples (post-forward) and advance active trajectories."""
    just_reset = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    if bool(torch.any(self._pending_reset)):
      if env_ids is None:
        pending_ids = self._pending_reset.nonzero(as_tuple=False).flatten()
      else:
        scope = torch.zeros_like(self._pending_reset)
        scope[env_ids] = True
        pending_ids = (self._pending_reset & scope).nonzero(as_tuple=False).flatten()
      if pending_ids.numel() > 0:
        self._pending_reset[pending_ids] = False
        self._bin_id[pending_ids] = -1
        self._cmd_age[pending_ids] = 0.0
        just_reset[pending_ids] = True
        self._resample(pending_ids)

    if env_ids is None:
      if isinstance(dt, torch.Tensor):
        dt_vec = dt.to(device=self.device, dtype=self._cmd_age.dtype).reshape(-1)
        self._cmd_age = torch.where(just_reset, self._cmd_age, self._cmd_age + dt_vec)
      else:
        self._cmd_age[~just_reset] += float(dt)
    else:
      scope_ids = self._index_env_ids(env_ids)
      non_reset = scope_ids[~just_reset[scope_ids]]
      if non_reset.numel() > 0:
        if isinstance(dt, torch.Tensor):
          self._cmd_age[non_reset] += dt[non_reset].to(
            device=self.device, dtype=self._cmd_age.dtype
          )
        else:
          self._cmd_age[non_reset] += float(dt)

    self._pending_step_dt = dt
    try:
      super().compute(dt, env_ids)
    finally:
      self._pending_step_dt = None

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    env_ids = self._as_env_ids(env_ids)
    if env_ids.numel() == 0:
      return
    if self.cfg.adaptive_sampling:
      eligible = (
        (~self._pending_reset[env_ids])
        & self._eval_env_mask[env_ids]
        & (self._bin_id[env_ids] >= 0)
        & (self._cmd_age[env_ids] >= float(self.cfg.adaptive_min_hold_s))
        & (~self._traj_active[env_ids])
      )
      if bool(torch.any(eligible)):
        self._update_bin_errors(env_ids[eligible])
    self._cmd_age[env_ids] = 0.0
    self._bin_id[env_ids] = -1

    goal_pos, goal_quat, sampled_bins = self._sample_safe_goal_with_bins(env_ids)
    traj_prob = float(self.cfg.trajectory_mode_prob)
    if traj_prob <= 0.0:
      self._commit_pose(env_ids, goal_pos, goal_quat)
      self._bin_id[env_ids] = sampled_bins
      return
    choose_traj = torch.rand(env_ids.shape[0], device=self.device) < traj_prob
    if bool(torch.any(~choose_traj)):
      static_ids = env_ids[~choose_traj]
      self._commit_pose(
        static_ids,
        goal_pos[~choose_traj],
        goal_quat[~choose_traj],
      )
      self._bin_id[static_ids] = sampled_bins[~choose_traj]
    if bool(torch.any(choose_traj)):
      self._start_trajectories(
        env_ids[choose_traj],
        goal_pos[choose_traj],
        goal_quat[choose_traj],
        goal_bins=sampled_bins[choose_traj],
      )

  def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
    active_mask = self._traj_active
    if env_ids is not None:
      scoped = torch.zeros_like(active_mask)
      scoped[env_ids] = True
      active_mask = active_mask & scoped
    if not bool(torch.any(active_mask)):
      return

    pending = self._pending_step_dt
    if pending is None:
      step_dt: float | torch.Tensor = float(self.cfg.trajectory_step_dt)
    elif isinstance(pending, torch.Tensor):
      step_dt = pending[active_mask]
    else:
      step_dt = float(pending)

    next_state, cmd_7d = advance_ee_traj_state(
      self._traj_state[active_mask],
      step_dt,
      make_quat_unique=self.cfg.make_quat_unique,
    )
    self._traj_state[active_mask] = next_state
    self.pose_command[active_mask] = cmd_7d
    self._certified_pose[active_mask] = cmd_7d

  def _sample_goal(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    position, quaternion, _ = self._sample_goal_with_bins(env_ids)
    return position, quaternion

  def _sample_goal_with_bins(
    self, env_ids: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    count = env_ids.shape[0]
    ranges = self.cfg.ranges
    assert ranges is not None
    x_lim = torch.tensor(ranges.pos_x, device=self.device, dtype=torch.float32)
    y_lim = torch.tensor(ranges.pos_y, device=self.device, dtype=torch.float32)
    z_lim = torch.tensor(ranges.pos_z, device=self.device, dtype=torch.float32)

    if self.cfg.adaptive_sampling:
      n_x = int(self.cfg.adaptive_x_bins)
      n_z = int(self.cfg.adaptive_z_bins)
      probs = self.bin_probabilities()
      bin_ids = torch.multinomial(probs, count, replacement=True)
      ix = bin_ids // n_z
      iz = bin_ids % n_z
      u_x = (ix.to(dtype=x_lim.dtype) + torch.rand(count, device=self.device)) / float(n_x)
      u_z = (iz.to(dtype=z_lim.dtype) + torch.rand(count, device=self.device)) / float(n_z)
    else:
      bin_ids = torch.full((count,), -1, dtype=torch.long, device=self.device)
      u_x = torch.rand(count, device=self.device)
      u_z = torch.rand(count, device=self.device)

    dx = x_lim[1] - x_lim[0]
    dz = z_lim[1] - z_lim[0]
    x = x_lim[0].expand(count) if float(dx.abs().item()) < 1e-6 else (x_lim[0] + u_x * dx)
    z = z_lim[0].expand(count) if float(dz.abs().item()) < 1e-6 else (z_lim[0] + u_z * dz)

    r_max = self.cfg.max_xy_radius
    if r_max is not None:
      r_max_f = float(r_max)
      x = x.clamp(min=-r_max_f, max=r_max_f)
      y_cap = torch.sqrt((r_max_f * r_max_f - x * x).clamp(min=0.0))
      y_lo_eff = y_lim[0].clamp(min=-y_cap, max=y_cap)
      y_hi_eff = y_lim[1].clamp(min=-y_cap, max=y_cap)
      y = y_lo_eff + torch.rand(count, device=self.device) * (y_hi_eff - y_lo_eff)
    else:
      y = y_lim[0] + torch.rand(count, device=self.device) * (y_lim[1] - y_lim[0])

    position = torch.stack((x, y, z), dim=-1)

    roll_lim = torch.tensor(ranges.roll, device=self.device)
    pitch_lim = torch.tensor(ranges.pitch, device=self.device)
    yaw_lim = torch.tensor(ranges.yaw, device=self.device)
    rand_euler = torch.rand(count, 3, device=self.device)

    link0_pos_w, link0_quat_w = self._link0_pose_w(env_ids)
    local, _, singular = mixed_curve_link0_points(
      position.unsqueeze(1), link0_pos_w, link0_quat_w
    )
    delta_b = local[:, 0]
    delta_b = torch.where(
      singular[:, 0].unsqueeze(-1), torch.zeros_like(delta_b), delta_b
    )
    dist_xy = torch.sqrt(delta_b[:, 0] ** 2 + delta_b[:, 1] ** 2)
    aim_pitch = -torch.atan2(delta_b[:, 2], dist_xy)
    aim_yaw = torch.atan2(delta_b[:, 1], delta_b[:, 0])

    euler = torch.stack(
      (
        roll_lim[0] + rand_euler[:, 0] * (roll_lim[1] - roll_lim[0]),
        aim_pitch + (pitch_lim[0] + rand_euler[:, 1] * (pitch_lim[1] - pitch_lim[0])),
        aim_yaw + (yaw_lim[0] + rand_euler[:, 2] * (yaw_lim[1] - yaw_lim[0])),
      ),
      dim=-1,
    )
    euler = clamp_wbc_euler(euler, self.cfg.ori_abs_max)
    quaternion = quat_from_euler_xyz(euler[:, 0], euler[:, 1], euler[:, 2])
    if self.cfg.make_quat_unique:
      quaternion = quat_unique(quaternion)
    return position, quaternion, bin_ids

  def _start_trajectories(
    self,
    env_ids: torch.Tensor,
    goal_pos: torch.Tensor,
    goal_quat: torch.Tensor,
    *,
    goal_bins: torch.Tensor | None = None,
  ) -> None:
    start_pos = self.pose_command[env_ids, :3].clone()
    start_quat = self.pose_command[env_ids, 3:7].clone()
    accepted = torch.zeros(env_ids.shape[0], dtype=torch.bool, device=self.device)
    control = torch.zeros_like(goal_pos)

    for attempt in range(_TRAJECTORY_ATTEMPTS):
      pending = ~accepted
      if not bool(torch.any(pending)):
        break
      chord = goal_pos[pending] - start_pos[pending]
      if attempt == _TRAJECTORY_ATTEMPTS - 1:
        delta = torch.zeros_like(chord)
      else:
        delta = self._sample_lateral_delta(chord)
      candidate = 0.5 * (start_pos[pending] + goal_pos[pending]) + delta
      safe = ~self._curve_is_forbidden(
        env_ids[pending],
        start_pos[pending],
        candidate,
        goal_pos[pending],
        start_quat[pending],
        goal_quat[pending],
      )
      if bool(torch.any(safe)):
        pending_index = pending.nonzero(as_tuple=False).flatten()
        accepted_index = pending_index[safe]
        control[accepted_index] = candidate[safe]
        accepted[accepted_index] = True

    if bool(torch.any(~accepted)):
      self._hold_rejected_goals(
        env_ids[~accepted],
        goal_pos[~accepted],
        goal_quat[~accepted],
        goal_bins=goal_bins[~accepted] if goal_bins is not None else None,
      )
    if not bool(torch.any(accepted)):
      return

    duration = self._sample_duration(int(accepted.sum().item()))
    ids = env_ids[accepted]
    self._traj_state[ids] = pack_ee_traj_state(
      start_pos[accepted],
      control[accepted],
      goal_pos[accepted],
      start_quat[accepted],
      goal_quat[accepted],
      torch.zeros_like(duration),
      duration,
      torch.ones_like(duration),
    )
    self.time_left[ids] = self._trajectory_time_left(duration)
    if goal_bins is not None:
      self._bin_id[ids] = goal_bins[accepted]

  def _curve_is_forbidden(
    self,
    env_ids: torch.Tensor,
    start: torch.Tensor,
    control: torch.Tensor,
    goal: torch.Tensor,
    start_quat: torch.Tensor,
    goal_quat: torch.Tensor,
  ) -> torch.Tensor:
    points = sample_quadratic_bezier(start, control, goal)
    link0_pos, link0_quat = self._link0_pose_w(env_ids)
    local, z_world, singular = mixed_curve_link0_points(points, link0_pos, link0_quat)
    position_bad = trajectory_is_forbidden(local, z_world, singular)
    orientation_bad = orientation_path_exceeds_limits(
      start_quat, goal_quat, ori_abs_max=self.cfg.ori_abs_max
    )
    return position_bad | orientation_bad

  def _sample_lateral_delta(self, chord: torch.Tensor) -> torch.Tensor:
    count = chord.shape[0]
    perpendicular = torch.stack(
      (
        -chord[:, 1],
        chord[:, 0],
        torch.zeros(count, device=chord.device, dtype=chord.dtype),
      ),
      dim=-1,
    )
    length = perpendicular.norm(dim=-1, keepdim=True)
    fallback = chord.new_tensor((1.0, 0.0, 0.0)).view(1, 3).expand(count, 3)
    perpendicular = torch.where(
      length > 1e-4, perpendicular / length.clamp(min=1e-6), fallback
    )
    use_curve = torch.rand(count, 1, device=chord.device) < float(
      self.cfg.trajectory_curve_prob
    )
    magnitude = 0.08 + 0.17 * torch.rand(count, 1, device=chord.device)
    sign = torch.where(torch.rand(count, 1, device=chord.device) < 0.5, -1.0, 1.0)
    delta = perpendicular * sign * magnitude
    return torch.where(use_curve, delta, torch.zeros_like(delta))

  def _sample_duration(self, count: int) -> torch.Tensor:
    low, high = self.cfg.trajectory_duration_range
    if abs(float(high) - float(low)) < 1e-8:
      return torch.full((count,), float(low), device=self.device)
    return torch.empty(count, device=self.device).uniform_(float(low), float(high))

  def _sample_safe_goal(
    self, env_ids: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    goal_pos, goal_quat, _ = self._sample_safe_goal_with_bins(env_ids)
    return goal_pos, goal_quat

  def _sample_safe_goal_with_bins(
    self, env_ids: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    count = env_ids.shape[0]
    goal_pos = torch.empty(count, 3, device=self.device)
    goal_quat = torch.empty(count, 4, device=self.device)
    sampled_bins = torch.full((count,), -1, dtype=torch.long, device=self.device)
    pending = torch.ones(count, dtype=torch.bool, device=self.device)
    for _ in range(_GOAL_SAMPLE_ATTEMPTS):
      if not bool(torch.any(pending)):
        break
      sampled_pos, sampled_quat, bins = self._sample_goal_with_bins(env_ids[pending])
      bad = self._positions_forbidden(env_ids[pending], sampled_pos)
      index = pending.nonzero(as_tuple=False).flatten()
      good = ~bad
      if bool(torch.any(good)):
        goal_pos[index[good]] = sampled_pos[good]
        goal_quat[index[good]] = sampled_quat[good]
        sampled_bins[index[good]] = bins[good]
        pending[index[good]] = False
    if bool(torch.any(pending)):
      fallback_pos, fallback_quat = self._fallback_pose(env_ids[pending])
      goal_pos[pending] = fallback_pos
      goal_quat[pending] = fallback_quat
    return goal_pos, goal_quat, sampled_bins

  def _hold_rejected_goals(
    self,
    env_ids: torch.Tensor,
    goal_pos: torch.Tensor,
    goal_quat: torch.Tensor,
    *,
    goal_bins: torch.Tensor | None = None,
  ) -> None:
    goal_bad = self._positions_forbidden(env_ids, goal_pos)
    if bool(torch.any(~goal_bad)):
      ok_ids = env_ids[~goal_bad]
      self._commit_pose(ok_ids, goal_pos[~goal_bad], goal_quat[~goal_bad])
      if goal_bins is not None:
        self._bin_id[ok_ids] = goal_bins[~goal_bad]
    if bool(torch.any(goal_bad)):
      bad_ids = env_ids[goal_bad]
      fallback_pos, fallback_quat = self._fallback_pose(bad_ids)
      self._commit_pose(bad_ids, fallback_pos, fallback_quat)
      self._bin_id[bad_ids] = -1

  def _fallback_pose(
    self, env_ids: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    nominal = self._nominal_pose(env_ids.shape[0])
    current_pos = self.pose_command[env_ids, :3]
    current_quat = self.pose_command[env_ids, 3:7]
    current_ok = ~self._positions_forbidden(env_ids, current_pos)
    position = torch.where(current_ok.unsqueeze(-1), current_pos, nominal[:, :3])
    quaternion = torch.where(current_ok.unsqueeze(-1), current_quat, nominal[:, 3:7])
    certified_ok = ~self._positions_forbidden(
      env_ids, self._certified_pose[env_ids, :3]
    )
    use_certified = (
      (~current_ok)
      & certified_ok
      & self._positions_forbidden(env_ids, nominal[:, :3])
    )
    if bool(torch.any(use_certified)):
      position = position.clone()
      quaternion = quaternion.clone()
      position[use_certified] = self._certified_pose[env_ids[use_certified], :3]
      quaternion[use_certified] = self._certified_pose[env_ids[use_certified], 3:7]
    return position, quaternion

  def _positions_forbidden(
    self, env_ids: torch.Tensor, position: torch.Tensor
  ) -> torch.Tensor:
    link0_pos, link0_quat = self._link0_pose_w(env_ids)
    local, z_world, singular = mixed_curve_link0_points(
      position.unsqueeze(1), link0_pos, link0_quat
    )
    return trajectory_is_forbidden(local, z_world, singular)

  def _nominal_pose(self, count: int) -> torch.Tensor:
    pose = torch.zeros(count, 7, device=self.device)
    pose[:, 0] = SAFE_HOVER_POSITION[0]
    pose[:, 1] = SAFE_HOVER_POSITION[1]
    pose[:, 2] = SAFE_HOVER_POSITION[2]
    pose[:, 3] = 1.0  # WXYZ identity
    return pose

  def _trajectory_time_left(self, duration: torch.Tensor) -> torch.Tensor:
    dt = float(self.cfg.trajectory_step_dt)
    return duration + 2.0 * dt

  def _sync_static_traj_state(
    self, env_ids: torch.Tensor, position: torch.Tensor, quaternion: torch.Tensor
  ) -> None:
    if env_ids.numel() == 0:
      return
    zeros = torch.zeros(env_ids.shape[0], device=self.device, dtype=position.dtype)
    ones = torch.ones_like(zeros)
    self._traj_state[env_ids] = pack_ee_traj_state(
      position, position, position, quaternion, quaternion, zeros, ones, zeros
    )

  def _commit_pose(
    self, env_ids: torch.Tensor, position: torch.Tensor, quaternion: torch.Tensor
  ) -> None:
    if env_ids.numel() == 0:
      return
    if self.cfg.make_quat_unique:
      quaternion = quat_unique(quaternion)
    self.pose_command[env_ids, :3] = position
    self.pose_command[env_ids, 3:7] = quaternion
    self._certified_pose[env_ids, :3] = position
    self._certified_pose[env_ids, 3:7] = quaternion
    self._sync_static_traj_state(env_ids, position, quaternion)

  def _index_env_ids(
    self, env_ids: Sequence[int] | torch.Tensor | slice | None
  ) -> torch.Tensor:
    if env_ids is None:
      return torch.arange(self.num_envs, device=self.device, dtype=torch.long)
    if isinstance(env_ids, slice):
      return torch.arange(self.num_envs, device=self.device, dtype=torch.long)[env_ids]
    return self._as_env_ids(env_ids)

  def _as_env_ids(
    self, env_ids: Sequence[int] | torch.Tensor
  ) -> torch.Tensor:
    if isinstance(env_ids, torch.Tensor):
      return env_ids.to(device=self.device, dtype=torch.long).reshape(-1)
    return torch.tensor(list(env_ids), device=self.device, dtype=torch.long)


@dataclass(kw_only=True)
class WbcEePoseCommandCfg(CommandTermCfg):
  """Configuration for mixed-frame 7D WBC end-effector pose commands (WXYZ)."""

  @dataclass
  class Ranges:
    pos_x: tuple[float, float]
    pos_y: tuple[float, float]
    pos_z: tuple[float, float]
    roll: tuple[float, float]
    pitch: tuple[float, float]
    yaw: tuple[float, float]

  entity_name: str = "robot"
  body_name: str = "end_effector"
  link_name: str = "link0"
  resampling_time_range: tuple[float, float] = (8.0, 10.0)
  stage: int = 1
  curriculum_progress: float = 0.0
  make_quat_unique: bool = True
  trajectory_mode_prob: float | None = None
  trajectory_duration_range: tuple[float, float] = (2.0, 5.0)
  trajectory_curve_prob: float = 0.5
  trajectory_step_dt: float = 0.02
  ori_abs_max: tuple[float, float, float] | None = None
  # Adaptive error-weighted command sampling (2D normalized x-z bins)
  adaptive_sampling: bool = True
  adaptive_x_bins: int = 3
  adaptive_z_bins: int = 5
  adaptive_uniform_prob: float = 0.3
  adaptive_beta: float = 1.25
  adaptive_err_clip: tuple[float, float] = (0.02, 0.20)
  adaptive_ema_alpha: float = 0.05
  adaptive_min_hold_s: float = 2.0
  adaptive_init_err: float = 0.05
  max_xy_radius: float | None = 0.69
  initial_ranges: Ranges = field(
    default_factory=lambda: WbcEePoseCommandCfg.Ranges(
      pos_x=(0.40, 0.45),
      pos_y=(-0.05, 0.05),
      pos_z=(0.50, 0.50),
      roll=(0.0, 0.0),
      pitch=(0.0, 0.0),
      yaw=(0.0, 0.0),
    )
  )
  limit_ranges: Ranges | None = None
  ranges: Ranges | None = None

  def __post_init__(self) -> None:
    if self.stage not in (1, 2):
      raise ValueError(f"stage must be 1 or 2, got {self.stage}")
    if self.adaptive_x_bins < 1:
      raise ValueError(f"adaptive_x_bins must be >= 1, got {self.adaptive_x_bins}")
    if self.adaptive_z_bins < 1:
      raise ValueError(f"adaptive_z_bins must be >= 1, got {self.adaptive_z_bins}")
    if not (0.0 <= float(self.adaptive_uniform_prob) <= 1.0):
      raise ValueError(
        f"adaptive_uniform_prob must be in [0.0, 1.0], got {self.adaptive_uniform_prob}"
      )
    if float(self.adaptive_beta) < 0.0:
      raise ValueError(f"adaptive_beta must be >= 0.0, got {self.adaptive_beta}")
    if (
      len(self.adaptive_err_clip) != 2
      or float(self.adaptive_err_clip[0]) <= 0.0
      or float(self.adaptive_err_clip[1]) < float(self.adaptive_err_clip[0])
    ):
      raise ValueError(
        f"adaptive_err_clip must satisfy 0 < lo <= hi, got {self.adaptive_err_clip}"
      )
    if not (0.0 < float(self.adaptive_ema_alpha) <= 1.0):
      raise ValueError(
        f"adaptive_ema_alpha must be in (0.0, 1.0], got {self.adaptive_ema_alpha}"
      )
    if float(self.adaptive_min_hold_s) < 0.0:
      raise ValueError(
        f"adaptive_min_hold_s must be >= 0.0, got {self.adaptive_min_hold_s}"
      )
    if float(self.adaptive_init_err) <= 0.0:
      raise ValueError(
        f"adaptive_init_err must be > 0.0, got {self.adaptive_init_err}"
      )
    if self.max_xy_radius is not None and float(self.max_xy_radius) <= 0.0:
      raise ValueError(
        f"max_xy_radius must be > 0.0 when set, got {self.max_xy_radius}"
      )
    if (
      self.trajectory_duration_range[0] <= 0.0
      or self.trajectory_duration_range[1] < self.trajectory_duration_range[0]
    ):
      raise ValueError(
        f"trajectory_duration_range must satisfy 0 < min <= max, got {self.trajectory_duration_range}"
      )
    if self.ori_abs_max is None:
      self.ori_abs_max = STAGE1_ORI_ABS_MAX if self.stage == 1 else STAGE2_ORI_ABS_MAX
    if self.trajectory_mode_prob is None:
      self.trajectory_mode_prob = 0.0 if self.stage == 1 else 0.40
    if self.limit_ranges is None:
      r_lim, p_lim, y_lim = self.ori_abs_max
      self.limit_ranges = WbcEePoseCommandCfg.Ranges(
        pos_x=(0.45, 0.70),
        pos_y=(-0.35, 0.35),
        pos_z=(0.10, 0.95),
        roll=(-r_lim, r_lim),
        pitch=(-p_lim, p_lim),
        yaw=(-y_lim, y_lim),
      )
    if self.ranges is None:
      p = max(0.0, min(1.0, float(self.curriculum_progress)))
      self.ranges = WbcEePoseCommandCfg.Ranges(
        pos_x=_lerp_range(self.initial_ranges.pos_x, self.limit_ranges.pos_x, p),
        pos_y=_lerp_range(self.initial_ranges.pos_y, self.limit_ranges.pos_y, p),
        pos_z=_lerp_range(self.initial_ranges.pos_z, self.limit_ranges.pos_z, p),
        roll=_lerp_range(self.initial_ranges.roll, self.limit_ranges.roll, p),
        pitch=_lerp_range(self.initial_ranges.pitch, self.limit_ranges.pitch, p),
        yaw=_lerp_range(self.initial_ranges.yaw, self.limit_ranges.yaw, p),
      )

  def build(self, env: ManagerBasedRlEnv) -> WbcEePoseCommand:
    return WbcEePoseCommand(self, env)
