"""50 Hz Piper arm slew-rate-limited joint position action term."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg

from go2_supermimic.robot.go2_piper import ARM_JOINT_NAMES

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _clamp_slew_target(
  q_ref_prev: torch.Tensor,
  q_arm_cmd: torch.Tensor,
  arm_lower: torch.Tensor | float | None,
  arm_upper: torch.Tensor | float | None,
  max_delta: float,
) -> torch.Tensor:
  """Clamp target to limits, apply 1-step delta limit, then re-clamp to limits."""
  if not bool(torch.isfinite(q_arm_cmd).all()) or not bool(
    torch.isfinite(q_ref_prev).all()
  ):
    raise ValueError("Arm slew target and base reference must be finite.")
  if (arm_lower is None) ^ (arm_upper is None):
    raise ValueError("arm_lower and arm_upper must both be provided or both be None.")
  lower_t: torch.Tensor | None = None
  upper_t: torch.Tensor | None = None
  if arm_lower is not None and arm_upper is not None:
    lower_t = torch.as_tensor(arm_lower, device=q_arm_cmd.device, dtype=q_arm_cmd.dtype)
    upper_t = torch.as_tensor(arm_upper, device=q_arm_cmd.device, dtype=q_arm_cmd.dtype)
    if not bool(torch.isfinite(lower_t).all()) or not bool(
      torch.isfinite(upper_t).all()
    ):
      raise ValueError("Arm joint limits must be finite.")
    if bool(torch.any(lower_t > upper_t)):
      raise ValueError("Arm joint lower limits must not exceed upper limits.")
    q_arm_cmd = q_arm_cmd.clamp(lower_t, upper_t)
  dq = (q_arm_cmd - q_ref_prev).clamp(-max_delta, max_delta)
  q_next = q_ref_prev + dq
  if lower_t is not None and upper_t is not None:
    q_next = q_next.clamp(lower_t, upper_t)
  return q_next


def slew_step_arm(
  q_ref_prev: torch.Tensor,
  raw_arm: torch.Tensor,
  default_arm: torch.Tensor | float,
  scale_arm: torch.Tensor | float,
  arm_lower: torch.Tensor | float | None,
  arm_upper: torch.Tensor | float | None,
  max_delta: float,
) -> torch.Tensor:
  """Pure differentiable 50 Hz arm slew-limiter step matching C++ ArmSlewLimiter.

  Computes ``q_target = clamp(default_arm + scale_arm * raw_arm, lower, upper)``
  and advances ``q_ref_prev`` by at most ``max_delta`` before re-clamping to
  ``[arm_lower, arm_upper]``.
  """
  if not math.isfinite(float(max_delta)) or float(max_delta) <= 0.0:
    raise ValueError(f"max_delta must be finite and positive, got {max_delta}")
  q_arm_cmd = default_arm + scale_arm * raw_arm
  return _clamp_slew_target(
    q_ref_prev=q_ref_prev,
    q_arm_cmd=q_arm_cmd,
    arm_lower=arm_lower,
    arm_upper=arm_upper,
    max_delta=float(max_delta),
  )


class PiperSlewJointPositionAction(JointPositionAction):
  """Pass-through leg PD targets + 50 Hz first-order slew limiter on Piper arm joints.

  ``raw_actions`` stores the clipped network actions (pre-slew) for the 210D
  ``actions`` history term. ``q_ref_arm`` exposes the 6D slewed arm reference
  angles, and ``processed_actions`` holds the 18D joint targets sent to PD.
  """

  cfg: PiperSlewJointPositionActionCfg

  def __init__(self, cfg: PiperSlewJointPositionActionCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(cfg=cfg, env=env)
    self._validate_slew_cfg()
    names = [str(name) for name in self._target_names]
    missing = [name for name in cfg.arm_joint_names if name not in names]
    if missing:
      raise ValueError(f"Piper slew action missing arm joints: {missing}")
    arm_index = [names.index(name) for name in cfg.arm_joint_names]
    self._arm_indices = torch.tensor(arm_index, device=self.device, dtype=torch.long)
    self._q_ref_prev = torch.zeros(
      self.num_envs, len(arm_index), device=self.device, dtype=torch.float32
    )
    self._needs_sync = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

  @property
  def raw_actions(self) -> torch.Tensor:
    """Clipped pre-slew 18D actions (alias matching Isaac Lab / 210D contract)."""
    return self._raw_actions

  @property
  def q_ref_arm(self) -> torch.Tensor:
    """Current 6D slewed arm joint position reference."""
    return self._q_ref_prev

  @property
  def processed_actions(self) -> torch.Tensor:
    """18D joint position targets sent to the PD actuators."""
    return self._processed_actions

  def _validate_slew_cfg(self) -> None:
    policy_dt = float(self.cfg.policy_dt)
    max_arm_vel = float(self.cfg.max_arm_vel)
    if not math.isfinite(policy_dt) or policy_dt <= 0.0:
      raise ValueError(f"policy_dt must be finite and positive, got {self.cfg.policy_dt}")
    if not math.isfinite(max_arm_vel) or max_arm_vel <= 0.0:
      raise ValueError(
        f"max_arm_vel must be finite and positive, got {self.cfg.max_arm_vel}"
      )

  def _clip_raw_actions(self, actions: torch.Tensor) -> torch.Tensor:
    if self.cfg.clip is None:
      return actions
    return torch.clamp(actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1])

  def _measured_arm_pos(self) -> torch.Tensor:
    joint_pos = self._entity.data.joint_pos
    selected = joint_pos[:, self._target_ids]
    return selected.index_select(dim=-1, index=self._arm_indices)

  def _arm_position_limits(self) -> tuple[torch.Tensor, torch.Tensor] | None:
    data = self._entity.data
    limits = getattr(data, "joint_pos_limits", None)
    if limits is None:
      return None
    tensor = limits.to(device=self.device, dtype=self._q_ref_prev.dtype)
    if tensor.ndim == 2:
      tensor = tensor.unsqueeze(0).expand(self.num_envs, -1, -1)
    selected = tensor[:, self._target_ids, :].index_select(dim=1, index=self._arm_indices)
    lower = selected[..., 0]
    upper = selected[..., 1]
    valid = torch.isfinite(lower) & torch.isfinite(upper) & (lower <= upper)
    if not bool(torch.all(valid)):
      raise ValueError("Arm joint limits must be finite with lower <= upper.")
    return lower, upper

  def _resolve_arm_base_reference(
    self,
    measured: torch.Tensor,
    limits: tuple[torch.Tensor, torch.Tensor] | None,
  ) -> torch.Tensor:
    q_meas = measured if limits is None else measured.clamp(limits[0], limits[1])
    return torch.where(self._needs_sync.unsqueeze(-1), q_meas, self._q_ref_prev)

  def process_actions(self, actions: torch.Tensor) -> None:
    """Clip raw actions, pass through legs, and slew-limit arm targets at 50 Hz."""
    if not bool(torch.isfinite(actions).all()):
      raise ValueError("Actions passed to PiperSlewJointPositionAction must be finite.")
    clipped = self._clip_raw_actions(actions)
    q_cmd = clipped * self._scale + self._offset
    q_arm_cmd = q_cmd.index_select(dim=-1, index=self._arm_indices)
    measured = self._measured_arm_pos()
    if not bool(torch.isfinite(measured).all()):
      raise ValueError("Measured arm joint positions must be finite.")
    limits = self._arm_position_limits()
    arm_lower = None if limits is None else limits[0]
    arm_upper = None if limits is None else limits[1]
    q_base = self._resolve_arm_base_reference(measured, limits)
    max_delta = float(self.cfg.max_arm_vel) * float(self.cfg.policy_dt)
    q_next = _clamp_slew_target(
      q_ref_prev=q_base,
      q_arm_cmd=q_arm_cmd,
      arm_lower=arm_lower,
      arm_upper=arm_upper,
      max_delta=max_delta,
    )
    processed = q_cmd.detach().clone()
    processed[:, self._arm_indices] = q_next.detach()
    self._q_ref_prev.copy_(q_next.detach())
    self._needs_sync.fill_(False)
    self._raw_actions[:] = clipped
    self._processed_actions = processed

  def reset(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> None:
    """Zero raw actions and mark selected envs to sync q_ref_arm on next step."""
    if env_ids is None:
      env_ids = slice(None)
    super().reset(env_ids)
    self._needs_sync[env_ids] = True


@dataclass(kw_only=True)
class PiperSlewJointPositionActionCfg(JointPositionActionCfg):
  """Configuration for Piper 50 Hz slew-limited joint position actions."""

  max_arm_vel: float = 3.0
  """Max arm reference velocity in rad/s (0.060 rad/step at 50 Hz)."""
  policy_dt: float = 0.02
  """Policy step period in seconds (50 Hz)."""
  arm_joint_names: tuple[str, ...] = ARM_JOINT_NAMES
  """Piper arm joint names subject to slew-rate limiting."""

  def build(self, env: ManagerBasedRlEnv) -> PiperSlewJointPositionAction:
    return PiperSlewJointPositionAction(self, env)
