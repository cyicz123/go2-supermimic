"""Unitree Go2 + Piper 18-DoF robot MJCF, constants, and mjlab articulation config."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path

import mujoco
import torch

from mjlab.actuator import ActuatorCmd, IdealPdActuator, IdealPdActuatorCfg
from mjlab.actuator.actuator import TransmissionType
from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and canonical topology constants.
##

GO2_PIPER_XML: Path = Path(__file__).parent / "go2_piper.xml"
assert GO2_PIPER_XML.exists()

LEG_JOINT_NAMES: tuple[str, ...] = (
  "FR_hip_joint",
  "FR_thigh_joint",
  "FR_calf_joint",
  "FL_hip_joint",
  "FL_thigh_joint",
  "FL_calf_joint",
  "RR_hip_joint",
  "RR_thigh_joint",
  "RR_calf_joint",
  "RL_hip_joint",
  "RL_thigh_joint",
  "RL_calf_joint",
)
ARM_JOINT_NAMES: tuple[str, ...] = (
  "joint1",
  "joint2",
  "joint3",
  "joint4",
  "joint5",
  "joint6",
)
CANONICAL_JOINT_NAMES: tuple[str, ...] = LEG_JOINT_NAMES + ARM_JOINT_NAMES

FOOT_SITE_NAMES: tuple[str, ...] = ("FR", "FL", "RR", "RL")
FOOT_GEOM_NAMES: tuple[str, ...] = tuple(
  f"{leg}_foot_collision" for leg in FOOT_SITE_NAMES
)

# Canonical default standing pose (FR, FL, RR, RL, joint1..joint6).
DEFAULT_JOINT_POS: tuple[float, ...] = (
  -0.1, 0.8, -1.5,
  0.1, 0.8, -1.5,
  -0.1, 1.0, -1.5,
  0.1, 1.0, -1.5,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
)
DEFAULT_JOINT_POS_DICT: dict[str, float] = dict(
  zip(CANONICAL_JOINT_NAMES, DEFAULT_JOINT_POS, strict=True)
)


def get_spec() -> mujoco.MjSpec:
  """Load a fresh MuJoCo MjSpec for the 18-DoF Go2+Piper model."""
  return mujoco.MjSpec.from_file(str(GO2_PIPER_XML))


##
# DelayedPD actuator aligned with Isaac Lab #460 DelayedPDActuator semantics.
##


class DelayedPdActuator(IdealPdActuator["DelayedPdActuatorCfg"]):
  """Isaac Lab #460 DelayedPDActuator parity: pure constant-effort PD + sim-level velocity clamp.

  Overrides ``compute()`` deliberately so ``mjlab.actuator.FusedActuatorGroup``
  keeps each actuator group un-fused:
  1. Matches Isaac Lab ``DelayedPDActuator`` group-level ``DelayBuffer(per_env=True)``:
     each actuator group samples one scalar episode lag per env (``leg_hip_thigh``
     8 joints share a lag, ``leg_calf`` 4 joints share a lag, and ``joint1..joint6``
     are 6 separate actuators so all 6 arm joints get independent per-env lags).
  2. Advances the per-env delay ring buffer strictly once per real ``sim.step()``
     (gated on ``sim.data.time`` advancement), so post-reset ``write_data_to_sim()``
     calls in ``ManagerBasedRlEnv`` never double-advance un-reset rows on partial resets.
  3. Decouples torque computation from velocity limits: ``compute()`` applies ONLY
     pure PD + constant ``effort_limit`` clamp (zero velocity-based torque derating),
     while ``update()`` enforces ``velocity_limit_sim`` post-substep on ``qvel`` so
     ``|qd| <= velocity_limit`` even under zero torque with gravity/Coriolis coupling.
  """

  def __init__(
    self,
    cfg: DelayedPdActuatorCfg,
    entity: Entity,
    target_ids: list[int],
    target_names: list[str],
  ) -> None:
    super().__init__(cfg, entity, target_ids, target_names)
    self.velocity_limit: torch.Tensor | None = None
    self._joint_v_adrs: torch.Tensor | None = None
    self._joint_q_adrs: torch.Tensor | None = None
    self._qpos_prev: torch.Tensor | None = None
    self._qpos_prev_time: torch.Tensor | None = None
    self._lags: torch.Tensor | None = None
    self._history_len: int = 0
    self._cmd_history: torch.Tensor | None = None
    self._write_idx: torch.Tensor | None = None
    self._has_pending: torch.Tensor | None = None
    self._last_cmd: torch.Tensor | None = None
    self._last_sim_time: torch.Tensor | None = None
    self._env_arange: torch.Tensor | None = None

  def _init_delay_buffer(self, num_envs: int, device: str) -> None:
    """Disable base shared DelayBuffer; delay is managed per actuator group below."""
    del num_envs, device
    self._delay_buffer = None

  @property
  def current_lags(self) -> torch.Tensor | None:
    """Per-env episode delay lags in physics steps for this actuator group. Shape (num_envs,)."""
    return self._lags

  @staticmethod
  def control_law(params: dict[str, torch.Tensor], cmd: ActuatorCmd) -> torch.Tensor:
    """Pure PD control law with constant ``effort_limit`` clamp (no velocity derating)."""
    return IdealPdActuator.control_law(params, cmd)

  def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
    assert self.stiffness is not None
    assert self.damping is not None
    assert self.force_limit is not None
    if self._qpos_prev is not None and self._qpos_prev_time is not None:
      # Capture pre-substep joint positions so update(dt) can align position integration
      # with velocity_limit_sim when sim.step() overshoots velocity bounds mid-step.
      self._qpos_prev.copy_(cmd.pos)
      self._qpos_prev_time.copy_(self.entity.data.data.time)
    params = {
      "stiffness": self.stiffness,
      "damping": self.damping,
      "force_limit": self.force_limit,
    }
    return DelayedPdActuator.control_law(params, cmd)

  def _sample_lags(self, count: int) -> torch.Tensor:
    assert self._lags is not None
    if not self.has_delay:
      return torch.zeros(count, dtype=torch.long, device=self._lags.device)
    return torch.randint(
      self.cfg.delay_min_lag,
      self.cfg.delay_max_lag + 1,
      (count,),
      dtype=torch.long,
      device=self._lags.device,
    )

  def set_lags(
    self,
    lags: torch.Tensor | int,
    env_ids: torch.Tensor | slice | None = None,
  ) -> None:
    """Set delay lag values for specified environments. Shape (num_env_ids,) or scalar."""
    if self._lags is None:
      return
    idx: torch.Tensor | slice = slice(None) if env_ids is None else env_ids
    lags_t = torch.as_tensor(lags, dtype=torch.long, device=self._lags.device)
    if lags_t.ndim > 1:
      lags_t = lags_t.squeeze(-1)
    self._lags[idx] = lags_t.clamp(self.cfg.delay_min_lag, self.cfg.delay_max_lag)

  def initialize(self, mj_model, model, data, device: str) -> None:
    super().initialize(mj_model, model, data, device)
    num_envs = data.nworld
    num_joints = len(self._target_names)

    if (
      self.cfg.velocity_limit is not None
      and self.cfg.transmission_type == TransmissionType.JOINT
    ):
      assert self._target_ids is not None
      self._joint_v_adrs = self.entity.indexing.joint_v_adr[self._target_ids]
      self._joint_q_adrs = self.entity.indexing.joint_q_adr[self._target_ids]
      self.velocity_limit = torch.full(
        (num_envs, num_joints), float(self.cfg.velocity_limit), dtype=torch.float, device=device
      )
      self._qpos_prev = torch.full(
        (num_envs, num_joints), float("nan"), dtype=torch.float, device=device
      )
      self._qpos_prev_time = torch.full(
        (num_envs,), float("nan"), dtype=data.time.dtype, device=device
      )
    else:
      self._joint_v_adrs = None
      self._joint_q_adrs = None
      self.velocity_limit = None
      self._qpos_prev = None
      self._qpos_prev_time = None

    self._lags = torch.zeros(num_envs, dtype=torch.long, device=device)
    if self.has_delay:
      self._history_len = int(self.cfg.delay_max_lag)
      self._cmd_history = torch.zeros(
        (num_envs, self._history_len, num_joints, 3), dtype=torch.float, device=device
      )
      self._write_idx = torch.zeros(num_envs, dtype=torch.long, device=device)
      self._has_pending = torch.zeros(num_envs, dtype=torch.bool, device=device)
      self._last_cmd = torch.zeros((num_envs, num_joints, 3), dtype=torch.float, device=device)
      self._last_sim_time = torch.full(
        (num_envs,), float("nan"), dtype=data.time.dtype, device=device
      )
      self._env_arange = torch.arange(num_envs, dtype=torch.long, device=device)
    self._lags.copy_(self._sample_lags(num_envs))

  def apply_delay(self, cmd: ActuatorCmd) -> ActuatorCmd:
    if not self.has_delay:
      return cmd
    assert self._lags is not None
    assert self._cmd_history is not None
    assert self._write_idx is not None
    assert self._has_pending is not None
    assert self._last_cmd is not None
    assert self._last_sim_time is not None
    assert self._env_arange is not None

    stacked = torch.stack((cmd.position_target, cmd.velocity_target, cmd.effort_target), dim=-1)
    sim_time = self.entity.data.data.time
    stepped = self._has_pending & (sim_time > self._last_sim_time)

    old_slot = self._cmd_history[self._env_arange, self._write_idx]
    self._cmd_history[self._env_arange, self._write_idx] = torch.where(
      stepped[:, None, None], self._last_cmd, old_slot
    )
    self._write_idx = (self._write_idx + stepped.to(torch.long)).remainder(self._history_len)

    self._last_cmd.copy_(stacked)
    self._last_sim_time.copy_(sim_time)
    self._has_pending.fill_(True)

    read_slots = (self._write_idx - self._lags).remainder(self._history_len)
    delayed_hist = self._cmd_history[self._env_arange, read_slots]
    delayed = torch.where((self._lags == 0)[:, None, None], stacked, delayed_hist)
    return dataclasses.replace(
      cmd,
      position_target=delayed[..., 0],
      velocity_target=delayed[..., 1],
      effort_target=delayed[..., 2],
    )

  def enforce_velocity_limit(self, dt: float | None = None) -> None:
    """Clamp joint velocities and step displacements to ``±velocity_limit`` (``velocity_limit_sim`` parity)."""
    if self.velocity_limit is None or self._joint_v_adrs is None:
      return
    sim_data = self.entity.data.data
    qvel = sim_data.qvel
    raw_qvel = qvel[:, self._joint_v_adrs]
    vel_exceeded = raw_qvel.abs() > self.velocity_limit
    qvel[:, self._joint_v_adrs] = torch.clamp(
      raw_qvel, min=-self.velocity_limit, max=self.velocity_limit
    )

    if (
      dt is not None
      and dt > 0.0
      and self._joint_q_adrs is not None
      and self._qpos_prev is not None
      and self._qpos_prev_time is not None
    ):
      sim_time = sim_data.time
      valid_step = torch.isfinite(self._qpos_prev_time) & (sim_time > self._qpos_prev_time)
      qpos = sim_data.qpos
      q_new = qpos[:, self._joint_q_adrs]
      max_dq = self.velocity_limit * float(dt)
      q_limited = self._qpos_prev + torch.clamp(q_new - self._qpos_prev, min=-max_dq, max=max_dq)
      q_out = torch.where(valid_step.unsqueeze(-1) & vel_exceeded, q_limited, q_new)
      qpos[:, self._joint_q_adrs] = q_out
      self._qpos_prev.copy_(q_out)
      self._qpos_prev_time.copy_(sim_time)

  def update(self, dt: float) -> None:
    """Enforce simulation-level ``velocity_limit_sim`` (qvel and Δq) after each ``sim.step()``."""
    self.enforce_velocity_limit(dt=dt)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if self._lags is None:
      return
    batch_size = self._lags.shape[0]
    if env_ids is None:
      idx: torch.Tensor | slice = slice(None)
      count = batch_size
    elif isinstance(env_ids, slice):
      count = len(range(*env_ids.indices(batch_size)))
      idx = env_ids
    else:
      count = int(env_ids.numel()) if isinstance(env_ids, torch.Tensor) else len(env_ids)
      idx = env_ids
    if count == 0:
      return
    self._lags[idx] = self._sample_lags(count)
    if self._qpos_prev is not None and self._qpos_prev_time is not None:
      self._qpos_prev[idx] = float("nan")
      self._qpos_prev_time[idx] = float("nan")
    if self.has_delay:
      assert self._cmd_history is not None
      assert self._write_idx is not None
      assert self._has_pending is not None
      assert self._last_cmd is not None
      assert self._last_sim_time is not None
      self._cmd_history[idx] = 0.0
      self._write_idx[idx] = 0
      self._has_pending[idx] = False
      self._last_cmd[idx] = 0.0
      self._last_sim_time[idx] = float("nan")


@dataclass(kw_only=True)
class DelayedPdActuatorCfg(IdealPdActuatorCfg):
  """Configuration for constant-effort DelayedPD actuators (#460 parity)."""

  velocity_limit: float | None = None
  """Nominal joint velocity limit in rad/s (metadata / reference bound)."""

  def __post_init__(self) -> None:
    # Keep lag constant within each episode; reset() samples once per episode.
    if self.delay_max_lag > 0 and self.delay_hold_prob == 0.0:
      self.delay_hold_prob = 1.0
    super().__post_init__()

  def build(
    self, entity: Entity, target_ids: list[int], target_names: list[str]
  ) -> DelayedPdActuator:
    return DelayedPdActuator(self, entity, target_ids, target_names)


##
# Actuator parameters (#460 golden baseline + canonical WBC leg PD).
##

LEG_KP: float = 40.0
LEG_KD: float = 1.0
LEG_EFFORT_LIMIT: tuple[float, ...] = (
  23.7, 23.7, 45.43,
  23.7, 23.7, 45.43,
  23.7, 23.7, 45.43,
  23.7, 23.7, 45.43,
)
LEG_VELOCITY_LIMIT: tuple[float, ...] = (
  30.1, 30.1, 15.7,
  30.1, 30.1, 15.7,
  30.1, 30.1, 15.7,
  30.1, 30.1, 15.7,
)

# Piper arm #460 golden SI baseline (10 A effort caps).
ARM_KP: tuple[float, ...] = (50.0, 65.0, 55.0, 25.0, 20.0, 15.0)
ARM_KD: tuple[float, ...] = (2.0, 3.0, 2.0, 0.8, 0.6, 0.4)
ARM_EFFORT_LIMIT: tuple[float, ...] = (
  11.8125, 11.8125, 11.8125,
  9.5844, 9.5844, 9.5844,
)
ARM_VELOCITY_LIMIT: tuple[float, ...] = (3.0, 3.0, 3.0, 3.0, 3.0, 3.0)

ARMATURE: float = 0.01
ACTION_SCALE: float = 0.25
GO2_PIPER_ACTION_SCALE: dict[str, float] = {
  name: ACTION_SCALE for name in CANONICAL_JOINT_NAMES
}

# Actuator delay in physics timesteps (dt = 0.005 s -> leg 0..6 = 0..30 ms, arm 3..9 = 15..45 ms).
# LeggedManip_Lab@b2cce16 (go2_piper_articulation_cfg.py) defines 2 leg DelayedPDActuatorCfg
# groups (leg_hip_thigh, leg_calf -> [0, 6] per-group shared env lag) and 6 separate arm
# DelayedPDActuatorCfg entries (joint1..joint6 -> [3, 9] independent per-joint env lags).
DEFAULT_LEG_DELAY_LAG: tuple[int, int] = (0, 6)
DEFAULT_ARM_DELAY_LAG: tuple[int, int] = (3, 9)


def _make_actuators(
  leg_delay: tuple[int, int] = DEFAULT_LEG_DELAY_LAG,
  arm_delay: tuple[int, int] = DEFAULT_ARM_DELAY_LAG,
) -> tuple[DelayedPdActuatorCfg, ...]:
  """Build leg and Piper arm DelayedPD actuator configs with per-episode lag."""
  leg_min, leg_max = leg_delay
  arm_min, arm_max = arm_delay
  if leg_min < 0 or leg_min > leg_max:
    raise ValueError(f"Invalid leg_delay={leg_delay}")
  if arm_min < 0 or arm_min > arm_max:
    raise ValueError(f"Invalid arm_delay={arm_delay}")

  leg_hip_thigh = DelayedPdActuatorCfg(
    target_names_expr=(".*_hip_joint", ".*_thigh_joint"),
    stiffness=LEG_KP,
    damping=LEG_KD,
    effort_limit=23.7,
    velocity_limit=30.1,
    armature=ARMATURE,
    delay_min_lag=leg_min,
    delay_max_lag=leg_max,
    delay_hold_prob=1.0,
  )
  leg_calf = DelayedPdActuatorCfg(
    target_names_expr=(".*_calf_joint",),
    stiffness=LEG_KP,
    damping=LEG_KD,
    effort_limit=45.43,
    velocity_limit=15.7,
    armature=ARMATURE,
    delay_min_lag=leg_min,
    delay_max_lag=leg_max,
    delay_hold_prob=1.0,
  )
  arm_cfgs = tuple(
    DelayedPdActuatorCfg(
      target_names_expr=(joint_name,),
      stiffness=kp,
      damping=kd,
      effort_limit=eff,
      velocity_limit=vel,
      armature=ARMATURE,
      delay_min_lag=arm_min,
      delay_max_lag=arm_max,
      delay_hold_prob=1.0,
    )
    for joint_name, kp, kd, eff, vel in zip(
      ARM_JOINT_NAMES,
      ARM_KP,
      ARM_KD,
      ARM_EFFORT_LIMIT,
      ARM_VELOCITY_LIMIT,
      strict=True,
    )
  )
  return (leg_hip_thigh, leg_calf, *arm_cfgs)


##
# Initial state and collision configuration.
##

INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.35),
  joint_pos={
    "FR_hip_joint": -0.1,
    "FL_hip_joint": 0.1,
    "RR_hip_joint": -0.1,
    "RL_hip_joint": 0.1,
    "FR_thigh_joint": 0.8,
    "FL_thigh_joint": 0.8,
    "RR_thigh_joint": 1.0,
    "RL_thigh_joint": 1.0,
    ".*_calf_joint": -1.5,
    "joint[1-6]": 0.0,
  },
  joint_vel={".*": 0.0},
)

_foot_regex = r"^[FR][LR]_foot_collision$"
_collision_regex = r".*_collision.*"

# All named collision geoms enabled (including arm/EE <-> link0/base self-collisions
# governed by <contact><exclude> in go2_piper.xml); feet get condim=6.
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(_collision_regex,),
  contype=1,
  conaffinity=1,
  solref=(0.01, 1),
  condim={_foot_regex: 6, _collision_regex: 1},
  priority={_foot_regex: 1, ".*": 0},
  friction={_foot_regex: (0.8, 0.02, 0.01)},
)


def get_go2_piper_robot_cfg(
  leg_delay: tuple[int, int] = DEFAULT_LEG_DELAY_LAG,
  arm_delay: tuple[int, int] = DEFAULT_ARM_DELAY_LAG,
) -> EntityCfg:
  """Return a fresh Go2+Piper 18-DoF EntityCfg instance."""
  articulation = EntityArticulationInfoCfg(
    actuators=_make_actuators(leg_delay=leg_delay, arm_delay=arm_delay),
    soft_joint_pos_limit_factor=0.9,
  )
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=articulation,
  )
