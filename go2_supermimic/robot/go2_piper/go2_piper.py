"""Unitree Go2 + Piper 18-DoF robot MJCF, constants, and mjlab articulation config."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import torch

from mjlab.actuator import IdealPdActuator, IdealPdActuatorCfg
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
  """Ideal PD actuator with constant effort clamp and per-episode sampled delay lag.

  Inherits ``IdealPdActuator.compute`` unchanged so ``FusedActuatorGroup`` still
  fuses actuators sharing delay config, while sampling a per-env delay lag in
  ``[delay_min_lag, delay_max_lag]`` on ``reset()`` and holding it constant for
  the entire episode (``delay_hold_prob = 1.0`` disables per-step resampling).
  """

  def _sample_and_set_lags(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if self._delay_buffer is None:
      return
    buf = self._delay_buffer
    if env_ids is None:
      count = buf.batch_size
      target_ids: torch.Tensor | slice = slice(None)
    elif isinstance(env_ids, slice):
      indices = range(*env_ids.indices(buf.batch_size))
      count = len(indices)
      target_ids = env_ids
    else:
      count = int(env_ids.numel()) if isinstance(env_ids, torch.Tensor) else len(env_ids)
      target_ids = env_ids
    if count == 0:
      return
    lags = torch.randint(
      self.cfg.delay_min_lag,
      self.cfg.delay_max_lag + 1,
      (count,),
      dtype=torch.long,
      device=buf.device,
    )
    buf.set_lags(lags, target_ids)

  def initialize(self, mj_model, model, data, device: str) -> None:
    super().initialize(mj_model, model, data, device)
    self._sample_and_set_lags(None)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids)
    self._sample_and_set_lags(env_ids)


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

# Actuator delay in physics timesteps (dt = 0.005 s -> arm 3..9 = 15..45 ms).
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
