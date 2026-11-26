"""Unitree Go2 + Piper 18-DoF robot MJCF, constants, and mjlab articulation config."""

from __future__ import annotations

from pathlib import Path

import mujoco

from mjlab.actuator import DcMotorActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
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
) -> tuple[DcMotorActuatorCfg, ...]:
  """Build leg and Piper arm actuator configs with configurable physics-step lag."""
  leg_min, leg_max = leg_delay
  arm_min, arm_max = arm_delay
  if leg_min < 0 or leg_min > leg_max:
    raise ValueError(f"Invalid leg_delay={leg_delay}")
  if arm_min < 0 or arm_min > arm_max:
    raise ValueError(f"Invalid arm_delay={arm_delay}")

  leg_hip_thigh = DcMotorActuatorCfg(
    target_names_expr=(".*_hip_joint", ".*_thigh_joint"),
    stiffness=LEG_KP,
    damping=LEG_KD,
    effort_limit=23.7,
    saturation_effort=2.0 * 23.7,
    velocity_limit=30.1,
    armature=ARMATURE,
    delay_min_lag=leg_min,
    delay_max_lag=leg_max,
  )
  leg_calf = DcMotorActuatorCfg(
    target_names_expr=(".*_calf_joint",),
    stiffness=LEG_KP,
    damping=LEG_KD,
    effort_limit=45.43,
    saturation_effort=2.0 * 45.43,
    velocity_limit=15.7,
    armature=ARMATURE,
    delay_min_lag=leg_min,
    delay_max_lag=leg_max,
  )
  arm_cfgs = tuple(
    DcMotorActuatorCfg(
      target_names_expr=(joint_name,),
      stiffness=kp,
      damping=kd,
      effort_limit=eff,
      saturation_effort=2.0 * eff,
      velocity_limit=vel,
      armature=ARMATURE,
      delay_min_lag=arm_min,
      delay_max_lag=arm_max,
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

# All named collision geoms enabled (including arm <-> link0/base self-collisions
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
