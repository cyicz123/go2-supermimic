"""Go2+Piper canonical 210D->18D WBC mjlab environment configuration."""

from __future__ import annotations

import copy
import math

import torch

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.config import random_rough
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from go2_supermimic.robot.go2_piper import (
  ACTION_SCALE,
  CANONICAL_JOINT_NAMES,
  DEFAULT_ARM_DELAY_LAG,
  DEFAULT_LEG_DELAY_LAG,
  FOOT_GEOM_NAMES,
  get_go2_piper_robot_cfg,
)
from go2_supermimic.wbc.actions import PiperSlewJointPositionActionCfg
from go2_supermimic.wbc.commands import WbcEePoseCommandCfg, WbcVelocityCommandCfg

WBC_HISTORY_LENGTH: int = 3
WBC_SINGLE_FRAME_DIM: int = 70
WBC_OBSERVATION_DIM: int = WBC_HISTORY_LENGTH * WBC_SINGLE_FRAME_DIM  # 210
WBC_ACTION_DIM: int = 18

WBC_OBS_TERMS: tuple[tuple[str, int], ...] = (
  ("base_ang_vel", 3),
  ("projected_gravity", 3),
  ("joint_pos", 18),
  ("joint_vel", 18),
  ("actions", 18),
  ("velocity_commands", 3),
  ("pos_commands", 7),
)


def wbc_previous_action(
  env: ManagerBasedRlEnv, action_name: str = "joint_pos"
) -> torch.Tensor:
  """Return clipped pre-slew 18D network action for the 210D policy contract."""
  return env.action_manager.get_term(action_name).raw_actions


def wbc_command_curriculum(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | slice | None,
  command_name: str,
  start_step: int = 0,
  end_step: int = 3_000,
  init_progress: float = 0.0,
) -> torch.Tensor:
  """Advance a WBC command term's ``curriculum_progress`` toward 1.0 by step counter."""
  del env_ids
  init_p = max(0.0, min(1.0, float(init_progress)))
  span = max(1, int(end_step) - int(start_step))
  frac = max(0.0, min(1.0, (int(env.common_step_counter) - int(start_step)) / span))
  progress = init_p + (1.0 - init_p) * frac
  term = env.command_manager.get_term(command_name)
  term.set_curriculum_progress(progress)
  return torch.tensor(progress, device=env.device, dtype=torch.float32)


def _randomize_actuator_pd_gains(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  target_regex: str,
  kp_range: tuple[float, float],
  kd_range: tuple[float, float],
) -> None:
  """Scale Kp/Kd gains on actuators whose target joint names match ``target_regex``."""
  import re

  asset = env.scene["robot"]
  ids = (
    torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if env_ids is None
    else env_ids.to(device=env.device, dtype=torch.long)
  )
  pattern = re.compile(target_regex)
  for actuator in asset.actuators:
    if not any(pattern.fullmatch(name) for name in actuator.target_names):
      continue
    n_targets = len(actuator.target_ids)
    kp_scale = torch.empty(len(ids), n_targets, device=env.device).uniform_(*kp_range)
    kd_scale = torch.empty(len(ids), n_targets, device=env.device).uniform_(*kd_range)
    assert actuator.default_stiffness is not None and actuator.default_damping is not None
    actuator.set_gains(
      ids,
      kp=actuator.default_stiffness[ids] * kp_scale,
      kd=actuator.default_damping[ids] * kd_scale,
    )


def _quiet_solver_warnings(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None) -> None:
  """Silence per-step solver iteration overflow warnings on large batches."""
  del env_ids
  from mujoco_warp import OverflowType

  opt = env.sim.wp_model.opt
  opt.warn_overflow = int(opt.warn_overflow) & ~int(
    OverflowType.ITERATIONS | OverflowType.LS_ITERATIONS
  )
  env.sim.create_graph()


def _micro_rough_terrain() -> TerrainEntityCfg:
  from go2_supermimic.env.terrains import clean_look

  """Gentle random micro-roughness terrain (1-2 cm amplitude) for Stage 2 robustness."""
  gen = TerrainGeneratorCfg(
    seed=11,
    size=(12.0, 12.0),
    border_width=10.0,
    num_rows=1,
    num_cols=1,
    curriculum=False,
    color_scheme="none",
    sub_terrains={
      "micro_rough": random_rough(
        proportion=1.0, noise_range=(0.005, 0.02), noise_step=0.005
      ),
    },
    add_lights=False,
  )
  return clean_look(
    TerrainEntityCfg(terrain_type="generator", terrain_generator=copy.deepcopy(gen))
  )


def go2_piper_wbc_env_cfg(
  stage: int = 1,
  micro_rough: bool = False,
  play: bool = False,
  initial_curriculum_progress: float = 0.0,
  curriculum_steps: int = 3_000,
) -> ManagerBasedRlEnvCfg:
  """Build canonical 210D->18D Go2+Piper WBC mjlab environment config.

  Physics & delay alignment:
  - ``sim.mujoco.timestep = 0.005 s`` (200 Hz) and ``decimation = 4`` give
    ``policy_dt = 0.02 s`` (50 Hz), matching ``LeggedManip_Lab@b2cce16`` and
    ``go2_supermimic``.
  - Arm actuator lag defaults to ``DEFAULT_ARM_DELAY_LAG = (3, 9)`` physics
    steps (15-45 ms at 200 Hz), and leg lag defaults to ``(0, 6)`` (0-30 ms),
    sampled once per episode reset.
  """
  if stage not in (1, 2):
    raise ValueError(f"stage must be 1 or 2, got {stage}")

  from go2_supermimic.env.terrains import clean_look

  terrain_cfg = (
    _micro_rough_terrain()
    if micro_rough
    else clean_look(TerrainEntityCfg(terrain_type="plane", terrain_generator=None))
  )

  curriculum_progress = 1.0 if play else max(0.0, min(1.0, float(initial_curriculum_progress)))
  arm_delay = (0, 0) if play else DEFAULT_ARM_DELAY_LAG
  leg_delay = (0, 0) if play else DEFAULT_LEG_DELAY_LAG

  commands = {
    "base_velocity": WbcVelocityCommandCfg(
      entity_name="robot",
      resampling_time_range=(4.0, 6.0),
      rel_standing_envs=0.08,
      rel_single_axis_envs=0.20,
      curriculum_progress=curriculum_progress,
      debug_vis=not play,
    ),
    "ee_pose": WbcEePoseCommandCfg(
      entity_name="robot",
      body_name="end_effector",
      link_name="link0",
      resampling_time_range=(8.0, 10.0),
      stage=stage,
      curriculum_progress=curriculum_progress,
      trajectory_step_dt=0.02,
      debug_vis=not play,
    ),
  }

  actions = {
    "joint_pos": PiperSlewJointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=ACTION_SCALE,
      use_default_offset=True,
      clip={".*": (-10.0, 10.0)},
      max_arm_vel=3.0,
      policy_dt=0.02,
    ),
  }

  canonical_joints_cfg = SceneEntityCfg(
    "robot", joint_names=CANONICAL_JOINT_NAMES, preserve_order=True
  )
  actor_terms = {
    "base_ang_vel": ObservationTermCfg(
      func=envs_mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
      scale=0.2,
    ),
    "projected_gravity": ObservationTermCfg(
      func=envs_mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "joint_pos": ObservationTermCfg(
      func=envs_mdp.joint_pos_rel,
      params={"asset_cfg": canonical_joints_cfg},
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=envs_mdp.joint_vel_rel,
      params={"asset_cfg": canonical_joints_cfg},
      noise=Unoise(n_min=-1.5, n_max=1.5),
      scale=0.05,
    ),
    "actions": ObservationTermCfg(func=wbc_previous_action),
    "velocity_commands": ObservationTermCfg(
      func=envs_mdp.generated_commands,
      params={"command_name": "base_velocity"},
    ),
    "pos_commands": ObservationTermCfg(
      func=envs_mdp.generated_commands,
      params={"command_name": "ee_pose"},
    ),
  }
  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=not play,
      history_length=WBC_HISTORY_LENGTH,
      flatten_history_dim=True,
    ),
  }

  # Domain randomization skeleton (push_robot explicitly disabled).
  # NOTE: body_mass_scale (pseudo_inertia) runs BEFORE base_com (body_com_offset)
  # because pseudo_inertia rebuilds body_ipos from defaults before adding COM shifts.
  events = {
    "reset_base": EventTermCfg(
      mode="reset",
      func=envs_mdp.reset_root_state_uniform,
      params={
        "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-math.pi, math.pi)},
        "velocity_range": {},
      },
    ),
    "reset_robot_joints": EventTermCfg(
      mode="reset",
      func=envs_mdp.reset_joints_by_offset,
      params={
        "position_range": (-0.1, 0.1),
        "velocity_range": (0.0, 0.0),
      },
    ),
    "foot_friction_slide": EventTermCfg(
      mode="startup",
      func=envs_mdp.dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=FOOT_GEOM_NAMES),
        "operation": "abs",
        "distribution": "uniform",
        "axes": [0],
        "ranges": (0.5, 1.2),
        "shared_random": True,
      },
    ),
    "foot_friction_spin": EventTermCfg(
      mode="startup",
      func=envs_mdp.dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=FOOT_GEOM_NAMES),
        "operation": "abs",
        "distribution": "log_uniform",
        "axes": [1],
        "ranges": (1e-4, 2e-2),
        "shared_random": True,
      },
    ),
    "foot_friction_roll": EventTermCfg(
      mode="startup",
      func=envs_mdp.dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=FOOT_GEOM_NAMES),
        "operation": "abs",
        "distribution": "log_uniform",
        "axes": [2],
        "ranges": (1e-5, 5e-3),
        "shared_random": True,
      },
    ),
    "body_mass_scale": EventTermCfg(
      mode="startup",
      func=envs_mdp.dr.pseudo_inertia,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=(".*",)),
        "alpha_range": (-0.05, 0.05),
      },
    ),
    "base_com": EventTermCfg(
      mode="startup",
      func=envs_mdp.dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=("base",)),
        "operation": "add",
        "ranges": {0: (-0.05, 0.05), 1: (-0.05, 0.05), 2: (-0.01, 0.01)},
      },
    ),
    "leg_pd_gains": EventTermCfg(
      mode="reset",
      func=_randomize_actuator_pd_gains,
      params={
        "target_regex": r".*_(hip|thigh|calf)_joint",
        "kp_range": (0.5, 1.5),
        "kd_range": (0.5, 1.5),
      },
    ),
    "arm_pd_gains": EventTermCfg(
      mode="reset",
      func=_randomize_actuator_pd_gains,
      params={
        "target_regex": r"joint[1-6]",
        "kp_range": (0.5, 1.5),
        "kd_range": (0.5, 1.5),
      },
    ),
    "quiet_solver_warnings": EventTermCfg(
      mode="startup",
      func=_quiet_solver_warnings,
      params={},
    ),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(
      func=envs_mdp.bad_orientation,
      params={"limit_angle": math.radians(70.0)},
    ),
  }

  curriculum = {}
  if not play:
    curriculum = {
      "base_velocity_levels": CurriculumTermCfg(
        func=wbc_command_curriculum,
        params={
          "command_name": "base_velocity",
          "start_step": 0,
          "end_step": int(curriculum_steps),
          "init_progress": curriculum_progress,
        },
      ),
      "ee_pose_levels": CurriculumTermCfg(
        func=wbc_command_curriculum,
        params={
          "command_name": "ee_pose",
          "start_step": 0,
          "end_step": int(curriculum_steps),
          "init_progress": curriculum_progress,
        },
      ),
    }

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=terrain_cfg,
      entities={"robot": get_go2_piper_robot_cfg(leg_delay=leg_delay, arm_delay=arm_delay)},
      num_envs=50 if play else 4096,
      extent=2.0,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards={},
    terminations=terminations,
    curriculum=curriculum,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="base",
      distance=1.8,
      elevation=-15.0,
      azimuth=45.0,
    ),
    sim=SimulationCfg(
      nconmax=None,
      njmax=400,
      contact_sensor_maxmatch=64,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=16,
        ls_iterations=20,
        impratio=100,
        cone="elliptic",
        ccd_iterations=50,
      ),
    ),
    decimation=4,
    episode_length_s=int(1e9) if play else 20.0,
  )
