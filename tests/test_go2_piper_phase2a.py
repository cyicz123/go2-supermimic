"""Phase 2A tests: Go2+Piper 18-DoF asset, DelayedPD, slew action, WBC commands, and env cfg."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch
from setuptools.dist import Distribution

from mjlab.actuator import ActuatorCmd
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.lab_api.math import euler_xyz_from_quat, quat_mul

from go2_supermimic.robot.go2_piper import (
  ACTION_SCALE,
  ARM_EFFORT_LIMIT,
  ARM_JOINT_NAMES,
  ARM_KD,
  ARM_KP,
  ARM_VELOCITY_LIMIT,
  ARMATURE,
  CANONICAL_JOINT_NAMES,
  DEFAULT_ARM_DELAY_LAG,
  DEFAULT_JOINT_POS,
  DEFAULT_LEG_DELAY_LAG,
  FOOT_GEOM_NAMES,
  FOOT_SITE_NAMES,
  GO2_PIPER_ACTION_SCALE,
  GO2_PIPER_XML,
  LEG_EFFORT_LIMIT,
  LEG_JOINT_NAMES,
  LEG_KD,
  LEG_KP,
  LEG_VELOCITY_LIMIT,
  DelayedPdActuator,
  DelayedPdActuatorCfg,
  get_go2_piper_robot_cfg,
  get_spec,
)
from go2_supermimic.wbc import (
  LINK0_BODY_BOX_MAX,
  LINK0_BODY_BOX_MIN,
  SAFE_HOVER_POSITION,
  TRAJ_STATE_DIM,
  WBC_OBS_TERMS,
  PiperSlewJointPositionAction,
  PiperSlewJointPositionActionCfg,
  WbcEePoseCommand,
  WbcEePoseCommandCfg,
  WbcVelocityCommand,
  WbcVelocityCommandCfg,
  advance_ee_traj_state,
  go2_piper_wbc_env_cfg,
  pack_ee_traj_state,
  slew_step_arm,
  unpack_ee_traj_state,
  wbc_previous_action,
)
from go2_supermimic.wbc.commands import (
  STAGE2_ORI_ABS_MAX,
  TRAJECTORY_Z_WORLD_MIN,
  mixed_curve_link0_points,
  slerp_wxyz,
  trajectory_is_forbidden,
)

EXPECTED_TOTAL_MASS_KG = 22.3751
MAX_ARM_STEP_RAD = 3.0 * 0.02  # 0.060 rad/step at 50 Hz
REPO_ROOT = Path(__file__).resolve().parents[1]


class _MockEntity:
  """Minimal Entity stub for unit-testing action and command terms on CPU."""

  def __init__(
    self,
    num_envs: int,
    *,
    limits: torch.Tensor | None = None,
    device: str = "cpu",
  ) -> None:
    self.num_envs = num_envs
    self.device = device
    default = (
      torch.tensor(DEFAULT_JOINT_POS, device=device)
      .unsqueeze(0)
      .repeat(num_envs, 1)
    )
    body_pos = torch.zeros(num_envs, 2, 3, device=device)
    body_pos[:, 0, 2] = 0.44
    body_pos[:, 1, 0] = 0.45
    body_pos[:, 1, 2] = 0.44
    body_quat = torch.zeros(num_envs, 2, 4, device=device)
    body_quat[:, :, 0] = 1.0  # WXYZ identity
    self.data = SimpleNamespace(
      default_joint_pos=default,
      joint_pos=default.clone(),
      joint_pos_limits=limits,
      encoder_bias=torch.zeros_like(default),
      root_link_lin_vel_b=torch.zeros(num_envs, 3, device=device),
      root_link_ang_vel_b=torch.zeros(num_envs, 3, device=device),
      body_link_pos_w=body_pos,
      body_link_quat_w=body_quat,
    )

  def find_joints_by_actuator_names(
    self, actuator_name_keys: tuple[str, ...] | str
  ) -> tuple[list[int], list[str]]:
    del actuator_name_keys
    return list(range(len(CANONICAL_JOINT_NAMES))), list(CANONICAL_JOINT_NAMES)

  def find_bodies(self, name: str) -> tuple[list[int], list[str]]:
    if name == "link0":
      return [0], ["link0"]
    if name == "end_effector":
      return [1], ["end_effector"]
    return [], []

  def set_joint_position_target(
    self, target: torch.Tensor, joint_ids: torch.Tensor | None = None
  ) -> None:
    del target, joint_ids


def _mock_env(num_envs: int, **entity_kwargs) -> SimpleNamespace:
  entity = _MockEntity(num_envs, **entity_kwargs)
  return SimpleNamespace(
    num_envs=num_envs,
    device="cpu",
    step_dt=0.02,
    scene={"robot": entity},
  )


# ---------------------------------------------------------------------------
# 1. MJCF asset, self-collision (incl. end_effector), and wheel packaging
# ---------------------------------------------------------------------------


def test_go2_piper_mjcf_mass_joints_sites_and_actuators() -> None:
  assert GO2_PIPER_XML.exists()
  spec = get_spec()
  raw_model = spec.compile()
  assert float(np.sum(raw_model.body_mass)) == pytest.approx(
    EXPECTED_TOTAL_MASS_KG, abs=1e-4
  )

  cfg = get_go2_piper_robot_cfg()
  entity = Entity(cfg)
  model = entity.spec.compile()
  assert float(np.sum(model.body_mass)) == pytest.approx(
    EXPECTED_TOTAL_MASS_KG, abs=1e-4
  )

  mj_joints = tuple(
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
    for j in range(model.njnt)
    if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE
  )
  assert mj_joints == CANONICAL_JOINT_NAMES
  assert tuple(entity.joint_names) == CANONICAL_JOINT_NAMES
  assert len(CANONICAL_JOINT_NAMES) == 18

  site_names = {
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, s) for s in range(model.nsite)
  }
  for required_site in (*FOOT_SITE_NAMES, "link0", "end_effector", "imu"):
    assert required_site in site_names
  sensor_names = {
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, s)
    for s in range(model.nsensor)
  }
  assert "imu_ang_vel" in sensor_names

  for fg in FOOT_GEOM_NAMES:
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, fg)
    assert gid >= 0
    assert model.geom_condim[gid] == 6

  actuators = cfg.articulation.actuators
  assert len(actuators) == 8
  for act in actuators:
    assert isinstance(act, DelayedPdActuatorCfg)
  hip_thigh, calf = actuators[0], actuators[1]
  assert (hip_thigh.stiffness, hip_thigh.damping) == (LEG_KP, LEG_KD)
  assert (calf.stiffness, calf.damping) == (LEG_KP, LEG_KD)
  assert hip_thigh.effort_limit == pytest.approx(23.7)
  assert calf.effort_limit == pytest.approx(45.43)
  assert hip_thigh.velocity_limit == pytest.approx(30.1)
  assert calf.velocity_limit == pytest.approx(15.7)
  assert hip_thigh.armature == pytest.approx(ARMATURE)
  assert calf.armature == pytest.approx(ARMATURE)
  assert (hip_thigh.delay_min_lag, hip_thigh.delay_max_lag) == DEFAULT_LEG_DELAY_LAG

  for i, arm_act in enumerate(actuators[2:]):
    assert arm_act.target_names_expr == (ARM_JOINT_NAMES[i],)
    assert arm_act.stiffness == pytest.approx(ARM_KP[i])
    assert arm_act.damping == pytest.approx(ARM_KD[i])
    assert arm_act.effort_limit == pytest.approx(ARM_EFFORT_LIMIT[i])
    assert arm_act.velocity_limit == pytest.approx(ARM_VELOCITY_LIMIT[i])
    assert arm_act.armature == pytest.approx(ARMATURE)
    assert (arm_act.delay_min_lag, arm_act.delay_max_lag) == DEFAULT_ARM_DELAY_LAG

  assert len(LEG_EFFORT_LIMIT) == 12
  assert len(LEG_VELOCITY_LIMIT) == 12
  assert all(v == pytest.approx(ACTION_SCALE) for v in GO2_PIPER_ACTION_SCALE.values())
  assert tuple(GO2_PIPER_ACTION_SCALE.keys()) == CANONICAL_JOINT_NAMES
  qpos0 = np.array(
    [
      cfg.init_state.joint_pos[".*_calf_joint"]
      if "calf" in n
      else cfg.init_state.joint_pos.get(n, 0.0)
      for n in CANONICAL_JOINT_NAMES
    ]
  )
  np.testing.assert_allclose(qpos0, np.array(DEFAULT_JOINT_POS), atol=1e-7)


def test_go2_piper_self_collision_including_end_effector_and_battery_d415() -> None:
  entity = Entity(get_go2_piper_robot_cfg())
  model = entity.spec.compile()
  data = mujoco.MjData(model)

  # end_effector_collision exists and has collision bitmask enabled.
  ee_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "end_effector_collision")
  assert ee_gid >= 0
  assert model.geom_contype[ee_gid] == 1
  assert model.geom_conaffinity[ee_gid] == 1

  # Negative samples: default standing pose & nominal nonzero pose produce 0 contacts.
  for arm_q in ((0.0,) * 6, (0.2, 1.57, -1.35, -0.3, -0.5, 0.4)):
    mujoco.mj_resetData(model, data)
    data.qpos[2] = 0.35
    data.qpos[3] = 1.0
    data.qpos[7:25] = list(DEFAULT_JOINT_POS[:12]) + list(arm_q)
    mujoco.mj_forward(model, data)
    assert data.ncon == 0, f"Unexpected self-collision at arm_q={arm_q}"

  # Verify link4..link6 / end_effector <-> link0 and link1..link6 / end_effector <-> base are NOT excluded.
  excludes = set()
  for i in range(model.nexclude):
    sig = int(model.exclude_signature[i])
    b1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, sig >> 16)
    b2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, sig & 0xFFFF)
    excludes.add(tuple(sorted((b1, b2))))

  for arm_body in ("link4", "link5", "link6", "end_effector"):
    assert tuple(sorted(("link0", arm_body))) not in excludes
  for arm_body in ("link1", "link2", "link3", "link4", "link5", "link6", "end_effector"):
    assert tuple(sorted(("base", arm_body))) not in excludes

  def _contact_geom_pairs(arm_pose: list[float]) -> set[tuple[str, str]]:
    mujoco.mj_resetData(model, data)
    data.qpos[2] = 0.35
    data.qpos[3] = 1.0
    data.qpos[7:25] = list(DEFAULT_JOINT_POS[:12]) + arm_pose
    mujoco.mj_forward(model, data)
    out = set()
    for c in range(data.ncon):
      g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, data.contact[c].geom1)
      g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, data.contact[c].geom2)
      out.add(tuple(sorted((g1, g2))))
    return out

  # Sweep poses and verify real contacts across link4, link5, link6/wrist, end_effector
  # against front_battery_collision_box (link0) and base/d415_head_camera_collision_camera_main.
  all_geom_pairs: set[tuple[str, str]] = set()
  for j2 in np.linspace(0.2, 3.0, 15):
    for j3 in (-0.2, -0.7, -1.2, -2.0, -2.7):
      for j5 in (-1.2, 0.0, 1.2):
        all_geom_pairs |= _contact_geom_pairs([0.0, float(j2), float(j3), 0.0, float(j5), 0.0])

  assert any(
    "front_battery_collision_box" in pair and "end_effector_collision" in pair
    for pair in all_geom_pairs
  ), all_geom_pairs
  assert any(
    "front_battery_collision_box" in pair
    and any("link6" in g or "d415_wrist" in g or "gripper_base" in g or "grip_collision" in g for g in pair)
    for pair in all_geom_pairs
  ), all_geom_pairs
  assert any(
    ("base_collision" in pair or "d415_head_camera_collision_camera_main" in pair)
    and any(arm_g in g for g in pair for arm_g in ("link4", "link5"))
    for pair in all_geom_pairs
  ), all_geom_pairs
  assert any(
    ("base_collision" in pair or "d415_head_camera_collision_camera_main" in pair)
    and "end_effector_collision" in pair
    for pair in all_geom_pairs
  ), all_geom_pairs


def test_pyproject_package_data_includes_all_go2_piper_assets() -> None:
  import tomllib

  pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
  pkg_patterns = pyproject["tool"]["setuptools"]["package-data"]["go2_supermimic"]
  for required in (
    "robot/go2_piper/*.xml",
    "robot/go2_piper/assets/*.obj",
    "robot/go2_piper/assets/*.stl",
    "robot/go2_piper/assets/*.STL",
  ):
    assert required in pkg_patterns

  dist = Distribution(
    {
      "name": "go2-supermimic",
      "version": "0.1.0",
      "packages": ["go2_supermimic", "go2_supermimic.robot", "go2_supermimic.robot.go2_piper", "go2_supermimic.wbc"],
      "package_data": {"go2_supermimic": pkg_patterns},
    }
  )
  cmd = dist.get_command_obj("build_py")
  cmd.ensure_finalized()
  # build_py resolves package_data files relative to package_dir (REPO_ROOT)
  data_files = set()
  for pkg, src_dir, _build_dir, filenames in cmd.data_files:
    for fn in filenames:
      data_files.add(f"{src_dir}/{fn}")
  assert "go2_supermimic/robot/go2_piper/go2_piper.xml" in data_files
  piper_assets = [f for f in data_files if f.startswith("go2_supermimic/robot/go2_piper/assets/")]
  assert len(piper_assets) == 39, f"Expected 39 Piper assets, got {len(piper_assets)}"


# ---------------------------------------------------------------------------
# 2. DelayedPdActuator constant effort & per-episode delay lag semantics
# ---------------------------------------------------------------------------


def test_delayed_pd_actuator_constant_effort_and_episode_constant_lag() -> None:
  torch.manual_seed(42)
  cfg = get_go2_piper_robot_cfg(leg_delay=(1, 4), arm_delay=(3, 9))
  entity = Entity(cfg)
  model = entity.spec.compile()
  model.opt.timestep = 0.005

  # Check actuator control_law and compute() strictly deliver full constant effort_limit
  # across 0.0, 2.25, and 2.99 rad/s (Isaac Lab #460 DelayedPDActuator formula, zero velocity derating).
  arm_act = entity.actuators[2]
  assert isinstance(arm_act, DelayedPdActuator)
  n_envs = 8
  pos = torch.zeros(n_envs, 1)
  pos_tgt = torch.full((n_envs, 1), 2.0)
  vel_tgt = torch.zeros(n_envs, 1)
  eff_tgt = torch.zeros(n_envs, 1)
  params = {
    "stiffness": torch.full((n_envs, 1), ARM_KP[0]),
    "damping": torch.full((n_envs, 1), ARM_KD[0]),
    "force_limit": torch.full((n_envs, 1), ARM_EFFORT_LIMIT[0]),
  }
  for vel_val in (0.0, 2.25, 2.99, -2.25, -2.99):
    vel = torch.full((n_envs, 1), vel_val)
    tau = DelayedPdActuator.control_law(params, ActuatorCmd(pos_tgt, vel_tgt, eff_tgt, pos, vel))
    torch.testing.assert_close(tau, torch.full((n_envs, 1), ARM_EFFORT_LIMIT[0]))

  # Check via mjlab Simulation on CPU:
  # 1) No actuator is fused into FusedActuatorGroup (each arm joint + leg group stays independent).
  # 2) All 6 arm joints sample independent per-env lags (not collapsed to a scalar), while each leg
  #    group (leg_hip_thigh, leg_calf) uses one per-env lag shared across its joints (b2cce16 parity),
  #    remaining strictly constant within the episode and resampling only reset rows on partial reset.
  # 3) arm_act.compute() at qd=2.99 and qd=2.25 rad/s still outputs full constant 11.8125 Nm.
  from mjlab.sim import MujocoCfg, Simulation, SimulationCfg

  sim = Simulation(
    num_envs=16,
    cfg=SimulationCfg(mujoco=MujocoCfg(timestep=0.005)),
    model=model,
    device="cpu",
  )
  entity.initialize(model, sim.model, sim.data, "cpu")
  assert len(entity._fused_actuator_group._groups) == 0
  assert len(entity._custom_actuators) == 8

  for vel_val in (2.25, 2.99):
    tau_j1 = arm_act.compute(
      ActuatorCmd(
        position_target=torch.full((16, 1), 2.0),
        velocity_target=torch.zeros(16, 1),
        effort_target=torch.zeros(16, 1),
        pos=torch.zeros(16, 1),
        vel=torch.full((16, 1), vel_val),
      )
    )
    torch.testing.assert_close(tau_j1, torch.full((16, 1), ARM_EFFORT_LIMIT[0]))

  entity.reset()
  arm_acts = entity.actuators[2:8]
  arm_lags_ep1 = torch.stack([act.current_lags for act in arm_acts], dim=1).clone()
  assert arm_lags_ep1.shape == (16, 6)
  assert bool((arm_lags_ep1 >= 3).all() and (arm_lags_ep1 <= 9).all())
  assert not bool((arm_lags_ep1 == arm_lags_ep1[:, :1]).all(dim=1).all()), (
    f"Arm 6-joint lags should be sampled independently, got {arm_lags_ep1}"
  )
  assert entity.actuators[0].current_lags.shape == (16,)
  assert entity.actuators[1].current_lags.shape == (16,)
  assert bool((entity.actuators[0].current_lags >= 1).all() and (entity.actuators[0].current_lags <= 4).all())
  assert bool((entity.actuators[1].current_lags >= 1).all() and (entity.actuators[1].current_lags <= 4).all())

  # Step 20 physics substeps (`sim.step()`): lags stay strictly constant within the episode,
  # and a 0.1 rad step target jump on joint1 at step 10 arrives at joint1 ctrl after exact lag[e] steps.
  lags_j1_ep1 = arm_lags_ep1[:, 0].clone()
  j1_ctrl_id = int(arm_acts[0].global_ctrl_ids[0])
  j1_dof_id = int(arm_acts[0]._target_ids[0])
  tgt_zero = torch.zeros(16, 18)
  tgt_step = torch.zeros(16, 18)
  tgt_step[:, j1_dof_id] = 0.1
  for step in range(20):
    entity.set_joint_position_target(tgt_zero if step < 10 else tgt_step)
    entity.write_data_to_sim()
    current_arm_lags = torch.stack([act.current_lags for act in arm_acts], dim=1)
    assert torch.equal(current_arm_lags, arm_lags_ep1)
    if step >= 10:
      elapsed_since_jump = step - 10
      arrived = sim.data.ctrl[:, j1_ctrl_id] > 1.0
      expected_arrived = elapsed_since_jump >= lags_j1_ep1
      assert torch.equal(arrived, expected_arrived), (step, arrived, expected_arrived)
    sim.step()
    entity.update(0.005)

  # Partial reset on subset [0, 5] resamples them while other envs stay untouched.
  other = [i for i in range(16) if i not in (0, 5)]
  lags_other_before = arm_lags_ep1[other].clone()
  entity.reset(torch.tensor([0, 5]))
  arm_lags_after = torch.stack([act.current_lags for act in arm_acts], dim=1)
  assert torch.equal(arm_lags_after[other], lags_other_before)
  assert not torch.equal(arm_lags_after[[0, 5]], arm_lags_ep1[[0, 5]])


def test_delayed_pd_partial_reset_preserves_unreset_delay_queue_and_velocity_limit() -> None:
  from mjlab.sim import MujocoCfg, Simulation, SimulationCfg

  # Regression 1: Fixed lag=3 across partial reset (including ManagerBasedRlEnv reset write_data_to_sim):
  # resetting env 0 between steps must NOT advance env 1's delay queue by 1 extra step.
  cfg_lag3 = get_go2_piper_robot_cfg(leg_delay=(0, 0), arm_delay=(3, 3))
  entity = Entity(cfg_lag3)
  model = entity.spec.compile()
  sim = Simulation(
    num_envs=2,
    cfg=SimulationCfg(mujoco=MujocoCfg(timestep=0.005)),
    model=model,
    device="cpu",
  )
  entity.initialize(model, sim.model, sim.data, "cpu")
  sim.reset()
  entity.reset()
  entity.write_data_to_sim()
  sim.forward()

  j1_act = entity.actuators[2]
  j1_ctrl_id = int(j1_act.global_ctrl_ids[0])
  j1_dof_id = int(j1_act._target_ids[0])
  assert torch.equal(j1_act.current_lags, torch.full((2,), 3, dtype=torch.long))

  tgt = torch.zeros(2, 18)
  tgt[:, j1_dof_id] = 2.0

  env1_torques: list[float] = []
  for step in range(5):
    entity.set_joint_position_target(tgt)
    entity.write_data_to_sim()
    env1_torques.append(float(sim.data.ctrl[1, j1_ctrl_id]))
    sim.step()
    entity.update(0.005)
    # Trigger a partial reset + post-reset write_data_to_sim() + sim.forward() on env 0 after step 1
    # (matching ManagerBasedRlEnv.step auto-reset at manager_based_rl_env.py:472).
    if step == 1:
      p_ids = torch.tensor([0], dtype=torch.long)
      sim.reset(p_ids)
      entity.reset(p_ids)
      entity.write_data_to_sim()
      sim.forward()

  assert abs(env1_torques[0]) < 0.05
  assert abs(env1_torques[1]) < 0.05
  assert abs(env1_torques[2]) < 0.05
  assert env1_torques[3] == pytest.approx(ARM_EFFORT_LIMIT[0], abs=1e-4)
  assert env1_torques[4] == pytest.approx(ARM_EFFORT_LIMIT[0], abs=1e-4)

  # Regression 2: Real ManagerBasedRlEnv partial _reset_idx + write_data_to_sim() preserves env 1 lag queue.
  env_cfg = go2_piper_wbc_env_cfg(stage=1, micro_rough=False, play=False)
  env_cfg.scene.num_envs = 2
  env_cfg.decimation = 1
  env_cfg.events.pop("arm_pd_gains", None)
  env_cfg.events.pop("leg_pd_gains", None)
  env = ManagerBasedRlEnv(env_cfg, device="cpu")
  try:
    env.reset(seed=7)
    robot = env.scene["robot"]
    env_j1_act = robot.actuators[2]
    env_j1_act.set_lags(3)
    env_j1_ctrl = int(env_j1_act.global_ctrl_ids[0])
    env_j1_dof = int(env_j1_act._target_ids[0])
    step_tgt = robot.data.joint_pos.clone()
    step_tgt[:, env_j1_dof] += 2.0
    # Run step 0 & step 1 with explicit step target, reset env 0 after step 1 via _reset_idx + write_data_to_sim.
    seq_env1: list[float] = []
    tau_zero_ref: list[float] = []
    for k in range(5):
      q1 = float(robot.data.joint_pos[1, env_j1_dof])
      qd1 = float(robot.data.joint_vel[1, env_j1_dof])
      tau_zero_ref.append(float(max(-ARM_EFFORT_LIMIT[0], min(ARM_EFFORT_LIMIT[0], -ARM_KP[0] * q1 - ARM_KD[0] * qd1))))
      robot.set_joint_position_target(step_tgt)
      env.scene.write_data_to_sim()
      seq_env1.append(float(env.sim.data.ctrl[1, env_j1_ctrl]))
      env.sim.step()
      env.scene.update(dt=env.physics_dt)
      if k == 1:
        p_ids = torch.tensor([0], dtype=torch.long)
        env._reset_idx(p_ids)
        env.scene.write_data_to_sim()
        env.sim.forward()
    assert seq_env1[0] == pytest.approx(tau_zero_ref[0], abs=1e-4)
    assert seq_env1[1] == pytest.approx(tau_zero_ref[1], abs=1e-4)
    assert seq_env1[2] == pytest.approx(tau_zero_ref[2], abs=1e-4)
    assert seq_env1[3] == pytest.approx(ARM_EFFORT_LIMIT[0], abs=1e-4)
  finally:
    env.close()

  # Regression 3: Decoupled actuator torque vs sim-level velocity_limit_sim enforcement.
  # - At qd=2.99 and 2.25 rad/s, actuator outputs FULL constant 11.8125 Nm (no derating).
  # - Across multi-pose arm states including Codex gravity-coupled case
  #   q_arm=[-0.09737, 0.11746, 0.17631, -0.14673, 0.17384, 0.03743] with joint2 qd=3.0 & tau=0,
  #   all 18 joints strictly satisfy |qd| <= velocity_limit + 1e-6 after every physics substep.
  cfg_vel = get_go2_piper_robot_cfg(leg_delay=(0, 0), arm_delay=(0, 0))
  v_entity = Entity(cfg_vel)
  v_model = v_entity.spec.compile()
  n_vel_envs = 8
  v_sim = Simulation(
    num_envs=n_vel_envs,
    cfg=SimulationCfg(mujoco=MujocoCfg(timestep=0.005)),
    model=v_model,
    device="cpu",
  )
  v_entity.initialize(v_model, v_sim.model, v_sim.data, "cpu")
  v_sim.reset()
  v_entity.reset()
  j1_ctrl = int(v_entity.actuators[2].global_ctrl_ids[0])
  j1_dof = int(v_entity.actuators[2]._target_ids[0])

  qpos_init = v_entity.data.default_joint_pos.clone()
  qvel_init = torch.zeros(n_vel_envs, 18)
  # Env 0..3: extreme +3.0, -3.0, +2.25, +2.99 rad/s on joint1
  qvel_init[0, j1_dof] = 3.0
  qvel_init[1, j1_dof] = -3.0
  qvel_init[2, j1_dof] = 2.25
  qvel_init[3, j1_dof] = 2.99
  # Env 4: Codex exact gravity-coupled random arm pose with joint2 vel=3.0 rad/s and tau=0
  codex_q_arm = torch.tensor([-0.09737, 0.11746, 0.17631, -0.14673, 0.17384, 0.03743])
  qpos_init[4, 12:18] = codex_q_arm
  qvel_init[4, 13] = 3.0  # joint2
  # Env 5..7: random arm poses with large initial velocities across multiple joints
  torch.manual_seed(99)
  qpos_init[5:8, 12:18] = torch.empty(3, 6).uniform_(-0.5, 0.5).clamp(
    v_entity.data.joint_pos_limits[5:8, 12:18, 0] + 0.05,
    v_entity.data.joint_pos_limits[5:8, 12:18, 1] - 0.05,
  )
  qvel_init[5:8, 12:18] = torch.empty(3, 6).uniform_(-3.0, 3.0)
  v_entity.write_joint_state_to_sim(qpos_init, qvel_init)
  v_sim.forward()

  vel_tgt = torch.zeros(n_vel_envs, 18)
  vel_tgt[0, j1_dof] = 2.0
  vel_tgt[1, j1_dof] = -2.0
  vel_tgt[2, j1_dof] = 2.0
  vel_tgt[3, j1_dof] = 2.0
  # Env 4 keeps tau=0 (zero gains temporarily on env 4 to test pure zero-torque gravity coupling)
  for a in v_entity.actuators:
    a.set_gains(torch.tensor([4]), kp=torch.zeros(1, len(a.target_ids)), kd=torch.zeros(1, len(a.target_ids)))
  vel_tgt[5:8, 12:18] = torch.tensor([[2.0, 2.0, -2.0, 2.0, -2.0, 2.0]] * 3)
  v_entity.set_joint_position_target(vel_tgt)
  v_entity.write_data_to_sim()

  assert float(v_sim.data.ctrl[2, j1_ctrl]) == pytest.approx(ARM_EFFORT_LIMIT[0], abs=1e-5)
  assert float(v_sim.data.ctrl[3, j1_ctrl]) == pytest.approx(ARM_EFFORT_LIMIT[0], abs=1e-5)
  assert float(v_sim.data.ctrl[4, j1_ctrl]) == pytest.approx(0.0, abs=1e-6)

  vel_limits = torch.tensor(LEG_VELOCITY_LIMIT + ARM_VELOCITY_LIMIT)
  for _ in range(20):
    v_entity.set_joint_position_target(vel_tgt)
    v_entity.write_data_to_sim()
    v_sim.step()
    v_entity.update(0.005)
    assert bool((v_entity.data.joint_vel.abs() <= vel_limits[None, :] + 1e-6).all()), (
      f"Exceeded velocity limit: max_excess={(v_entity.data.joint_vel.abs() - vel_limits[None, :]).max().item()}"
    )

  # Regression 4 (Codex Round 4 probe): joint6 at default pose with initial vel=3.0 rad/s
  # and full 9.5844 Nm effort must integrate at most v_lim*dt per substep (<= 0.015 rad in 5 ms,
  # <= 0.060 rad in 20 ms / 4 substeps), AND partial reset must NOT pull reset env back to old qpos_prev.
  cfg_j6 = get_go2_piper_robot_cfg(leg_delay=(0, 0), arm_delay=(0, 0))
  j6_entity = Entity(cfg_j6)
  j6_model = j6_entity.spec.compile()
  j6_sim = Simulation(
    num_envs=2,
    cfg=SimulationCfg(mujoco=MujocoCfg(timestep=0.005)),
    model=j6_model,
    device="cpu",
  )
  j6_entity.initialize(j6_model, j6_sim.model, j6_sim.data, "cpu")
  j6_sim.reset()
  j6_entity.reset()

  j6_dof = 17  # canonical index of joint6
  j6_ctrl = int(j6_entity.actuators[7].global_ctrl_ids[0])
  qpos_j6 = j6_entity.data.default_joint_pos.clone()
  qvel_j6 = torch.zeros(2, 18)
  qvel_j6[:, j6_dof] = 3.0
  j6_entity.write_joint_state_to_sim(qpos_j6, qvel_j6)
  j6_sim.forward()

  tgt_j6 = j6_entity.data.default_joint_pos.clone()
  tgt_j6[:, j6_dof] = 2.0  # drives full 9.5844 Nm positive torque
  q0_j6 = j6_entity.data.joint_pos[:, j6_dof].clone()

  for step in range(4):
    j6_entity.set_joint_position_target(tgt_j6)
    j6_entity.write_data_to_sim()
    assert float(j6_sim.data.ctrl[0, j6_ctrl]) == pytest.approx(ARM_EFFORT_LIMIT[5], abs=1e-4)
    j6_sim.step()
    j6_entity.update(0.005)
    assert bool((j6_entity.data.joint_vel[:, j6_dof].abs() <= 3.0 + 1e-6).all())
    if step == 0:
      dq_1 = (j6_entity.data.joint_pos[:, j6_dof] - q0_j6).abs()
      assert bool((dq_1 <= 3.0 * 0.005 + 1e-5).all()), f"1-step dq exceeded 0.015: {dq_1}"

  dq_4 = (j6_entity.data.joint_pos[:, j6_dof] - q0_j6).abs()
  assert bool((dq_4 <= 3.0 * 0.02 + 1e-5).all()), f"4-step dq exceeded 0.060: {dq_4}"

  # Partial reset env 0 to a far pose (-1.2 rad) after old episode reached +0.06 rad;
  # verify first post-reset step does NOT pull env 0 back toward old qpos_prev.
  p0 = torch.tensor([0], dtype=torch.long)
  j6_sim.reset(p0)
  j6_entity.reset(p0)
  new_q = j6_entity.data.default_joint_pos.clone()
  new_q[0, j6_dof] = -1.2
  new_v = torch.zeros(2, 18)
  new_v[0, j6_dof] = 3.0
  j6_entity.write_joint_state_to_sim(new_q[0:1], new_v[0:1], env_ids=p0)
  j6_sim.forward()
  tgt_reset = tgt_j6.clone()
  tgt_reset[0, j6_dof] = 1.0  # saturated +9.5844 Nm from -1.2 rad at vel=+3.0
  j6_entity.set_joint_position_target(tgt_reset)
  j6_entity.write_data_to_sim()
  j6_sim.step()
  j6_entity.update(0.005)
  dq_after_reset = float(j6_entity.data.joint_pos[0, j6_dof] - (-1.2))
  assert abs(dq_after_reset) <= 3.0 * 0.005 + 1e-5, f"Reset env pulled/overshot: {dq_after_reset}"
  assert float(j6_entity.data.joint_pos[0, j6_dof]) < -1.15

  # Regression 5 (Codex Round 5 probe): Stage 1 play env with joint2 starting near upper limit
  # (3.137 rad, +3.0 rad/s) under target 3.14 rad across 20 x 5ms substeps must NEVER snap back
  # past -v_lim*dt when MuJoCo soft limit lets q_t cross 3.14 rad.
  play_cfg = go2_piper_wbc_env_cfg(stage=1, micro_rough=False, play=True)
  play_cfg.scene.num_envs = 2
  play_cfg.events.pop("arm_pd_gains", None)
  play_cfg.events.pop("leg_pd_gains", None)
  play_env = ManagerBasedRlEnv(play_cfg, device="cpu")
  try:
    play_env.reset(seed=11)
    p_robot = play_env.scene["robot"]
    j2_dof = 13  # joint2
    q_play = p_robot.data.default_joint_pos.clone()
    q_play[:, j2_dof] = 3.137
    v_play = torch.zeros_like(q_play)
    v_play[:, j2_dof] = 3.0
    p_robot.write_joint_state_to_sim(q_play, v_play)
    play_env.sim.forward()
    tgt_play = p_robot.data.default_joint_pos.clone()
    tgt_play[:, j2_dof] = 3.14

    q_prev_j2 = p_robot.data.joint_pos[:, j2_dof].clone()
    for _ in range(20):
      p_robot.set_joint_position_target(tgt_play)
      play_env.scene.write_data_to_sim()
      play_env.sim.step()
      play_env.scene.update(dt=play_env.physics_dt)
      q_curr_j2 = p_robot.data.joint_pos[:, j2_dof].clone()
      dq_step = (q_curr_j2 - q_prev_j2).abs()
      assert bool((dq_step <= 3.0 * 0.005 + 1e-5).all()), (
        f"Soft-limit step jump exceeded 0.015 rad: dq={q_curr_j2 - q_prev_j2}, q={q_curr_j2}"
      )
      assert bool((p_robot.data.joint_vel[:, j2_dof].abs() <= 3.0 + 1e-6).all())
      q_prev_j2 = q_curr_j2
  finally:
    play_env.close()


# ---------------------------------------------------------------------------
# 3. 50 Hz Piper slew-rate action tests (immediate q_ref_arm sync on reset)
# ---------------------------------------------------------------------------


def test_slew_action_passthrough_slew_limits_reset_sync_and_parity() -> None:
  limits = torch.zeros(2, 18, 2)
  limits[:, :12, 0] = -4.0
  limits[:, :12, 1] = 4.0
  arm_low = torch.tensor([-2.168, 0.0, -2.967, -1.745, -1.22, -2.0944])
  arm_high = torch.tensor([2.168, 3.14, 0.0, 1.745, 1.22, 2.0944])
  limits[:, 12:, 0] = arm_low
  limits[:, 12:, 1] = arm_high

  env = _mock_env(2, limits=limits)
  cfg = PiperSlewJointPositionActionCfg(
    entity_name="robot",
    actuator_names=(".*",),
    scale=0.25,
    use_default_offset=True,
    clip={".*": (-10.0, 10.0)},
    max_arm_vel=3.0,
    policy_dt=0.02,
  )
  term = cfg.build(env)
  assert isinstance(term, PiperSlewJointPositionAction)

  raw_in = torch.full((2, 18), 25.0)
  raw_in[1, 12:] = -25.0
  term.process_actions(raw_in)
  assert torch.all(term.raw_actions[0] == 10.0)
  assert torch.all(term.raw_actions[1, :12] == 10.0)
  assert torch.all(term.raw_actions[1, 12:] == -10.0)

  default = env.scene["robot"].data.default_joint_pos
  torch.testing.assert_close(
    term.processed_actions[:, :12],
    default[:, :12] + 0.25 * term.raw_actions[:, :12],
  )
  expected_arm = slew_step_arm(
    torch.zeros(2, 6),
    term.raw_actions[:, 12:],
    default[:, 12:],
    0.25,
    arm_low.unsqueeze(0).expand(2, 6),
    arm_high.unsqueeze(0).expand(2, 6),
    MAX_ARM_STEP_RAD,
  )
  torch.testing.assert_close(term.q_ref_arm, expected_arm, atol=1e-7, rtol=1e-7)
  assert term.q_ref_arm[1, 1].item() == pytest.approx(0.0, abs=1e-7)
  assert term.q_ref_arm[0, 2].item() == pytest.approx(0.0, abs=1e-7)

  prev = term.q_ref_arm.clone()
  for step_idx in range(25):
    act = torch.full((2, 18), 8.0 if step_idx < 12 else -8.0)
    term.process_actions(act)
    assert (term.q_ref_arm - prev).abs().max().item() <= MAX_ARM_STEP_RAD + 1e-7
    prev = term.q_ref_arm.clone()

  # Partial reset: q_ref_arm[0] IMMEDIATELY reflects clamped measured pose right after reset()
  # (including clamping out-of-limit measured position), while env 1 keeps its reference.
  kept_env1 = term.q_ref_arm[1].clone()
  new_meas_env0 = torch.tensor([0.5, -0.3, -1.0, 0.2, -0.3, 0.4])  # -0.3 on j2 is < 0.0 limit
  expected_clamped_env0 = torch.tensor([0.5, 0.0, -1.0, 0.2, -0.3, 0.4])
  env.scene["robot"].data.joint_pos[0, 12:] = new_meas_env0
  term.reset(torch.tensor([0]))
  assert torch.all(term.raw_actions[0] == 0.0)
  torch.testing.assert_close(term.q_ref_arm[0], expected_clamped_env0, atol=1e-7, rtol=1e-7)
  torch.testing.assert_close(term.q_ref_arm[1], kept_env1, atol=1e-7, rtol=1e-7)

  # Next step proceeds strictly from the published q_ref_arm!
  term.process_actions(torch.zeros(2, 18))
  expected_env0 = slew_step_arm(
    expected_clamped_env0.unsqueeze(0),
    torch.zeros(1, 6),
    torch.zeros(1, 6),
    0.25,
    arm_low.unsqueeze(0),
    arm_high.unsqueeze(0),
    MAX_ARM_STEP_RAD,
  )
  torch.testing.assert_close(term.q_ref_arm[0:1], expected_env0, atol=1e-7, rtol=1e-7)
  assert float((term.q_ref_arm[1] - kept_env1).abs().max()) <= MAX_ARM_STEP_RAD + 1e-7

  q_prev = torch.zeros(4, 6, requires_grad=True)
  raw_a = torch.ones(4, 6, requires_grad=True)
  out = slew_step_arm(q_prev, raw_a, 0.0, 0.25, -2.0, 2.0, MAX_ARM_STEP_RAD)
  out.sum().backward()
  assert q_prev.grad is not None and torch.isfinite(q_prev.grad).all()

  bad_act = torch.zeros(2, 18)
  bad_act[0, 13] = float("nan")
  with pytest.raises(ValueError):
    term.process_actions(bad_act)


# ---------------------------------------------------------------------------
# 4. WBC velocity, EE pose commands, 20D trajectory math & autograd
# ---------------------------------------------------------------------------


def test_wbc_velocity_command_sampling_and_curriculum() -> None:
  torch.manual_seed(42)
  num_envs = 4096
  env = _mock_env(num_envs)
  cmd = WbcVelocityCommandCfg(
    resampling_time_range=(4.0, 6.0),
    rel_standing_envs=0.08,
    rel_single_axis_envs=0.20,
    curriculum_progress=0.0,
  ).build(env)
  assert isinstance(cmd, WbcVelocityCommand)
  cmd.reset(torch.arange(num_envs))

  vel = cmd.command
  assert vel.shape == (num_envs, 3)
  assert float(vel.abs().max()) <= 0.2 + 1e-6

  standing = (vel == 0.0).all(dim=-1)
  assert 0.065 <= float(standing.float().mean()) <= 0.095

  non_standing = vel[~standing]
  single_axis = (non_standing.abs() > 1e-9).sum(dim=-1) == 1
  assert 0.175 <= float(single_axis.float().mean()) <= 0.225

  sa_cmds = non_standing[single_axis]
  axis_counts = [(sa_cmds[:, i].abs() > 1e-9).float().mean().item() for i in range(3)]
  for frac in axis_counts:
    assert 0.25 <= frac <= 0.42

  cmd.set_curriculum_progress(1.0)
  cmd.reset(torch.arange(num_envs))
  vel_full = cmd.command
  assert float(vel_full[:, 0].max()) > 0.9 and float(vel_full[:, 0].min()) < -0.9
  assert float(vel_full[:, 1].max()) <= 0.6 + 1e-6 and float(vel_full[:, 1].max()) > 0.5
  assert float(vel_full[:, 2].max()) > 0.9 and float(vel_full[:, 2].min()) < -0.9


def test_wbc_ee_pose_stage1_stage2_box_and_pending_reset() -> None:
  torch.manual_seed(7)
  num_envs = 2048

  env1 = _mock_env(num_envs)
  cmd1 = WbcEePoseCommandCfg(stage=1, curriculum_progress=1.0).build(env1)
  assert isinstance(cmd1, WbcEePoseCommand)
  cmd1.reset(torch.arange(num_envs))
  cmd1.compute(dt=0.0)
  pose1 = cmd1.command
  assert float(cmd1._traj_active.float().sum()) == 0.0
  torch.testing.assert_close(pose1[:, 3:7].norm(dim=-1), torch.ones(num_envs), atol=1e-6, rtol=1e-6)
  assert bool((pose1[:, 3] >= -1e-7).all())
  r1, p1, y1 = euler_xyz_from_quat(pose1[:, 3:7])
  cap1 = math.radians(20.0) + 1e-4
  assert float(r1.abs().max()) <= cap1
  assert float(p1.abs().max()) <= cap1
  assert float(y1.abs().max()) <= cap1
  assert not bool(cmd1._positions_forbidden(torch.arange(num_envs), pose1[:, :3]).any())

  env2 = _mock_env(num_envs)
  cmd2 = WbcEePoseCommandCfg(stage=2, curriculum_progress=1.0).build(env2)
  cmd2.reset(torch.arange(num_envs))
  cmd2.compute(dt=0.0)
  assert 0.34 <= float(cmd2._traj_active.float().mean()) <= 0.46

  r2, p2, y2 = euler_xyz_from_quat(cmd2.traj_state[:, 13:17])
  assert float(r2.abs().max()) > math.radians(80.0)
  assert float(r2.abs().max()) <= STAGE2_ORI_ABS_MAX[0] + 1e-4
  assert float(p2.abs().max()) <= STAGE2_ORI_ABS_MAX[1] + 1e-4
  assert float(y2.abs().max()) <= STAGE2_ORI_ABS_MAX[2] + 1e-4

  assert LINK0_BODY_BOX_MIN == (-0.50, -0.24, -0.50)
  assert LINK0_BODY_BOX_MAX == (0.38, 0.24, 0.15)
  assert SAFE_HOVER_POSITION == (0.55, 0.0, 0.45)

  cmd2.time_left.fill_(100.0)
  link0_pos = env2.scene["robot"].data.body_link_pos_w[:, 0]
  link0_quat = env2.scene["robot"].data.body_link_quat_w[:, 0]
  state = cmd2.traj_state.clone()
  for _ in range(100):
    cmd2.compute(dt=0.02)
    state, expected_cmd = advance_ee_traj_state(state, 0.02, make_quat_unique=True)
    torch.testing.assert_close(cmd2.command, expected_cmd, atol=1e-6, rtol=1e-6)
    local, z_w, singular = mixed_curve_link0_points(
      cmd2.command[:, :3].unsqueeze(1), link0_pos, link0_quat
    )
    assert not bool(trajectory_is_forbidden(local, z_w, singular).any())
    assert bool((z_w >= TRAJECTORY_Z_WORLD_MIN - 1e-6).all())


def test_advance_ee_traj_state_dims_zero_dur_autograd_and_analytical() -> None:
  # 1. Arbitrary leading dimensions: (20,), (4, 20), (2, 3, 20)
  p0 = torch.tensor([0.45, -0.1, 0.3])
  p1 = torch.tensor([0.55, 0.2, 0.6])
  p2 = torch.tensor([0.65, 0.1, 0.5])
  q0 = torch.tensor([1.0, 0.0, 0.0, 0.0])
  ang = 0.7
  q1 = torch.tensor([math.cos(ang / 2), math.sin(ang / 2), 0.0, 0.0])  # roll = 0.7 rad
  dur = torch.tensor(2.0)
  s1 = pack_ee_traj_state(p0, p1, p2, q0, q1, torch.tensor(0.0), dur, torch.tensor(1.0))
  assert s1.shape == (TRAJ_STATE_DIM,)
  ns1, cmd1 = advance_ee_traj_state(s1, 0.5)
  assert ns1.shape == (TRAJ_STATE_DIM,) and cmd1.shape == (7,)

  # Independent analytical Bezier + SLERP reference at t = 0.5, dur = 2.0 -> u = 0.25
  u = 0.25
  s = 10 * u**3 - 15 * u**4 + 6 * u**5
  ref_pos = (1 - s) ** 2 * p0 + 2 * (1 - s) * s * p1 + s**2 * p2
  ref_ang = s * ang
  ref_quat = torch.tensor([math.cos(ref_ang / 2), math.sin(ref_ang / 2), 0.0, 0.0])
  torch.testing.assert_close(cmd1[:3], ref_pos, atol=1e-6, rtol=1e-6)
  torch.testing.assert_close(cmd1[3:7], ref_quat, atol=1e-6, rtol=1e-6)

  s2 = s1.unsqueeze(0).expand(4, TRAJ_STATE_DIM).clone()
  ns2, cmd2 = advance_ee_traj_state(s2, 0.5)
  assert ns2.shape == (4, TRAJ_STATE_DIM) and cmd2.shape == (4, 7)
  torch.testing.assert_close(cmd2[0], cmd1, atol=1e-7, rtol=1e-7)

  s3 = s1.view(1, 1, TRAJ_STATE_DIM).expand(2, 3, TRAJ_STATE_DIM).clone()
  ns3, cmd3 = advance_ee_traj_state(s3, 0.5)
  assert ns3.shape == (2, 3, TRAJ_STATE_DIM) and cmd3.shape == (2, 3, 7)
  torch.testing.assert_close(cmd3[1, 2], cmd1, atol=1e-7, rtol=1e-7)

  # 2. duration <= 0 with dt = 0: immediately lands on goal (p2, q1) and stays steady
  s_zero_dur = pack_ee_traj_state(p0, p1, p2, q0, q1, torch.tensor(0.0), torch.tensor(0.0), torch.tensor(1.0))
  ns_z0, cmd_z0 = advance_ee_traj_state(s_zero_dur, 0.0)
  ns_z1, cmd_z1 = advance_ee_traj_state(ns_z0, 0.02)
  assert ns_z0[19].item() == 0.0
  torch.testing.assert_close(cmd_z0[:3], p2, atol=1e-6, rtol=1e-6)
  torch.testing.assert_close(cmd_z0, cmd_z1, atol=1e-6, rtol=1e-6)

  # 3. Autograd finite check for equal AND antipodal quaternions
  for q1_base in (q0.clone(), -q0.clone()):
    st = pack_ee_traj_state(p0, p1, p2, q0, q1_base, torch.tensor(0.0), dur, torch.tensor(1.0))
    st = st.unsqueeze(0).clone().requires_grad_(True)
    _, out = advance_ee_traj_state(st, 0.5)
    out.sum().backward()
    assert st.grad is not None and bool(torch.isfinite(st.grad).all())

    qg0 = q0.unsqueeze(0).clone().requires_grad_(True)
    qg1 = q1_base.unsqueeze(0).clone().requires_grad_(True)
    q_slerp = slerp_wxyz(qg0, qg1, torch.tensor([0.3]))
    q_slerp.sum().backward()
    assert qg0.grad is not None and bool(torch.isfinite(qg0.grad).all())
    assert qg1.grad is not None and bool(torch.isfinite(qg1.grad).all())

  # 4. make_quat_unique=False with antipodal goal q1 = -ref_q1 (dot(q0, q1) < 0):
  # No sign jump between trajectory end and subsequent static step.
  q1_anti = -q1
  st_anti = pack_ee_traj_state(p0, p1, p2, q0, q1_anti, torch.tensor(1.98), dur, torch.tensor(1.0))
  st_end, cmd_end = advance_ee_traj_state(st_anti, 0.02, make_quat_unique=False)
  st_after, cmd_after = advance_ee_traj_state(st_end, 0.02, make_quat_unique=False)
  assert float(torch.sum(cmd_end[3:7] * cmd_after[3:7])) > 0.9999
  torch.testing.assert_close(cmd_end, cmd_after, atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# 5. Real mjlab GPU tests: rollout, 3-frame obs, curriculum, DR order, reset parity
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA GPU")
@pytest.mark.parametrize("stage", [1, 2])
def test_go2_piper_wbc_gpu_50_step_rollout_and_3frame_obs_contract(stage: int) -> None:
  torch.manual_seed(123 + stage)
  cfg = go2_piper_wbc_env_cfg(stage=stage, micro_rough=False, play=True)
  cfg.scene.num_envs = 16
  assert "push_robot" not in cfg.events

  env = ManagerBasedRlEnv(cfg, device="cuda:0")
  try:
    obs, _ = env.reset()
    actor_obs = obs["actor"]
    assert actor_obs.shape == (16, 210)
    assert bool(torch.isfinite(actor_obs).all())

    action_term = env.action_manager.get_term("joint_pos")
    assert isinstance(action_term, PiperSlewJointPositionAction)
    robot = env.scene["robot"]

    def _current_70d() -> dict[str, torch.Tensor]:
      return {
        "base_ang_vel": 0.2 * robot.data.root_link_ang_vel_b.clone(),
        "projected_gravity": robot.data.projected_gravity_b.clone(),
        "joint_pos": (robot.data.joint_pos - robot.data.default_joint_pos).clone(),
        "joint_vel": 0.05 * robot.data.joint_vel.clone(),
        "actions": wbc_previous_action(env).clone(),
        "velocity_commands": env.command_manager.get_command("base_velocity").clone(),
        "pos_commands": env.command_manager.get_command("ee_pose").clone(),
      }

    def _slice_3frames(obs_210: torch.Tensor) -> dict[str, torch.Tensor]:
      out = {}
      offset = 0
      for name, dim in WBC_OBS_TERMS:
        out[name] = obs_210[:, offset : offset + 3 * dim].reshape(16, 3, dim)
        offset += 3 * dim
      assert offset == 210
      return out

    # P2: Right after reset(), all 3 history frames must be backfilled with t0.
    f0 = _current_70d()
    slices0 = _slice_3frames(actor_obs)
    for name, _ in WBC_OBS_TERMS:
      for slot in range(3):
        torch.testing.assert_close(slices0[name][:, slot, :], f0[name], atol=1e-4, rtol=1e-4)

    # Track 3 consecutive frames across 3 steps and verify ALL 3 frames [t-2, t-1, t].
    frame_hist = [f0]
    for _ in range(3):
      act = torch.zeros(16, 18, device="cuda:0")
      obs, _, _, _, _ = env.step(act)
      frame_hist.append(_current_70d())
    slices3 = _slice_3frames(obs["actor"])
    for name, _ in WBC_OBS_TERMS:
      torch.testing.assert_close(slices3[name][:, 0, :], frame_hist[-3][name], atol=1e-4, rtol=1e-4)
      torch.testing.assert_close(slices3[name][:, 1, :], frame_hist[-2][name], atol=1e-4, rtol=1e-4)
      torch.testing.assert_close(slices3[name][:, 2, :], frame_hist[-1][name], atol=1e-4, rtol=1e-4)

    # Full 50-step rollout with random actions, continuous arm slew check, and GPU velocity_limit_sim check.
    vel_limits_cuda = torch.tensor(LEG_VELOCITY_LIMIT + ARM_VELOCITY_LIMIT, device="cuda:0")
    prev_q_ref_arm = action_term.q_ref_arm.clone()
    for step_idx in range(50):
      actions = (torch.rand(16, 18, device="cuda:0") * 2.0 - 1.0) * 4.0
      obs, _, _, _, _ = env.step(actions)
      actor_obs = obs["actor"]
      assert actor_obs.shape == (16, 210)
      assert bool(torch.isfinite(actor_obs).all())
      assert bool((robot.data.joint_vel.abs() <= vel_limits_cuda + 1e-5).all())
      if step_idx >= 1:
        continuous = env.episode_length_buf >= 2
        if bool(continuous.any()):
          arm_diff = (
            (action_term.q_ref_arm[continuous] - prev_q_ref_arm[continuous])
            .abs()
            .max()
            .item()
          )
          assert arm_diff <= MAX_ARM_STEP_RAD + 1e-6
      prev_q_ref_arm = action_term.q_ref_arm.clone()
  finally:
    env.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA GPU")
def test_go2_piper_wbc_gpu_curriculum_startup_dr_and_post_forward_reset() -> None:
  # P1 #3, #4, #6, #7 in real GPU train-mode ManagerBasedRlEnv
  cfg = go2_piper_wbc_env_cfg(stage=2, micro_rough=False, play=False, curriculum_steps=100)
  cfg.scene.num_envs = 16
  env = ManagerBasedRlEnv(cfg, device="cuda:0")
  try:
    # P1 #7: Verify startup DR produced BOTH mass variation and cm-scale base COM offset.
    base_body_id = env.scene["robot"].find_bodies("base")[0][0]
    global_base_id = int(env.scene["robot"].indexing.body_ids[base_body_id])
    base_ipos_x = env.sim.model.body_ipos[:, global_base_id, 0]
    total_masses = env.sim.model.body_mass.sum(dim=-1)
    assert float(base_ipos_x.std()) > 5e-3, f"base COM x offset wiped: std={base_ipos_x.std()}"
    assert float(total_masses.std()) > 0.1, f"body_mass_scale inactive: std={total_masses.std()}"

    # P1 #3: Verify curriculum advances from narrow window (step 0) -> full Stage 2 range (step 100).
    env.reset()
    vel_term = env.command_manager.get_term("base_velocity")
    pose_term = env.command_manager.get_term("ee_pose")
    assert vel_term.cfg.ranges.lin_vel_x == pytest.approx((-0.2, 0.2))
    assert pose_term.cfg.ranges.pos_x == pytest.approx((0.40, 0.45))
    assert pose_term.cfg.ranges.roll == pytest.approx((0.0, 0.0))

    env.common_step_counter = 50
    env.curriculum_manager.compute()
    assert vel_term.cfg.ranges.lin_vel_x == pytest.approx((-0.6, 0.6))
    assert pose_term.cfg.ranges.roll[1] == pytest.approx(math.radians(50.0), abs=1e-4)

    env.common_step_counter = 100
    env.curriculum_manager.compute()
    assert vel_term.cfg.ranges.lin_vel_x == pytest.approx((-1.0, 1.0))
    assert pose_term.cfg.ranges.pos_x == pytest.approx((0.45, 0.70))
    assert pose_term.cfg.ranges.roll == pytest.approx((-math.radians(100.0), math.radians(100.0)))
    assert pose_term.cfg.trajectory_mode_prob == pytest.approx(0.40)

    # P1 #4 & #6: Inject 60 deg tilt before reset(), verify post-reset ee_pose matches
    # a clean upright reset under identical seed, AND action q_ref_arm matches measured joint pos.
    obs_clean, _ = env.reset(seed=777)
    clean_pose_cmd = env.command_manager.get_command("ee_pose").clone()
    clean_q_ref = env.action_manager.get_term("joint_pos").q_ref_arm.clone()
    assert torch.isfinite(obs_clean["actor"]).all()

    # Tilt robot by 60 deg around X and step sim.forward() to pollute pre-reset body_link_quat_w.
    tilt_pose = torch.zeros(16, 7, device="cuda:0")
    tilt_pose[:, 2] = 0.35
    tilt_pose[:, 3] = math.cos(math.radians(30.0))
    tilt_pose[:, 4] = math.sin(math.radians(30.0))
    env.scene["robot"].write_root_link_pose_to_sim(tilt_pose)
    env.sim.forward()

    env.reset(seed=777)
    tilted_reset_pose_cmd = env.command_manager.get_command("ee_pose").clone()
    torch.testing.assert_close(tilted_reset_pose_cmd, clean_pose_cmd, atol=1e-6, rtol=1e-6)

    # Also verify partial _reset_idx + sim.forward() + command_manager.compute(0.0, env_ids)
    # ignores pre-reset tilt and immediately syncs q_ref_arm to clamped joint pos.
    action_term = env.action_manager.get_term("joint_pos")
    env.step(torch.full((16, 18), 5.0, device="cuda:0"))
    kept_q_ref_env1 = action_term.q_ref_arm[1].clone()
    env.scene["robot"].write_root_link_pose_to_sim(tilt_pose)
    env.sim.forward()
    partial_ids = torch.tensor([0, 3], device="cuda:0", dtype=torch.long)
    env._reset_idx(partial_ids)
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.command_manager.compute(dt=0.0, env_ids=partial_ids)
    # q_ref_arm[0, 3] immediately matches newly reset arm joint pos; env 1 is untouched.
    arm_ids = action_term._target_ids[action_term._arm_indices]
    meas_arm = env.scene["robot"].data.joint_pos[partial_ids][:, arm_ids]
    torch.testing.assert_close(action_term.q_ref_arm[partial_ids], meas_arm, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(action_term.q_ref_arm[1], kept_q_ref_env1, atol=1e-6, rtol=1e-6)
  finally:
    env.close()
