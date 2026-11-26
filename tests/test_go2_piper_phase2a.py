"""Phase 2A tests: Go2+Piper 18-DoF asset, 50 Hz slew action, WBC commands, and env cfg."""

from __future__ import annotations

import math
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.lab_api.math import euler_xyz_from_quat

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
  STAGE1_ORI_ABS_MAX,
  STAGE2_ORI_ABS_MAX,
  TRAJECTORY_Z_WORLD_MIN,
  mixed_curve_link0_points,
  trajectory_is_forbidden,
)

EXPECTED_TOTAL_MASS_KG = 22.3751
MAX_ARM_STEP_RAD = 3.0 * 0.02  # 0.060 rad/step at 50 Hz


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
    default = torch.tensor(DEFAULT_JOINT_POS, device=device).unsqueeze(0).repeat(num_envs, 1)
    # Minimal bodies: index 0 = link0, index 1 = end_effector
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
# 1. MJCF asset & physics verification
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

  # Canonical 18-DoF joint order in both MuJoCo and Entity.
  mj_joints = tuple(
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
    for j in range(model.njnt)
    if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE
  )
  assert mj_joints == CANONICAL_JOINT_NAMES
  assert tuple(entity.joint_names) == CANONICAL_JOINT_NAMES
  assert len(CANONICAL_JOINT_NAMES) == 18

  # Sites and IMU sensor.
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

  # Foot condim == 6.
  for fg in FOOT_GEOM_NAMES:
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, fg)
    assert gid >= 0
    assert model.geom_condim[gid] == 6

  # Actuator parameter parity (#460 golden baseline + leg PD + delay lag).
  actuators = cfg.articulation.actuators
  assert len(actuators) == 8  # leg_hip_thigh, leg_calf, joint1..joint6
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

  custom_cfg = get_go2_piper_robot_cfg(leg_delay=(1, 2), arm_delay=(4, 7))
  assert (
    custom_cfg.articulation.actuators[0].delay_min_lag,
    custom_cfg.articulation.actuators[0].delay_max_lag,
  ) == (1, 2)
  assert (
    custom_cfg.articulation.actuators[2].delay_min_lag,
    custom_cfg.articulation.actuators[2].delay_max_lag,
  ) == (4, 7)

  # Action scale & default joint angles.
  assert len(LEG_EFFORT_LIMIT) == 12
  assert len(LEG_VELOCITY_LIMIT) == 12
  assert all(v == pytest.approx(ACTION_SCALE) for v in GO2_PIPER_ACTION_SCALE.values())
  assert tuple(GO2_PIPER_ACTION_SCALE.keys()) == CANONICAL_JOINT_NAMES
  qpos0 = np.array(
    [cfg.init_state.joint_pos[".*_calf_joint"] if "calf" in n else cfg.init_state.joint_pos.get(n, 0.0) for n in CANONICAL_JOINT_NAMES]
  )
  np.testing.assert_allclose(qpos0, np.array(DEFAULT_JOINT_POS), atol=1e-7)


def test_go2_piper_self_collision_rules_and_real_contact() -> None:
  entity = Entity(get_go2_piper_robot_cfg())
  model = entity.spec.compile()
  data = mujoco.MjData(model)

  # Default standing pose & nominal nonzero arm pose produce zero self-contacts.
  for arm_q in ((0.0,) * 6, (0.2, 1.57, -1.35, -0.3, -0.5, 0.4)):
    mujoco.mj_resetData(model, data)
    data.qpos[2] = 0.35
    data.qpos[3] = 1.0
    data.qpos[7:25] = list(DEFAULT_JOINT_POS[:12]) + list(arm_q)
    mujoco.mj_forward(model, data)
    assert data.ncon == 0, f"Unexpected self-collision at arm_q={arm_q}"

  # Verify link4..link6 (incl welded link7/8) <-> link0 and link1..link6 <-> base are NOT excluded.
  excludes = set()
  for i in range(model.nexclude):
    # model.exclude_signature encodes (b1 << 16) | b2
    sig = int(model.exclude_signature[i])
    b1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, sig >> 16)
    b2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, sig & 0xFFFF)
    excludes.add(tuple(sorted((b1, b2))))

  for arm_body in ("link4", "link5", "link6"):
    assert tuple(sorted(("link0", arm_body))) not in excludes
  for arm_body in ("link1", "link2", "link3", "link4", "link5", "link6"):
    assert tuple(sorted(("base", arm_body))) not in excludes

  # Fold arm into link0 (front battery) and base (head D415) and verify real contacts.
  for bad_arm_q, expected_body in (
    ([0.0, 1.0, -0.2, 0.0, 1.2, 0.0], "link0"),
    ([0.0, 2.5, -1.0, 0.0, -1.2, 0.0], "base"),
  ):
    mujoco.mj_resetData(model, data)
    data.qpos[2] = 0.35
    data.qpos[3] = 1.0
    data.qpos[7:25] = list(DEFAULT_JOINT_POS[:12]) + bad_arm_q
    mujoco.mj_forward(model, data)
    assert data.ncon > 0
    contact_bodies = set()
    for i in range(data.ncon):
      b1 = mujoco.mj_id2name(
        model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[data.contact[i].geom1]
      )
      b2 = mujoco.mj_id2name(
        model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[data.contact[i].geom2]
      )
      contact_bodies.add(tuple(sorted((b1, b2))))
    assert any(
      expected_body in pair
      and any(arm in pair for arm in ("link1", "link2", "link3", "link4", "link5", "link6"))
      for pair in contact_bodies
    ), (expected_body, contact_bodies)


# ---------------------------------------------------------------------------
# 2. 50 Hz Piper slew-rate action tests
# ---------------------------------------------------------------------------


def test_slew_action_passthrough_slew_limits_clip_and_parity() -> None:
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

  # Out-of-clip network actions: raw_actions clipped to [-10, 10], legs pass through,
  # arm slews by <= 0.060 rad/step and stays within physical joint limits.
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
  # Check slew step from initial measured arm pos (0.0) -> clamped within [arm_low, arm_high].
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
  torch.testing.assert_close(term.processed_actions[:, 12:], expected_arm, atol=1e-7, rtol=1e-7)
  assert float(term.q_ref_arm.abs().max()) <= MAX_ARM_STEP_RAD + 1e-7
  # joint2 lower limit is 0.0, so env 1 joint2 must be clamped to 0.0!
  assert term.q_ref_arm[1, 1].item() == pytest.approx(0.0, abs=1e-7)
  # joint3 upper limit is 0.0, so env 0 joint3 must be clamped to 0.0!
  assert term.q_ref_arm[0, 2].item() == pytest.approx(0.0, abs=1e-7)

  # Multi-step rollout: verify pure slew_step_arm parity and <= 0.060 rad/step bound.
  prev = term.q_ref_arm.clone()
  for step_idx in range(30):
    act = torch.full((2, 18), 8.0 if step_idx < 15 else -8.0)
    term.process_actions(act)
    delta = (term.q_ref_arm - prev).abs().max().item()
    assert delta <= MAX_ARM_STEP_RAD + 1e-7
    expected = slew_step_arm(
      prev,
      term.raw_actions[:, 12:],
      default[:, 12:],
      0.25,
      arm_low.unsqueeze(0).expand(2, 6),
      arm_high.unsqueeze(0).expand(2, 6),
      MAX_ARM_STEP_RAD,
    )
    torch.testing.assert_close(term.q_ref_arm, expected, atol=1e-7, rtol=1e-7)
    prev = term.q_ref_arm.clone()

  # Partial reset: env 0 syncs to new measured joint pos, env 1 stays continuous.
  kept_env1 = term.q_ref_arm[1].clone()
  env.scene["robot"].data.joint_pos[0, 12:] = torch.tensor([0.5, 1.0, -1.0, 0.2, -0.3, 0.4])
  term.reset(torch.tensor([0]))
  assert torch.all(term.raw_actions[0] == 0.0)
  term.process_actions(torch.zeros(2, 18))
  expected_env0 = slew_step_arm(
    torch.tensor([[0.5, 1.0, -1.0, 0.2, -0.3, 0.4]]),
    torch.zeros(1, 6),
    torch.zeros(1, 6),
    0.25,
    arm_low.unsqueeze(0),
    arm_high.unsqueeze(0),
    MAX_ARM_STEP_RAD,
  )
  torch.testing.assert_close(term.q_ref_arm[0:1], expected_env0, atol=1e-7, rtol=1e-7)
  assert float((term.q_ref_arm[1] - kept_env1).abs().max()) <= MAX_ARM_STEP_RAD + 1e-7

  # Pure slew_step_arm supports autograd for 2B/2C differentiable imagination.
  q_prev = torch.zeros(4, 6, requires_grad=True)
  raw_a = torch.ones(4, 6, requires_grad=True)
  out = slew_step_arm(q_prev, raw_a, 0.0, 0.25, -2.0, 2.0, MAX_ARM_STEP_RAD)
  out.sum().backward()
  assert q_prev.grad is not None and torch.isfinite(q_prev.grad).all()

  # Invalid non-finite action raises ValueError.
  bad_act = torch.zeros(2, 18)
  bad_act[0, 13] = float("nan")
  with pytest.raises(ValueError):
    term.process_actions(bad_act)


# ---------------------------------------------------------------------------
# 3. WBC velocity & mixed-frame EE pose command tests
# ---------------------------------------------------------------------------


def test_wbc_velocity_command_sampling_and_curriculum() -> None:
  torch.manual_seed(42)
  num_envs = 4096
  env = _mock_env(num_envs)
  cfg = WbcVelocityCommandCfg(
    resampling_time_range=(4.0, 6.0),
    rel_standing_envs=0.08,
    rel_single_axis_envs=0.20,
    curriculum_progress=0.0,
  )
  cmd = cfg.build(env)
  assert isinstance(cmd, WbcVelocityCommand)
  cmd.reset(torch.arange(num_envs))

  # Curriculum progress = 0 -> [-0.2, 0.2]
  vel = cmd.command
  assert vel.shape == (num_envs, 3)
  assert float(vel.abs().max()) <= 0.2 + 1e-6

  standing = (vel == 0.0).all(dim=-1)
  standing_frac = float(standing.float().mean())
  assert 0.065 <= standing_frac <= 0.095, standing_frac

  non_standing = vel[~standing]
  nonzero_axes = (non_standing.abs() > 1e-9).sum(dim=-1)
  single_axis = nonzero_axes == 1
  single_frac = float(single_axis.float().mean())
  assert 0.175 <= single_frac <= 0.225, single_frac

  # Pure vx / pure vy / pure wz roughly 1/3 each.
  sa_cmds = non_standing[single_axis]
  axis_counts = [(sa_cmds[:, i].abs() > 1e-9).float().mean().item() for i in range(3)]
  for frac in axis_counts:
    assert 0.25 <= frac <= 0.42, axis_counts

  # Curriculum progress -> 1.0 expands smoothly to vx [-1, 1], vy [-0.6, 0.6], wz [-1, 1].
  cmd.set_curriculum_progress(0.5)
  assert cmd.cfg.ranges.lin_vel_x == pytest.approx((-0.6, 0.6))
  assert cmd.cfg.ranges.lin_vel_y == pytest.approx((-0.4, 0.4))
  assert cmd.cfg.ranges.ang_vel_z == pytest.approx((-0.6, 0.6))

  cmd.set_curriculum_progress(1.0)
  cmd.reset(torch.arange(num_envs))
  vel_full = cmd.command
  assert float(vel_full[:, 0].max()) > 0.9 and float(vel_full[:, 0].min()) < -0.9
  assert float(vel_full[:, 1].max()) <= 0.6 + 1e-6 and float(vel_full[:, 1].max()) > 0.5
  assert float(vel_full[:, 2].max()) > 0.9 and float(vel_full[:, 2].min()) < -0.9


def test_wbc_ee_pose_stage1_stage2_box_and_traj_determinism() -> None:
  torch.manual_seed(7)
  num_envs = 2048

  # Stage 1: <= 20 deg attitude cap, no continuous trajectories, no LINK0_BODY_BOX violations.
  env1 = _mock_env(num_envs)
  cmd1 = WbcEePoseCommandCfg(stage=1, curriculum_progress=1.0).build(env1)
  assert isinstance(cmd1, WbcEePoseCommand)
  cmd1.reset(torch.arange(num_envs))
  pose1 = cmd1.command
  assert pose1.shape == (num_envs, 7)
  assert float(cmd1._traj_active.float().sum()) == 0.0
  # Unit WXYZ quaternions with qw >= 0.
  norms1 = pose1[:, 3:7].norm(dim=-1)
  torch.testing.assert_close(norms1, torch.ones_like(norms1), atol=1e-6, rtol=1e-6)
  assert bool((pose1[:, 3] >= -1e-7).all())
  r1, p1, y1 = euler_xyz_from_quat(pose1[:, 3:7])
  cap1 = math.radians(20.0) + 1e-4
  assert float(r1.abs().max()) <= cap1
  assert float(p1.abs().max()) <= cap1
  assert float(y1.abs().max()) <= cap1
  assert not bool(cmd1._positions_forbidden(torch.arange(num_envs), pose1[:, :3]).any())

  # Stage 2: wide attitude, ~40% trajectories, LINK0_BODY_BOX clear, exact 20D deterministic rollout.
  env2 = _mock_env(num_envs)
  cmd2 = WbcEePoseCommandCfg(stage=2, curriculum_progress=1.0).build(env2)
  cmd2.reset(torch.arange(num_envs))
  active_frac = float(cmd2._traj_active.float().mean())
  assert 0.34 <= active_frac <= 0.46, active_frac

  # Check wide attitude coverage across non-traj (goal) poses and all sampled poses.
  r2, p2, y2 = euler_xyz_from_quat(cmd2.traj_state[:, 13:17])
  assert float(r2.abs().max()) > math.radians(80.0)
  assert float(r2.abs().max()) <= STAGE2_ORI_ABS_MAX[0] + 1e-4
  assert float(p2.abs().max()) <= STAGE2_ORI_ABS_MAX[1] + 1e-4
  assert float(y2.abs().max()) <= STAGE2_ORI_ABS_MAX[2] + 1e-4

  # Verify LINK0_BODY_BOX constants exported and active during 100-step deterministic rollout.
  assert LINK0_BODY_BOX_MIN == (-0.50, -0.24, -0.50)
  assert LINK0_BODY_BOX_MAX == (0.38, 0.24, 0.15)
  assert SAFE_HOVER_POSITION == (0.55, 0.0, 0.45)

  state = cmd2.traj_state.clone()
  assert state.shape == (num_envs, TRAJ_STATE_DIM)
  # Check pack/unpack roundtrip.
  unpacked = unpack_ee_traj_state(state)
  repacked = pack_ee_traj_state(*unpacked)
  torch.testing.assert_close(repacked, state, atol=0.0, rtol=0.0)

  # Run 100 steps with fixed command timer (prevent resampling) and assert < 1e-6 parity.
  cmd2.time_left.fill_(100.0)
  link0_pos = env2.scene["robot"].data.body_link_pos_w[:, 0]
  link0_quat = env2.scene["robot"].data.body_link_quat_w[:, 0]
  for _ in range(100):
    cmd2.compute(dt=0.02)
    state, expected_cmd = advance_ee_traj_state(state, 0.02, make_quat_unique=True)
    torch.testing.assert_close(cmd2.command, expected_cmd, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(cmd2.traj_state, state, atol=1e-6, rtol=1e-6)
    local, z_w, singular = mixed_curve_link0_points(
      cmd2.command[:, :3].unsqueeze(1), link0_pos, link0_quat
    )
    assert not bool(trajectory_is_forbidden(local, z_w, singular).any())
    assert bool((z_w >= TRAJECTORY_Z_WORLD_MIN - 1e-6).all())


# ---------------------------------------------------------------------------
# 4. Real mjlab GPU 50-step rollout (Stage 1 & Stage 2)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA GPU")
@pytest.mark.parametrize("stage", [1, 2])
def test_go2_piper_wbc_gpu_50_step_rollout_and_obs_contract(stage: int) -> None:
  torch.manual_seed(123 + stage)
  cfg = go2_piper_wbc_env_cfg(stage=stage, micro_rough=False, play=True)
  cfg.scene.num_envs = 16
  assert "push_robot" not in cfg.events
  for dr_key in (
    "foot_friction_slide",
    "foot_friction_spin",
    "foot_friction_roll",
    "base_com",
    "body_mass_scale",
    "leg_pd_gains",
    "arm_pd_gains",
  ):
    assert dr_key in cfg.events

  env = ManagerBasedRlEnv(cfg, device="cuda:0")
  try:
    obs, _ = env.reset()
    actor_obs = obs["actor"]
    assert actor_obs.shape == (16, 210)
    assert bool(torch.isfinite(actor_obs).all())

    action_term = env.action_manager.get_term("joint_pos")
    assert isinstance(action_term, PiperSlewJointPositionAction)
    prev_q_ref_arm = action_term.q_ref_arm.clone()
    robot = env.scene["robot"]

    # Check term-major 7 terms x 3 frames layout and exact physical parity in play mode.
    expected_order = [
      ("base_ang_vel", 3),
      ("projected_gravity", 3),
      ("joint_pos", 18),
      ("joint_vel", 18),
      ("actions", 18),
      ("velocity_commands", 3),
      ("pos_commands", 7),
    ]
    assert WBC_OBS_TERMS == tuple(expected_order)

    for step_idx in range(50):
      actions = (torch.rand(16, 18, device="cuda:0") * 2.0 - 1.0) * 4.0
      obs, _, _, _, _ = env.step(actions)
      actor_obs = obs["actor"]
      assert actor_obs.shape == (16, 210)
      assert bool(torch.isfinite(actor_obs).all())

      # Continuous slew check: exclude envs that reset on the previous step
      # (episode_length_buf == 1 right after their first post-reset step) or on
      # the current step (episode_length_buf == 0).
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

    # Verify newest frame (frame index 2 in [t-2, t-1, t] term-major layout)
    # of every observation term matches scaled physical / command state.
    offset = 0
    term_slices = {}
    for name, dim in expected_order:
      block = actor_obs[:, offset : offset + 3 * dim].reshape(16, 3, dim)
      term_slices[name] = block[:, -1, :]  # newest frame
      offset += 3 * dim
    assert offset == 210

    imu_ang_vel = robot.data.root_link_ang_vel_b
    torch.testing.assert_close(term_slices["base_ang_vel"], 0.2 * imu_ang_vel, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(
      term_slices["projected_gravity"], robot.data.projected_gravity_b, atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(
      term_slices["joint_pos"],
      robot.data.joint_pos - robot.data.default_joint_pos,
      atol=1e-5,
      rtol=1e-5,
    )
    torch.testing.assert_close(
      term_slices["joint_vel"], 0.05 * robot.data.joint_vel, atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(
      term_slices["actions"], wbc_previous_action(env), atol=1e-6, rtol=1e-6
    )
    torch.testing.assert_close(
      term_slices["velocity_commands"],
      env.command_manager.get_command("base_velocity"),
      atol=1e-6,
      rtol=1e-6,
    )
    torch.testing.assert_close(
      term_slices["pos_commands"],
      env.command_manager.get_command("ee_pose"),
      atol=1e-6,
      rtol=1e-6,
    )
  finally:
    env.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA GPU")
@pytest.mark.parametrize("stage", [1, 2])
def test_go2_piper_wbc_gpu_train_cfg_with_delays_and_dr(stage: int) -> None:
  """Verify non-play training cfg (with actuator delays 3..9, DR, and obs noise) runs 50 steps."""
  torch.manual_seed(99 + stage)
  cfg = go2_piper_wbc_env_cfg(stage=stage, micro_rough=False, play=False)
  cfg.scene.num_envs = 16
  env = ManagerBasedRlEnv(cfg, device="cuda:0")
  try:
    obs, _ = env.reset()
    assert obs["actor"].shape == (16, 210)
    assert bool(torch.isfinite(obs["actor"]).all())
    action_term = env.action_manager.get_term("joint_pos")
    prev_q = action_term.q_ref_arm.clone()
    for step_idx in range(50):
      actions = (torch.rand(16, 18, device="cuda:0") * 2.0 - 1.0) * 3.0
      obs, _, _, _, _ = env.step(actions)
      assert obs["actor"].shape == (16, 210)
      assert bool(torch.isfinite(obs["actor"]).all())
      if step_idx >= 1:
        continuous = env.episode_length_buf >= 2
        if bool(continuous.any()):
          diff = (action_term.q_ref_arm[continuous] - prev_q[continuous]).abs().max().item()
          assert diff <= MAX_ARM_STEP_RAD + 1e-6
      prev_q = action_term.q_ref_arm.clone()
  finally:
    env.close()
