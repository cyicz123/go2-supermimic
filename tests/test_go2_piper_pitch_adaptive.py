"""Tests for WBC pitch reward recalibration and 15-bin adaptive EE command sampling."""

from __future__ import annotations

import math
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from go2_supermimic.robot.go2_piper import DEFAULT_JOINT_POS
from go2_supermimic.supermimic.buffer import WbcReplayBuffer
from go2_supermimic.wbc import (
  WBC_ACTOR_OBS_DIM,
  WBC_CONTEXT_DIM,
  WBC_CRITIC_BASE_DIM,
  WBC_NUM_ARM_JOINTS,
  WBC_NUM_JOINTS,
  WBC_NUM_LEG_JOINTS,
  WbcEePoseCommand,
  WbcEePoseCommandCfg,
  WbcImagineCfg,
  WbcImagination,
  WbcRewardWeights,
  WbcRobotConsts,
  go2_piper_wbc_env_cfg,
)
from go2_supermimic.wbc import state as wbc_st
from go2_supermimic.wbc.trainer import WbcTrainCfg, WbcTrainer


def _synthetic_consts(num_envs: int, device: str = "cpu") -> WbcRobotConsts:
  dev = torch.device(device)
  default_q = (
    torch.tensor(DEFAULT_JOINT_POS, device=dev, dtype=torch.float32)
    .unsqueeze(0)
    .expand(num_envs, WBC_NUM_JOINTS)
    .clone()
  )
  action_scale = torch.full((num_envs, WBC_NUM_JOINTS), 0.25, device=dev, dtype=torch.float32)
  kp = torch.tensor([40.0] * 12 + [50.0, 65.0, 55.0, 25.0, 20.0, 15.0], device=dev)
  kd = torch.tensor([1.0] * 12 + [2.0, 3.0, 2.0, 0.8, 0.6, 0.4], device=dev)
  effort = torch.tensor(
    [23.7, 23.7, 45.43] * 4 + [11.8125, 11.8125, 11.8125, 9.5844, 9.5844, 9.5844],
    device=dev,
  )
  joint_limits = torch.zeros((num_envs, WBC_NUM_JOINTS, 2), device=dev, dtype=torch.float32)
  joint_limits[..., 0] = -2.5
  joint_limits[..., 1] = 2.5
  soft_limits = joint_limits * 0.9
  arm_idx = torch.arange(WBC_NUM_LEG_JOINTS, WBC_NUM_JOINTS, device=dev, dtype=torch.long)
  clip = torch.zeros((num_envs, WBC_NUM_JOINTS, 2), device=dev, dtype=torch.float32)
  clip[..., 0] = -10.0
  clip[..., 1] = 10.0
  return WbcRobotConsts(
    default_q=default_q,
    action_scale=action_scale,
    kp=kp,
    kd=kd,
    effort=effort,
    soft_limits=soft_limits,
    joint_limits=joint_limits,
    arm_indices=arm_idx,
    default_arm=default_q[:, arm_idx],
    scale_arm=action_scale[:, arm_idx],
    arm_lower=joint_limits[:, arm_idx, 0],
    arm_upper=joint_limits[:, arm_idx, 1],
    max_arm_delta=0.06,
    action_clip=clip,
  )


def _make_standing_state(
  num_envs: int, device: str = "cpu", base_z: float = 0.30, foot_z: float = 0.01
) -> torch.Tensor:
  s = torch.zeros((num_envs, wbc_st.WBC_STATE_DIM), device=device, dtype=torch.float32)
  s[:, 0] = base_z
  s[:, 3] = -1.0
  feet_rel_z = foot_z - base_z
  s[:, wbc_st.FEET] = torch.tensor(
    [0.2, -0.15, feet_rel_z, 0.2, 0.15, feet_rel_z, -0.2, -0.15, feet_rel_z, -0.2, 0.15, feet_rel_z],
    device=device,
    dtype=torch.float32,
  )
  return s


class _MockEntity:
  def __init__(self, num_envs: int, device: str = "cpu") -> None:
    self.num_envs = num_envs
    self.device = device
    body_pos = torch.zeros(num_envs, 2, 3, device=device)
    body_pos[:, 0, 2] = 0.44
    body_pos[:, 1, 0] = 0.45
    body_pos[:, 1, 2] = 0.44
    body_quat = torch.zeros(num_envs, 2, 4, device=device)
    body_quat[:, :, 0] = 1.0
    self.data = SimpleNamespace(
      body_link_pos_w=body_pos,
      body_link_quat_w=body_quat,
    )

  def find_bodies(self, name: str) -> tuple[list[int], list[str]]:
    if name == "link0":
      return [0], ["link0"]
    if name == "end_effector":
      return [1], ["end_effector"]
    return [], []


def _mock_env(num_envs: int, device: str = "cpu") -> SimpleNamespace:
  return SimpleNamespace(
    num_envs=num_envs,
    device=device,
    step_dt=0.02,
    scene={"robot": _MockEntity(num_envs, device=device)},
  )


def test_leg_joint_pos_no_standing_amplification() -> None:
  consts = _synthetic_consts(2, "cpu")
  imag = WbcImagination(WbcImagineCfg(dt=0.02), consts, "cpu")

  pose0 = torch.zeros(2, 3)
  pose1 = torch.zeros(2, 3)
  s0 = _make_standing_state(2, "cpu")
  s1 = _make_standing_state(2, "cpu")
  s1[:, wbc_st.Q_LEG] = 0.15

  cmd_vel = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
  a = torch.zeros(2, WBC_NUM_JOINTS)
  a_prev = torch.zeros(2, WBC_NUM_JOINTS)
  q_ref_arm = torch.zeros(2, WBC_NUM_ARM_JOINTS)
  ee_cmd = torch.tensor([[0.5, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0]]).expand(2, 7)
  xi = torch.ones(2, WBC_CONTEXT_DIM)
  k = imag.rollout_constants(torch.arange(2))

  _, terms = imag.compute_rewards(s0, s1, pose0, pose1, a, a_prev, q_ref_arm, cmd_vel, ee_cmd, xi, k)
  expected = float((0.15**2) * WBC_NUM_LEG_JOINTS)
  assert terms["leg_joint_pos"][0].item() == pytest.approx(expected, rel=1e-5)
  assert terms["leg_joint_pos"][1].item() == pytest.approx(expected, rel=1e-5)
  assert terms["leg_joint_pos"][0].item() == pytest.approx(terms["leg_joint_pos"][1].item(), rel=1e-6)


def test_pitch_reward_defaults_and_contracts() -> None:
  w = WbcRewardWeights()
  assert w.flat_orientation_l2 == pytest.approx(-0.4)
  assert w.ee_pos_exp == pytest.approx(4.5)
  assert w.leg_joint_pos == pytest.approx(-0.1)
  assert w.standing_feet_air_soft == pytest.approx(-1.0)

  ee_cfg = WbcEePoseCommandCfg(stage=1)
  assert ee_cfg.limit_ranges is not None
  assert ee_cfg.limit_ranges.pos_z == pytest.approx((0.10, 0.95))
  assert ee_cfg.adaptive_sampling is True
  assert ee_cfg.adaptive_x_bins == 3
  assert ee_cfg.adaptive_z_bins == 5
  assert ee_cfg.adaptive_uniform_prob == pytest.approx(0.3)
  assert ee_cfg.max_xy_radius == pytest.approx(0.69)

  assert WBC_ACTOR_OBS_DIM == 210
  assert WBC_CRITIC_BASE_DIM == 114
  imag = WbcImagination(WbcImagineCfg(), _synthetic_consts(2), "cpu")
  assert imag.critic_dim == 114

  buf = WbcReplayBuffer(4, 16, WBC_ACTOR_OBS_DIM, "cpu")
  assert buf.obs.shape[-1] == 210
  assert buf.s.shape[-1] == wbc_st.WBC_STATE_DIM
  env_cfg = go2_piper_wbc_env_cfg(stage=1)
  assert env_cfg.commands["ee_pose"].limit_ranges.pos_z == pytest.approx((0.10, 0.95))


def test_ee_adaptive_uniform_prob_1_matches_uniform() -> None:
  for bad_kwargs in (
    {"adaptive_uniform_prob": -0.01},
    {"adaptive_uniform_prob": 1.01},
    {"adaptive_x_bins": 0},
    {"adaptive_z_bins": 0},
    {"adaptive_beta": -0.1},
    {"adaptive_err_clip": (0.0, 0.2)},
    {"adaptive_err_clip": (0.2, 0.1)},
    {"adaptive_ema_alpha": 0.0},
    {"adaptive_ema_alpha": 1.1},
    {"adaptive_min_hold_s": -0.1},
    {"adaptive_init_err": 0.0},
    {"max_xy_radius": 0.0},
  ):
    with pytest.raises(ValueError):
      WbcEePoseCommandCfg(**bad_kwargs)

  torch.manual_seed(0)
  cfg = WbcEePoseCommandCfg(stage=1, curriculum_progress=1.0, adaptive_uniform_prob=1.0)
  cmd = WbcEePoseCommand(cfg, _mock_env(256))
  cmd._bin_err[0] = 0.20
  cmd._bin_err[1:] = 0.02

  probs = cmd.bin_probabilities()
  assert torch.allclose(probs, torch.full((15,), 1.0 / 15.0), atol=1e-6)
  m = cmd.adaptive_metrics()
  assert m["prob_low_z"] == pytest.approx(0.20, abs=1e-6)
  assert m["prob_high_z"] == pytest.approx(0.20, abs=1e-6)

  all_bins = []
  env_ids = torch.arange(256)
  for _ in range(60):
    _, _, b = cmd._sample_goal_with_bins(env_ids)
    all_bins.append(b)
  cat = torch.cat(all_bins)
  counts = torch.bincount(cat, minlength=15).float()
  empirical = counts / counts.sum()
  assert (empirical - (1.0 / 15.0)).abs().max().item() < 0.015

  # Degenerate progress=0.0 (pos_z=(0.50, 0.50)) must not produce NaN or zero-division.
  cmd_zero = WbcEePoseCommand(
    WbcEePoseCommandCfg(stage=1, curriculum_progress=0.0, adaptive_uniform_prob=0.3),
    _mock_env(16),
  )
  pos0, quat0 = cmd_zero._sample_goal(torch.arange(16))
  assert torch.isfinite(pos0).all() and torch.isfinite(quat0).all()
  assert torch.allclose(pos0[:, 2], torch.full((16,), 0.50))


def test_ee_adaptive_sampling_weights_high_error_bin() -> None:
  torch.manual_seed(123)
  cfg = WbcEePoseCommandCfg(
    stage=1,
    curriculum_progress=1.0,
    adaptive_uniform_prob=0.3,
    adaptive_beta=1.25,
    adaptive_err_clip=(0.02, 0.20),
  )
  cmd = WbcEePoseCommand(cfg, _mock_env(512))
  # Deliberately exceed clip bounds to verify clipping at [0.02, 0.20]
  cmd._bin_err.fill_(0.005)
  cmd._bin_err[0] = 0.50

  e_min, e_max = 0.02, 0.20
  w_hi = e_max**1.25
  w_lo = e_min**1.25
  w_sum = w_hi + 14.0 * w_lo
  expected = torch.full((15,), 0.7 * (w_lo / w_sum) + 0.3 / 15.0)
  expected[0] = 0.7 * (w_hi / w_sum) + 0.3 / 15.0

  probs = cmd.bin_probabilities()
  assert torch.allclose(probs, expected, atol=1e-6)

  samples = []
  env_ids = torch.arange(512)
  for _ in range(40):
    _, _, b = cmd._sample_goal_with_bins(env_ids)
    samples.append(b)
  cat = torch.cat(samples)
  empirical = torch.bincount(cat, minlength=15).float() / float(cat.numel())
  assert (empirical - expected).abs().max().item() < 0.02


def test_ee_adaptive_excludes_explore_and_transient() -> None:
  torch.manual_seed(7)
  cfg = WbcEePoseCommandCfg(
    stage=1,
    curriculum_progress=1.0,
    adaptive_min_hold_s=2.0,
    adaptive_ema_alpha=0.05,
    adaptive_init_err=0.05,
    trajectory_mode_prob=0.0,
  )
  cmd = WbcEePoseCommand(cfg, _mock_env(6))
  # Env 0 is explore -> eval_mask=False
  cmd.set_eval_env_mask(torch.tensor([False, True, True, True, True, True]))

  # Case A: explore, short age, active traj, and reset-after-fall must NOT update bin 0
  cmd._bin_id[:] = 0
  cmd._cmd_age[:] = torch.tensor([5.0, 1.0, 5.0, 5.0, 1.5, 0.0])
  cmd._traj_state[2, 19] = 1.0
  cmd.metrics["position_error"][:] = 0.25

  # Simulate fall reset on env 3 first
  cmd.reset(torch.tensor([3]))
  cmd.compute(0.02)
  assert cmd._bin_err[0].item() == pytest.approx(0.05, abs=1e-6)
  assert cmd._bin_counts[0].item() == 0

  # Direct resample on envs 0, 1, 2 (explore, age<2.0, traj_active)
  cmd._bin_id[0:3] = 0
  cmd._cmd_age[0:3] = torch.tensor([5.0, 1.0, 5.0])
  cmd._traj_state[2, 19] = 1.0
  cmd.metrics["position_error"][0:3] = 0.25
  cmd._resample_command(torch.tensor([0, 1, 2]))
  assert cmd._bin_err[0].item() == pytest.approx(0.05, abs=1e-6)
  assert cmd._bin_counts[0].item() == 0

  # Case B: envs 4 and 5 are eval, non-traj, age >= 2.0 -> MUST update bin 2 via EMA
  cmd._traj_state[:, 19] = 0.0
  cmd._bin_id[4:6] = 2
  cmd._cmd_age[4:6] = torch.tensor([2.5, 4.0])
  cmd.metrics["position_error"][4:6] = torch.tensor([0.10, 0.20])  # mean = 0.15
  cmd._resample_command(torch.tensor([4, 5]))
  expected_err = 0.95 * 0.05 + 0.05 * 0.15
  assert cmd._bin_err[2].item() == pytest.approx(expected_err, abs=1e-6)
  assert cmd._bin_counts[2].item() == 1


def test_ee_max_xy_radius_clamp() -> None:
  torch.manual_seed(99)
  cfg = WbcEePoseCommandCfg(stage=1, curriculum_progress=1.0, max_xy_radius=0.69)
  cmd = WbcEePoseCommand(cfg, _mock_env(10_000))
  t0 = time.perf_counter()
  pos, _ = cmd._sample_goal(torch.arange(10_000))
  dt = time.perf_counter() - t0
  r_xy = torch.hypot(pos[:, 0], pos[:, 1])
  assert bool((r_xy <= 0.69 + 1e-6).all().item())
  assert pos[:, 0].min().item() >= 0.45 - 1e-6
  assert pos[:, 0].max().item() <= 0.69 + 1e-6
  assert pos[:, 2].min().item() >= 0.10 - 1e-6
  assert pos[:, 2].max().item() <= 0.95 + 1e-6
  assert dt < 2.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA GPU")
def test_ee_adaptive_checkpoint_roundtrip_and_backcompat(tmp_path: Path) -> None:
  def _make_cfg(
    run_name: str,
    *,
    resume: str = "",
    continue_run: bool = False,
    reset_critic: bool = False,
    init_cur: float = 0.0,
  ) -> WbcTrainCfg:
    return WbcTrainCfg(
      task="Go2Piper-WBC-Flat",
      num_envs=16,
      buffer_len=32,
      iterations=2,
      warmup_iterations=0,
      wm_window=4,
      wm_batch=16,
      wm_updates=1,
      imag_batch=8,
      horizon=4,
      wm_hidden=(64, 64),
      actor_hidden=(64, 32),
      critic_hidden=(64, 32),
      critic_epochs=1,
      critic_minibatches=2,
      wbc_stage=1,
      resume=resume,
      continue_run=continue_run,
      reset_critic=reset_critic,
      initial_curriculum_progress=init_cur,
      log_root=str(tmp_path),
      run_name=run_name,
    )

  t1 = WbcTrainer(_make_cfg("adapt_save"))
  try:
    ee_term1 = t1.env.command_manager.get_term("ee_pose")
    assert torch.equal(ee_term1._eval_env_mask, ~t1.explore)
    ee_term1._bin_err[0] = 0.19
    ee_term1._bin_counts[0] = 42
    ckpt_path = t1.save(5)
  finally:
    t1.close()

  raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
  assert "ee_adaptive_state" in raw and raw["ee_adaptive_state"] is not None

  # Roundtrip restore on continue_run=True
  t2 = WbcTrainer(_make_cfg("adapt_load", resume=str(ckpt_path), continue_run=True))
  try:
    ee_term2 = t2.env.command_manager.get_term("ee_pose")
    assert ee_term2._bin_err[0].item() == pytest.approx(0.19, abs=1e-6)
    assert int(ee_term2._bin_counts[0].item()) == 42
  finally:
    t2.close()

  # Backward compatibility with legacy checkpoints lacking ee_adaptive_state
  legacy_path = tmp_path / "legacy.pt"
  del raw["ee_adaptive_state"]
  torch.save(raw, legacy_path)
  t3 = WbcTrainer(_make_cfg("adapt_legacy", resume=str(legacy_path), continue_run=True))
  try:
    ee_term3 = t3.env.command_manager.get_term("ee_pose")
    assert ee_term3._bin_err[0].item() == pytest.approx(0.05, abs=1e-6)
  finally:
    t3.close()

  # Warm-start with --reset-critic and --initial-curriculum-progress 1.0
  t4 = WbcTrainer(
    _make_cfg(
      "adapt_warmstart",
      resume=str(ckpt_path),
      continue_run=False,
      reset_critic=True,
      init_cur=1.0,
    )
  )
  try:
    ee_term4 = t4.env.command_manager.get_term("ee_pose")
    vel_term4 = t4.env.command_manager.get_term("base_velocity")
    assert ee_term4.cfg.curriculum_progress == pytest.approx(1.0, abs=1e-6)
    assert vel_term4.cfg.curriculum_progress == pytest.approx(1.0, abs=1e-6)
    # Must NOT carry stale bin_err when reset_critic=True
    assert ee_term4._bin_err[0].item() == pytest.approx(0.05, abs=1e-6)
    assert math.isfinite(ee_term4.adaptive_metrics()["prob_low_z"])
  finally:
    t4.close()
