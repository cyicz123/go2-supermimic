"""Tests for Config M gait frequency control rewards, diagnostics, and invariants."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch import nn

from go2_supermimic.supermimic.train import Actor, Critic
from go2_supermimic.supermimic.world_model import WorldModel
from go2_supermimic.target import terrain as tr
from go2_supermimic.target.terrain import SCAN_POINTS, HeightField
from go2_supermimic.wbc import (
  WBC_ACTOR_OBS_DIM,
  WBC_CONTEXT_DIM,
  WBC_CRITIC_BASE_DIM,
  WBC_NUM_ARM_JOINTS,
  WBC_NUM_JOINTS,
  WBC_NUM_LEG_JOINTS,
  WBC_POSE_DELTA_DIM,
  WBC_Q_REF_ARM_DIM,
  WBC_STATE_DIM,
  WBC_STATE_LAYOUT,
  WbcImagineCfg,
  WbcImagination,
  WbcRewardWeights,
  WbcRobotConsts,
  export_wbc_actor_torchscript,
  pack_ee_traj_state,
  soft_contact_prob,
  wbc_foot_positions_w,
  wbc_init_history,
  wbc_obs_frame,
)
from go2_supermimic.wbc import state as wbc_st
from go2_supermimic.wbc.trainer import WbcTrainCfg, WbcTrainer


def _synthetic_consts(num_envs: int, device: str | torch.device) -> WbcRobotConsts:
  dev = torch.device(device)
  default_q = (
    torch.tensor(
      [
        -0.1, 0.8, -1.5,
        0.1, 0.8, -1.5,
        -0.1, 1.0, -1.5,
        0.1, 1.0, -1.5,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      ],
      device=dev,
      dtype=torch.float32,
    )
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
  batch: int,
  device: str | torch.device,
  *,
  base_z: float = 0.30,
  feet_z_rel_ground: float | torch.Tensor = 0.0205,
) -> torch.Tensor:
  dev = torch.device(device)
  s = torch.zeros((batch, WBC_STATE_DIM), device=dev, dtype=torch.float32)
  s[:, 0] = base_z
  s[:, wbc_st.GRAV] = torch.tensor([0.0, 0.0, -1.0], device=dev)
  feet_xy = torch.tensor(
    [[0.2, -0.15], [0.2, 0.15], [-0.2, -0.15], [-0.2, 0.15]],
    device=dev,
    dtype=torch.float32,
  ).unsqueeze(0).expand(batch, 4, 2)
  if torch.is_tensor(feet_z_rel_ground):
    fz = feet_z_rel_ground.to(device=dev, dtype=torch.float32)
  else:
    fz = torch.full((batch, 4), float(feet_z_rel_ground), device=dev, dtype=torch.float32)
  feet_rel = torch.cat([feet_xy, (fz - base_z).unsqueeze(-1)], dim=-1)
  s[:, wbc_st.FEET] = feet_rel.reshape(batch, 12)
  return s


def _make_rollout_seed(
  batch: int,
  device: str | torch.device,
  consts: WbcRobotConsts,
  *,
  cmd_vel: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
  dev = torch.device(device)
  s0 = _make_standing_state(batch, dev, base_z=0.31, feet_z_rel_ground=0.0205)
  s0[:, wbc_st.Q_ARM] = torch.tensor([0.0, 0.6, -0.7, 0.0, 0.3, 0.0], device=dev)
  pose0 = torch.zeros((batch, 3), device=dev)
  a_prev0 = torch.zeros((batch, WBC_NUM_JOINTS), device=dev)
  q_ref_arm0 = s0[:, wbc_st.Q_ARM].clone()
  xi = torch.ones((batch, WBC_CONTEXT_DIM), device=dev)
  xi[:, 1:4] = 0.0
  if cmd_vel is None:
    cmd_vel = torch.tensor([[0.4, -0.1, 0.3]], device=dev).expand(batch, 3).clone()
  p0 = torch.tensor([[0.45, 0.0, 0.45]], device=dev).expand(batch, 3).clone()
  q0 = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=dev).expand(batch, 4).clone()
  traj_state = pack_ee_traj_state(
    p0,
    p0,
    p0,
    q0,
    q0,
    torch.zeros(batch, device=dev),
    torch.ones(batch, device=dev),
    torch.zeros(batch, device=dev),
  )
  ee_cmd0 = torch.cat([p0, q0], dim=-1)
  frame0 = wbc_obs_frame(s0, a_prev0, cmd_vel, ee_cmd0, consts=consts, add_noise=False)
  obs0 = wbc_init_history(frame0)
  return {
    "env": torch.arange(batch, device=dev, dtype=torch.long),
    "s": s0,
    "pose": pose0,
    "a_prev": a_prev0,
    "q_ref_arm": q_ref_arm0,
    "obs": obs0,
    "cmd_vel": cmd_vel,
    "ee_pose_cmd": ee_cmd0,
    "ee_traj_state": traj_state,
    "xi": xi,
    "enc_bias": torch.zeros((batch, WBC_NUM_JOINTS), device=dev),
  }


def test_config_m_defaults() -> None:
  cfg = WbcImagineCfg()
  w = WbcRewardWeights()
  assert cfg.ang_vel_sigma == pytest.approx(0.35)
  assert cfg.weights == w

  assert w.contact_switch_tv == pytest.approx(-0.55)
  assert w.contact_switch_l2 == pytest.approx(0.0)
  assert w.leg_joint_acc_l2 == pytest.approx(-2.5e-7)
  assert w.action_rate_leg_l2 == pytest.approx(-0.05)
  assert w.action_rate_arm_l2 == pytest.approx(-0.02)
  assert w.joint_power == pytest.approx(-5e-4)
  assert w.joint_torques_l2 == pytest.approx(-2.0e-5)
  assert w.feet_clearance_soft == pytest.approx(-80.0)
  assert w.feet_slide_world_soft == pytest.approx(-1.0)
  assert w.diagonal_trot_soft == pytest.approx(-0.50)

  # Unchanged task, stability, and safety weights
  assert w.track_lin_vel_xy == pytest.approx(2.0)
  assert w.track_ang_vel_z == pytest.approx(1.0)
  assert w.ee_pos_exp == pytest.approx(3.0)
  assert w.ee_ori_huber == pytest.approx(-1.5)
  assert w.track_base_height_exp == pytest.approx(1.0)
  assert w.flat_orientation_l2 == pytest.approx(-2.0)
  assert w.lin_vel_z_l2 == pytest.approx(-1.0)
  assert w.ang_vel_xy_l2 == pytest.approx(-0.05)
  assert w.standing_feet_air_soft == pytest.approx(-1.0)
  assert w.leg_joint_pos == pytest.approx(-0.1)
  assert w.joint_mirror == pytest.approx(-0.05)
  assert w.dof_pos_limits == pytest.approx(-1.0)
  assert w.arm_vel_hinge == pytest.approx(-1.0)
  assert w.self_collision_penetration == pytest.approx(-20.0)
  assert w.terminated == pytest.approx(-200.0)


def test_p_contact0_matches_p_foot_w0_and_terrain_with_grads() -> None:
  xs = torch.linspace(-1.0, 2.0, 61)
  ys = torch.linspace(-1.0, 2.0, 61)
  gx, gy = torch.meshgrid(xs, ys, indexing="xy")
  field = HeightField(0.08 * gx - 0.04 * gy, origin=(-1.0, -1.0), res=0.05)

  consts = _synthetic_consts(2, "cpu")
  imag = WbcImagination(WbcImagineCfg(dt=0.02), consts, "cpu", scan_dim=SCAN_POINTS, field=field)

  pose0 = torch.tensor([[0.10, -0.20, 0.25], [0.45, 0.30, -0.40]], requires_grad=True)
  pose1 = torch.tensor([[0.25, -0.10, 0.30], [0.60, 0.25, -0.35]])
  z0_rel_base_ground = torch.tensor(
    [[0.018, 0.040, 0.026, 0.065], [0.055, 0.021, 0.032, 0.015]]
  )
  z1_rel_base_ground = torch.tensor(
    [[0.070, 0.020, 0.065, 0.022], [0.020, 0.075, 0.020, 0.080]]
  )
  s0 = _make_standing_state(2, "cpu", base_z=0.30, feet_z_rel_ground=z0_rel_base_ground).requires_grad_(True)
  s1 = _make_standing_state(2, "cpu", base_z=0.30, feet_z_rel_ground=z1_rel_base_ground)

  p_c, _, _, _, p_c0 = imag.soft_contact_and_foot_vel(s0, pose0, s1, pose1, 0.02)

  gz0 = tr.ground(pose0[..., :2], field)
  p_foot_w0 = wbc_foot_positions_w(s0, pose0, ground_z=gz0)
  foot_gz0 = tr.ground(p_foot_w0[..., :2], field)
  z_foot0 = p_foot_w0[..., 2] - foot_gz0
  expected_p_c0 = soft_contact_prob(z_foot0, imag.cfg.contact_height_m, imag.cfg.contact_tau_m)

  assert torch.allclose(p_c0, expected_p_c0, atol=1e-6)
  assert not torch.allclose(p_c0, p_c)

  p_c0.sum().backward()
  assert s0.grad is not None and torch.isfinite(s0.grad).all()
  assert s0.grad[:, wbc_st.FEET].abs().sum().item() > 0.0
  assert pose0.grad is not None and torch.isfinite(pose0.grad).all()
  assert pose0.grad.abs().sum().item() > 0.0


def test_moving_gate_regimes() -> None:
  B = 7
  consts = _synthetic_consts(B, "cpu")
  imag = WbcImagination(WbcImagineCfg(dt=0.02), consts, "cpu")

  pose0 = torch.zeros(B, 3)
  pose1 = torch.zeros(B, 3)
  s0 = _make_standing_state(B, "cpu", feet_z_rel_ground=0.0205)
  s1 = _make_standing_state(B, "cpu", feet_z_rel_ground=torch.tensor([[0.075, 0.0205, 0.0205, 0.075]] * B))

  cmd_vel = torch.tensor(
    [
      [0.0, 0.0, 0.0],    # 0: stationary -> 0.0
      [0.04, 0.0, 0.0],   # 1: below cmd_lin_thresh (0.05) -> 0.0
      [0.06, 0.0, 0.0],   # 2: just above cmd_lin_thresh -> 0.35 + 0.65 * (0.01 / 0.25) = 0.376
      [0.0, 0.0, 0.06],   # 3: just above cmd_ang_thresh (0.05), < 0.08 -> 0.35
      [0.0, 0.0, 0.20],   # 4: pure yaw turn -> 0.35 + 0.65 * ((0.20 - 0.08) / 0.42)
      [0.60, 0.0, 0.0],   # 5: high lin speed -> 1.0
      [0.0, 0.0, -0.60],  # 6: high yaw turn -> 1.0
    ]
  )
  expected_gate = torch.tensor(
    [
      0.0,
      0.0,
      0.35 + 0.65 * (0.01 / 0.25),
      0.35,
      0.35 + 0.65 * ((0.20 - 0.08) / 0.42),
      1.0,
      1.0,
    ]
  )
  assert expected_gate[2].item() >= 0.35 and expected_gate[2].item() < 1.0

  a = torch.zeros(B, WBC_NUM_JOINTS)
  a_prev = torch.zeros(B, WBC_NUM_JOINTS)
  q_ref_arm = torch.zeros(B, WBC_NUM_ARM_JOINTS)
  ee_cmd = torch.tensor([[0.45, 0.0, 0.45, 1.0, 0.0, 0.0, 0.0]]).expand(B, 7)
  xi = torch.ones(B, WBC_CONTEXT_DIM)
  xi[:, 1:4] = 0.0
  k = imag.rollout_constants(torch.arange(B))

  _, terms = imag.compute_rewards(s0, s1, pose0, pose1, a, a_prev, q_ref_arm, cmd_vel, ee_cmd, xi, k)

  p_c, _, _, _, p_c0 = imag.soft_contact_and_foot_vel(s0, pose0, s1, pose1, 0.02)
  dp = p_c - p_c0
  tv_raw = (torch.sqrt(dp.square() + 0.01**2) - 0.01).sum(dim=-1)
  l2_raw = dp.square().sum(dim=-1)

  assert torch.allclose(terms["contact_switch_tv"], tv_raw * expected_gate, atol=1e-6)
  assert torch.allclose(terms["contact_switch_l2"], l2_raw * expected_gate, atol=1e-6)
  assert terms["contact_switch_tv"][0].item() == 0.0
  assert terms["contact_switch_tv"][1].item() == 0.0
  assert terms["contact_switch_l2"][0].item() == 0.0
  assert terms["contact_switch_l2"][1].item() == 0.0


def test_contact_switch_monotonic_in_freq_stationary_zero_and_grads() -> None:
  consts = _synthetic_consts(2, "cpu")
  imag = WbcImagination(WbcImagineCfg(dt=0.02), consts, "cpu")
  k = imag.rollout_constants(torch.arange(2))
  pose0 = torch.zeros(2, 3)
  pose1 = torch.zeros(2, 3)
  a = torch.zeros(2, WBC_NUM_JOINTS)
  a_prev = torch.zeros(2, WBC_NUM_JOINTS)
  q_ref_arm = torch.zeros(2, WBC_NUM_ARM_JOINTS)
  ee_cmd = torch.tensor([[0.45, 0.0, 0.45, 1.0, 0.0, 0.0, 0.0]]).expand(2, 7)
  xi = torch.ones(2, WBC_CONTEXT_DIM)
  xi[:, 1:4] = 0.0
  # Row 0: stationary, Row 1: moving
  cmd_vel = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
  phases = torch.tensor([0.0, math.pi, math.pi, 0.0])

  mean_tv: dict[float, float] = {}
  mean_l2: dict[float, float] = {}
  for freq_hz in (2.0, 4.0, 6.0):
    tv_vals = []
    l2_vals = []
    for step in range(50):
      t0 = step * 0.02
      t1 = (step + 1) * 0.02
      z0 = 0.0205 + 0.06 * torch.relu(torch.sin(2.0 * math.pi * freq_hz * t0 + phases))
      z1 = 0.0205 + 0.06 * torch.relu(torch.sin(2.0 * math.pi * freq_hz * t1 + phases))
      s0 = _make_standing_state(2, "cpu", feet_z_rel_ground=z0.unsqueeze(0).expand(2, 4))
      s1 = _make_standing_state(2, "cpu", feet_z_rel_ground=z1.unsqueeze(0).expand(2, 4))
      _, terms = imag.compute_rewards(s0, s1, pose0, pose1, a, a_prev, q_ref_arm, cmd_vel, ee_cmd, xi, k)
      assert terms["contact_switch_tv"][0].item() == 0.0
      assert terms["contact_switch_l2"][0].item() == 0.0
      tv_vals.append(float(terms["contact_switch_tv"][1].item()))
      l2_vals.append(float(terms["contact_switch_l2"][1].item()))
    mean_tv[freq_hz] = sum(tv_vals) / len(tv_vals)
    mean_l2[freq_hz] = sum(l2_vals) / len(l2_vals)

  assert mean_tv[2.0] < mean_tv[4.0] < mean_tv[6.0]
  assert mean_l2[2.0] < mean_l2[4.0] < mean_l2[6.0]

  # Gradient check on s1 / p_foot_w1 and sign consistency with total reward
  s0_g = _make_standing_state(1, "cpu", feet_z_rel_ground=0.0205)
  s1_g = _make_standing_state(1, "cpu", feet_z_rel_ground=0.045).requires_grad_(True)
  k1 = imag.rollout_constants(torch.arange(1))
  r_total, terms_g = imag.compute_rewards(
    s0_g,
    s1_g,
    pose0[:1],
    pose1[:1],
    a[:1],
    a_prev[:1],
    q_ref_arm[:1],
    cmd_vel[1:2],
    ee_cmd[:1],
    xi[:1],
    k1,
  )
  terms_g["contact_switch_tv"].sum().backward()
  assert s1_g.grad is not None and torch.isfinite(s1_g.grad).all()
  assert s1_g.grad[0, wbc_st.FEET].abs().sum().item() > 0.0

  # Negative weight means positive TV penalty reduces total reward
  imag_zero_tv = WbcImagination(
    WbcImagineCfg(dt=0.02, weights=WbcRewardWeights(contact_switch_tv=0.0)),
    consts[:1] if False else _synthetic_consts(1, "cpu"),
    "cpu",
  )
  r_no_tv, _ = imag_zero_tv.compute_rewards(
    s0_g,
    s1_g.detach(),
    pose0[:1],
    pose1[:1],
    a[:1],
    a_prev[:1],
    q_ref_arm[:1],
    cmd_vel[1:2],
    ee_cmd[:1],
    xi[:1],
    k1,
  )
  assert r_total.item() < r_no_tv.item()


def test_leg_joint_acc_l2_exact_and_arm_invariant() -> None:
  consts = _synthetic_consts(2, "cpu")
  imag = WbcImagination(WbcImagineCfg(dt=0.02), consts, "cpu")
  k = imag.rollout_constants(torch.arange(2))
  pose0 = torch.zeros(2, 3)
  pose1 = torch.zeros(2, 3)
  s0 = _make_standing_state(2, "cpu", feet_z_rel_ground=0.0205)
  s1 = _make_standing_state(2, "cpu", feet_z_rel_ground=0.0205)

  qd_leg_0 = torch.linspace(-1.0, 1.2, 12)
  qd_leg_1 = torch.linspace(0.5, -0.8, 12)
  s0[:, wbc_st.QD_LEG] = qd_leg_0
  s1[:, wbc_st.QD_LEG] = qd_leg_1
  # Perturb arm joint velocities strongly on row 1 vs row 0
  s0[:, wbc_st.QD_ARM] = 0.0
  s1[0, wbc_st.QD_ARM] = 0.0
  s1[1, wbc_st.QD_ARM] = 12.0
  s1.requires_grad_(True)

  a = torch.zeros(2, WBC_NUM_JOINTS)
  a_prev = torch.zeros(2, WBC_NUM_JOINTS)
  q_ref_arm = torch.zeros(2, WBC_NUM_ARM_JOINTS)
  cmd_vel = torch.tensor([[0.4, 0.0, 0.0], [0.4, 0.0, 0.0]])
  ee_cmd = torch.tensor([[0.45, 0.0, 0.45, 1.0, 0.0, 0.0, 0.0]]).expand(2, 7)
  xi = torch.ones(2, WBC_CONTEXT_DIM)
  xi[:, 1:4] = 0.0

  _, terms = imag.compute_rewards(s0, s1, pose0, pose1, a, a_prev, q_ref_arm, cmd_vel, ee_cmd, xi, k)
  expected_val = ((qd_leg_1 - qd_leg_0) / 0.02).square().sum()
  assert terms["leg_joint_acc_l2"][0].item() == pytest.approx(expected_val.item(), rel=1e-5)
  assert terms["leg_joint_acc_l2"][1].item() == pytest.approx(expected_val.item(), rel=1e-5)

  terms["leg_joint_acc_l2"].sum().backward()
  assert s1.grad is not None and torch.isfinite(s1.grad).all()
  assert s1.grad[:, wbc_st.QD_LEG].abs().sum().item() > 0.0
  assert s1.grad[:, wbc_st.QD_ARM].abs().sum().item() == 0.0


def test_config_m_32step_bptt_and_imag_gait_diagnostics() -> None:
  torch.manual_seed(42)
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  B = 6
  consts = _synthetic_consts(B, device)
  wm = WorldModel(
    hidden=(128, 128),
    state_dim=WBC_STATE_DIM,
    act_dim=WBC_NUM_JOINTS,
    context_dim=WBC_CONTEXT_DIM,
    scan_dim=0,
    q_ref_arm_dim=WBC_Q_REF_ARM_DIM,
    pose_delta_dim=WBC_POSE_DELTA_DIM,
    z_slice=wbc_st.Z,
    state_slices=WBC_STATE_LAYOUT,
  ).to(device)
  imag = WbcImagination(WbcImagineCfg(horizon=32, obs_noise=False), consts, device)
  actor = Actor(WBC_ACTOR_OBS_DIM, (128, 64), WBC_NUM_JOINTS).to(device)
  critic = Critic(imag.critic_dim, (128, 64)).to(device)
  assert imag.critic_dim == WBC_CRITIC_BASE_DIM  # 114D unchanged

  seed = _make_rollout_seed(B, device, consts)
  objective, data = imag.rollout(wm, actor, critic, seed)
  assert torch.isfinite(objective).all()
  for gkey in (
    "imag_gait/f_implied_hz",
    "imag_gait/p_moving_mean",
    "imag_gait/p_mid_band_frac",
    "imag_gait/feet_slide_raw",
  ):
    val = data[gkey]
    assert val.ndim == 0
    assert not val.requires_grad
    assert torch.isfinite(val)

  loss = -objective.mean()
  actor.zero_grad(set_to_none=True)
  critic.zero_grad(set_to_none=True)
  wm.zero_grad(set_to_none=True)
  loss.backward()
  for name, p in actor.named_parameters():
    assert p.grad is not None, f"Missing grad for {name}"
    assert torch.isfinite(p.grad).all()
    assert p.grad.abs().sum().item() > 0.0
  for p in wm.parameters():
    assert p.grad is None
  for p in critic.parameters():
    assert p.grad is None

  # Controlled analytical check of 4 imag_gait/* scalars (mixed moving+standing vs all-standing)
  class _ScriptedTrotWM(nn.Module):
    def __init__(self, target_z_seq: list[torch.Tensor], dx: float):
      super().__init__()
      self.register_buffer("_seq", torch.stack(target_z_seq, dim=0))
      self.dx = dx
      self.step_idx = 0

    def forward(self, s, a, a_prev, xi, relief=None, q_ref_arm=None):
      del a, a_prev, xi, relief, q_ref_arm
      z_next = self._seq[self.step_idx % self._seq.shape[0]].to(device=s.device, dtype=s.dtype)
      self.step_idx += 1
      s_next = s.clone()
      feet = s_next[:, wbc_st.FEET].reshape(s.shape[0], 4, 3)
      feet[:, :, 2] = z_next.unsqueeze(0) - s_next[:, 0:1]
      s_next[:, wbc_st.FEET] = feet.reshape(s.shape[0], 12)
      dpose = torch.zeros(s.shape[0], 3, device=s.device, dtype=s.dtype)
      dpose[:, 0] = self.dx
      return s_next, dpose

  H = 20
  phases = torch.tensor([0.0, math.pi, math.pi, 0.0])
  z_seq = [
    0.0205 + 0.06 * torch.relu(torch.sin(2.0 * math.pi * 2.5 * (t + 1) * 0.02 + phases))
    for t in range(H)
  ]
  consts_s = _synthetic_consts(2, "cpu")
  imag_s = WbcImagination(WbcImagineCfg(horizon=H, obs_noise=False), consts_s, "cpu")
  actor_s = Actor(WBC_ACTOR_OBS_DIM, (64, 32), WBC_NUM_JOINTS)
  critic_s = Critic(imag_s.critic_dim, (64, 32))

  # Mixed batch: row 0 standing (cmd=0), row 1 moving (cmd=[0.5, 0, 0])
  cmd_mixed = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
  seed_mixed = _make_rollout_seed(2, "cpu", consts_s, cmd_vel=cmd_mixed)
  wm_scripted = _ScriptedTrotWM(z_seq, dx=0.01)
  _, data_mixed = imag_s.rollout(wm_scripted, actor_s, critic_s, seed_mixed)

  # Independently compute expected metrics over row 1 only
  z_prev = torch.full((4,), 0.0205)
  p_prev = soft_contact_prob(z_prev, 0.032, 0.006)
  tv_list, p_mean_list, p_mid_list, slide_list = [], [], [], []
  for z_t in z_seq:
    p_t = soft_contact_prob(z_t, 0.032, 0.006)
    dp = p_t - p_prev
    tv_list.append((torch.sqrt(dp.square() + 0.01**2) - 0.01).sum().item())
    p_mean_list.append(p_t.mean().item())
    p_mid_list.append(((p_t > 0.25) & (p_t < 0.75)).float().mean().item())
    slide_list.append((p_t * (0.01 / 0.02)).sum().item())
    p_prev = p_t

  exp_tv_mean = sum(tv_list) / len(tv_list)
  exp_hz = exp_tv_mean / (8.0 * 0.85 * 0.02)
  assert data_mixed["imag_gait/f_implied_hz"].item() == pytest.approx(exp_hz, rel=1e-4)
  assert data_mixed["imag_gait/p_moving_mean"].item() == pytest.approx(
    sum(p_mean_list) / len(p_mean_list), abs=1e-5
  )
  assert data_mixed["imag_gait/p_mid_band_frac"].item() == pytest.approx(
    sum(p_mid_list) / len(p_mid_list), abs=1e-5
  )
  assert data_mixed["imag_gait/feet_slide_raw"].item() == pytest.approx(
    sum(slide_list) / len(slide_list), rel=1e-4
  )

  # All-standing boundary: must be exact 0.0 without NaN/Inf
  cmd_stand = torch.zeros(2, 3)
  seed_stand = _make_rollout_seed(2, "cpu", consts_s, cmd_vel=cmd_stand)
  wm_scripted2 = _ScriptedTrotWM(z_seq, dx=0.01)
  _, data_stand = imag_s.rollout(wm_scripted2, actor_s, critic_s, seed_stand)
  for gkey in (
    "imag_gait/f_implied_hz",
    "imag_gait/p_moving_mean",
    "imag_gait/p_mid_band_frac",
    "imag_gait/feet_slide_raw",
  ):
    assert torch.isfinite(data_stand[gkey])
    assert data_stand[gkey].item() == 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA GPU")
def test_trainer_tb_gait_scalars_and_export_contract(
  tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
  cfg = WbcTrainCfg(
    task="Go2Piper-WBC-Flat",
    num_envs=8,
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
    log_interval=2,
    save_interval=100,
    log_root=str(tmp_path),
    run_name="gait_tb_smoke",
  )
  trainer = WbcTrainer(cfg)
  trainer.run()
  log_dir = trainer.log_dir
  actor = trainer.actor.cpu().eval()

  captured = capsys.readouterr().out
  assert "wm_loss " in captured
  assert "imagined_reward " in captured
  assert "f_hz=" in captured
  assert "p_mov=" in captured
  assert "p_mid=" in captured
  assert "slide_raw=" in captured

  ea = EventAccumulator(str(log_dir))
  ea.Reload()
  tags = set(ea.Tags()["scalars"])
  for expected_tag in (
    "imag_gait/f_implied_hz",
    "imag_gait/p_moving_mean",
    "imag_gait/p_mid_band_frac",
    "imag_gait/feet_slide_raw",
    "train/wm_loss",
    "train/imagined_reward",
  ):
    assert expected_tag in tags, f"Missing TB tag {expected_tag}; got {sorted(tags)}"
    assert f"train/{expected_tag}" not in tags
    events = ea.Scalars(expected_tag)
    assert [e.step for e in events] == [2]
    assert math.isfinite(events[0].value)

  # Confirm 210D -> 18D TorchScript export parity < 1e-6 unaffected
  ts_path = tmp_path / "actor_ts.pt"
  export_wbc_actor_torchscript(actor, ts_path, verify=True, atol=1e-6)
  loaded = torch.jit.load(str(ts_path), map_location="cpu").eval()
  obs = torch.randn(4, WBC_ACTOR_OBS_DIM)
  with torch.no_grad():
    err = float((loaded(obs) - actor(obs)).abs().max().item())
  assert err < 1e-6
