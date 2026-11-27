"""Phase 2C tests: differentiable WBC imagination, rewards, soft termination,
32-step BPTT gradients, Critic 114D/191D inputs, TorchScript export parity,
task registration, curriculum clock resume, and MicroRough pipeline verification.
"""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import mujoco
import pytest
import torch
from torch import nn

from go2_supermimic.robot.go2_piper import DEFAULT_JOINT_POS, GO2_PIPER_XML
from go2_supermimic.supermimic.train import Actor, Critic, TrainCfg
from go2_supermimic.supermimic.world_model import WorldModel
from go2_supermimic.target.terrain import SCAN_POINTS, HeightField
from go2_supermimic.wbc import (
  LINK0_BODY_BOX_MAX,
  LINK0_BODY_BOX_MIN,
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
  WbcRobotConsts,
  box_sdf_penetration,
  export_wbc_actor_torchscript,
  pack_ee_traj_state,
  piper_fk,
  quat_geodesic_angle,
  soft_contact_prob,
  wbc_init_history,
  wbc_obs_frame,
)
from go2_supermimic.wbc import state as wbc_st
from go2_supermimic.wbc.trainer import WbcTrainCfg, WbcTrainer


def _synthetic_consts(num_envs: int, device: str | torch.device) -> WbcRobotConsts:
  dev = torch.device(device)
  default_q = torch.tensor(
    [
      -0.1, 0.8, -1.5,
      0.1, 0.8, -1.5,
      -0.1, 1.0, -1.5,
      0.1, 1.0, -1.5,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ],
    device=dev,
    dtype=torch.float32,
  ).unsqueeze(0).expand(num_envs, WBC_NUM_JOINTS).clone()
  action_scale = torch.full((num_envs, WBC_NUM_JOINTS), 0.25, device=dev, dtype=torch.float32)
  kp = torch.tensor(
    [40.0] * 12 + [50.0, 65.0, 55.0, 25.0, 20.0, 15.0],
    device=dev,
    dtype=torch.float32,
  )
  kd = torch.tensor(
    [1.0] * 12 + [2.0, 3.0, 2.0, 0.8, 0.6, 0.4],
    device=dev,
    dtype=torch.float32,
  )
  effort = torch.tensor(
    [23.7, 23.7, 45.43] * 4 + [11.8125, 11.8125, 11.8125, 9.5844, 9.5844, 9.5844],
    device=dev,
    dtype=torch.float32,
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
  feet_z_rel_ground: float = 0.01,
) -> torch.Tensor:
  s = torch.zeros((batch, WBC_STATE_DIM), device=device, dtype=torch.float32)
  s[:, 0] = base_z
  s[:, wbc_st.GRAV] = torch.tensor([0.0, 0.0, -1.0], device=device)
  feet_rel = torch.tensor(
    [
      [0.2, -0.15, feet_z_rel_ground - base_z],
      [0.2, 0.15, feet_z_rel_ground - base_z],
      [-0.2, -0.15, feet_z_rel_ground - base_z],
      [-0.2, 0.15, feet_z_rel_ground - base_z],
    ],
    device=device,
    dtype=torch.float32,
  )
  s[:, wbc_st.FEET] = feet_rel.reshape(1, 12).expand(batch, 12)
  return s


def _make_rollout_seed(
  batch: int,
  device: str | torch.device,
  consts: WbcRobotConsts,
  *,
  stage2_traj: bool = False,
  q_arm: tuple[float, ...] = (0.0, 0.6, -0.7, 0.0, 0.3, 0.0),
) -> dict[str, torch.Tensor]:
  dev = torch.device(device)
  s0 = _make_standing_state(batch, dev, base_z=0.31, feet_z_rel_ground=0.015)
  s0[:, wbc_st.Q_ARM] = torch.tensor(q_arm, device=dev, dtype=torch.float32)
  pose0 = torch.zeros((batch, 3), device=dev)
  a_prev0 = torch.zeros((batch, WBC_NUM_JOINTS), device=dev)
  q_ref_arm0 = s0[:, wbc_st.Q_ARM].clone()
  xi = torch.ones((batch, WBC_CONTEXT_DIM), device=dev)
  xi[:, 1:4] = 0.0
  cmd_vel = torch.tensor([[0.4, -0.1, 0.3]], device=dev).expand(batch, 3).clone()
  p0 = torch.tensor([[0.45, 0.0, 0.45]], device=dev).expand(batch, 3).clone()
  p1 = torch.tensor([[0.50, 0.12, 0.52]], device=dev).expand(batch, 3).clone()
  p2 = torch.tensor([[0.58, -0.10, 0.48]], device=dev).expand(batch, 3).clone()
  q0 = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=dev).expand(batch, 4).clone()
  q1 = torch.tensor([[0.9238795, 0.3826834, 0.0, 0.0]], device=dev).expand(batch, 4).clone()
  if stage2_traj:
    traj_state = pack_ee_traj_state(
      p0,
      p1,
      p2,
      q0,
      q1,
      torch.zeros(batch, device=dev),
      torch.full((batch,), 2.0, device=dev),
      torch.ones(batch, device=dev),
    )
    ee_cmd0 = torch.cat([p0, q0], dim=-1)
  else:
    traj_state = pack_ee_traj_state(
      p0, p0, p0, q0, q0,
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


# ---------------------------------------------------------------------------
# 1. 32-step BPTT full-parameter finite non-zero gradient gate + P1 #2 freeze
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
  ("stage2_traj", "scan_dim"),
  [(False, 0), (True, 0), (True, SCAN_POINTS)],
)
def test_32step_bptt_all_actor_params_finite_nonzero_grads(
  stage2_traj: bool, scan_dim: int
) -> None:
  torch.manual_seed(7)
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  B = 6
  consts = _synthetic_consts(B, device)

  field = None
  if scan_dim > 0:
    xs = torch.linspace(-2.0, 2.0, 81, device=device)
    ys = torch.linspace(-2.0, 2.0, 81, device=device)
    gx, gy = torch.meshgrid(xs, ys, indexing="xy")
    field = HeightField(0.015 * torch.sin(gx * 2.0) * torch.cos(gy * 2.0), (-2.0, -2.0), 0.05)

  wm = WorldModel(
    hidden=(128, 128),
    state_dim=WBC_STATE_DIM,
    act_dim=WBC_NUM_JOINTS,
    context_dim=WBC_CONTEXT_DIM,
    scan_dim=scan_dim,
    q_ref_arm_dim=WBC_Q_REF_ARM_DIM,
    pose_delta_dim=WBC_POSE_DELTA_DIM,
    z_slice=wbc_st.Z,
    state_slices=WBC_STATE_LAYOUT,
  ).to(device)

  imag = WbcImagination(
    WbcImagineCfg(horizon=32, obs_noise=False, sensor_grad=True),
    consts,
    device,
    scan_dim=scan_dim,
    field=field,
  )
  assert imag.critic_dim == WBC_CRITIC_BASE_DIM + scan_dim

  actor = Actor(WBC_ACTOR_OBS_DIM, (128, 64), WBC_NUM_JOINTS).to(device)
  critic = Critic(imag.critic_dim, (128, 64)).to(device)

  # P1 #2: Leave wm and critic requires_grad=True before rollout; rollout must freeze
  # both parameter sets (no .grad written) while preserving critic input gradients to actor.
  assert all(p.requires_grad for p in wm.parameters())
  assert all(p.requires_grad for p in critic.parameters())

  seed = _make_rollout_seed(B, device, consts, stage2_traj=stage2_traj)
  objective, data = imag.rollout(wm, actor, critic, seed)
  assert objective.shape == (B,)
  assert data["rewards"].shape == (B, 32)
  assert data["cont"].shape == (B, 32)
  assert data["surv"].shape == (B, 33)
  assert data["critic_in"].shape == (B, 33, imag.critic_dim)
  assert torch.isfinite(objective).all()
  assert torch.isfinite(data["rewards"]).all()
  assert torch.isfinite(data["cont"]).all()
  assert torch.isfinite(data["surv"]).all()
  assert torch.isfinite(data["critic_in"]).all()

  loss = -objective.mean()
  actor.zero_grad(set_to_none=True)
  critic.zero_grad(set_to_none=True)
  wm.zero_grad(set_to_none=True)
  loss.backward()

  for name, p in critic.named_parameters():
    assert p.grad is None, f"Critic param {name} should not receive grad during actor backward"
    assert p.requires_grad is True
  for name, p in wm.named_parameters():
    assert p.grad is None, f"WM param {name} should not receive grad during actor backward"
    assert p.requires_grad is True

  for name, p in actor.named_parameters():
    assert p.grad is not None, f"Missing grad for {name}"
    assert torch.isfinite(p.grad).all(), f"Non-finite grad for {name}"
    assert p.grad.abs().sum().item() > 0.0, f"Zero grad for {name}"

  out_w_grad = actor.net[-1].weight.grad
  out_b_grad = actor.net[-1].bias.grad
  assert out_w_grad is not None and out_b_grad is not None
  assert out_w_grad[:WBC_NUM_LEG_JOINTS].abs().sum().item() > 0.0
  assert out_w_grad[WBC_NUM_LEG_JOINTS:].abs().sum().item() > 0.0
  assert out_b_grad[:WBC_NUM_LEG_JOINTS].abs().sum().item() > 0.0
  assert out_b_grad[WBC_NUM_LEG_JOINTS:].abs().sum().item() > 0.0


# ---------------------------------------------------------------------------
# 2. Differentiable rewards, B1 wz gating, calibrated collision proxy, P1 #1/#3
# ---------------------------------------------------------------------------


def test_soft_contact_world_fd_slide_and_heightfield() -> None:
  xs = torch.linspace(-1.0, 2.0, 61)
  ys = torch.linspace(-1.0, 2.0, 61)
  gx, gy = torch.meshgrid(xs, ys, indexing="xy")
  field = HeightField(0.05 * gx + 0.02 * gy, origin=(-1.0, -1.0), res=0.05)

  consts = _synthetic_consts(2, "cpu")
  imag = WbcImagination(WbcImagineCfg(dt=0.02), consts, "cpu", scan_dim=SCAN_POINTS, field=field)

  pose0 = torch.tensor([[0.0, 0.0, 0.0], [0.2, 0.2, 0.5]])
  pose1 = torch.tensor([[0.1, 0.0, 0.0], [0.2, 0.2, 0.5]])
  s0 = _make_standing_state(2, "cpu", base_z=0.30, feet_z_rel_ground=0.032)
  s1 = _make_standing_state(2, "cpu", base_z=0.30, feet_z_rel_ground=0.032)

  p_c, z_foot, v_xy_w, _, p_c0 = imag.soft_contact_and_foot_vel(s0, pose0, s1, pose1, 0.02)
  assert torch.allclose(
    p_c0, soft_contact_prob(z_foot, imag.cfg.contact_height_m, imag.cfg.contact_tau_m), atol=1e-6
  )
  assert torch.allclose(
    p_c, soft_contact_prob(z_foot, imag.cfg.contact_height_m, imag.cfg.contact_tau_m), atol=1e-6
  )
  assert torch.allclose(v_xy_w[0, :, 0], torch.full((4,), 5.0), atol=1e-4)
  assert torch.allclose(v_xy_w[1], torch.zeros(4, 2), atol=1e-5)


def test_gait_wz_gating_standing_5x_and_reward_terms() -> None:
  consts = _synthetic_consts(3, "cpu")
  imag = WbcImagination(WbcImagineCfg(dt=0.02, lin_vel_kernel="smooth_l1"), consts, "cpu")

  pose0 = torch.zeros(3, 3)
  pose1 = torch.zeros(3, 3)
  s0 = _make_standing_state(3, "cpu", base_z=0.30, feet_z_rel_ground=0.01)
  s1 = _make_standing_state(3, "cpu", base_z=0.30, feet_z_rel_ground=0.01)
  s1[:, wbc_st.FEET.start + 0] += 0.02
  s1[:, wbc_st.FEET.start + 2] += 0.03
  s1[:, wbc_st.Q_LEG] = 0.1
  s1[:, wbc_st.QD_ARM] = torch.tensor([[2.0] * 6, [3.05] * 6, [0.0] * 6])

  cmd_vel = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 0.8]])
  a = torch.zeros(3, WBC_NUM_JOINTS)
  a_prev = torch.zeros(3, WBC_NUM_JOINTS)
  q_ref_arm = torch.full((3, WBC_NUM_ARM_JOINTS), 0.25)
  ee_cmd = torch.tensor([[0.5, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0]]).expand(3, 7)
  xi = torch.ones(3, WBC_CONTEXT_DIM)
  xi[:, 1:4] = 0.0
  k = imag.rollout_constants(torch.arange(3))

  _, terms = imag.compute_rewards(s0, s1, pose0, pose1, a, a_prev, q_ref_arm, cmd_vel, ee_cmd, xi, k)

  assert terms["feet_clearance_soft"][0].item() == pytest.approx(0.0, abs=1e-7)
  assert terms["feet_clearance_soft"][1].item() > 0.0
  assert terms["feet_clearance_soft"][2].item() > 0.0

  assert terms["standing_feet_air_soft"][0].item() > 0.0
  assert terms["standing_feet_air_soft"][1].item() == pytest.approx(0.0, abs=1e-7)
  assert terms["standing_feet_air_soft"][2].item() == pytest.approx(0.0, abs=1e-7)

  assert terms["leg_joint_pos"][0].item() / terms["leg_joint_pos"][1].item() == pytest.approx(
    5.0, rel=1e-5
  )

  assert terms["arm_vel_hinge"][0].item() == pytest.approx(0.0, abs=1e-7)
  assert terms["arm_vel_hinge"][1].item() > 0.0
  assert terms["arm_vel_hinge"][2].item() == pytest.approx(0.0, abs=1e-7)

  expected_arm_tau = (
    (k["kp"][12:] * 0.25).clamp(-k["effort"][12:], k["effort"][12:]).square().sum().item()
  )
  assert terms["joint_torques_l2"][0].item() >= expected_arm_tau - 1e-3

  q_id = torch.tensor([[1.0, 0.0, 0.0, 0.0]], requires_grad=True)
  q_anti = torch.tensor([[-1.0, 0.0, 0.0, 0.0]])
  q_pi = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
  ang_0 = quat_geodesic_angle(q_id, q_anti)
  ang_pi = quat_geodesic_angle(q_id, q_pi)
  assert ang_0.item() < 1e-3
  assert ang_pi.item() == pytest.approx(math.pi, abs=1e-3)
  (ang_0 + ang_pi).backward()
  assert torch.isfinite(q_id.grad).all()


def test_self_collision_box_sdf_and_soft_termination_and_critic_dim() -> None:
  consts = _synthetic_consts(4, "cpu")
  imag = WbcImagination(WbcImagineCfg(horizon=32, obs_noise=False), consts, "cpu")

  # Direct Box-SDF check: outside -> 0.0, inside -> > 0 and -grad pushes outward
  bmax = torch.tensor(LINK0_BODY_BOX_MAX)
  bmin = torch.tensor(LINK0_BODY_BOX_MIN)
  outside_pt = (bmax + torch.tensor([0.3, 0.3, 0.3])).view(1, 1, 3)
  inside_pt = (
    (0.5 * (bmin + bmax) + torch.tensor([0.05, 0.0, 0.0]))
    .view(1, 1, 3)
    .requires_grad_(True)
  )
  r = torch.tensor([0.05])
  assert box_sdf_penetration(outside_pt, r).item() == 0.0
  pen_in = box_sdf_penetration(inside_pt, r)
  assert pen_in.item() > 0.05
  (-pen_in).backward()
  assert inside_pt.grad is not None and torch.isfinite(inside_pt.grad).all()
  assert inside_pt.grad[0, 0, 0].item() > 0.0

  # P1 #1: Verify real MuJoCo safe poses (q_arm=0 & safe workspace) have ZERO false penetration
  # and real Phase 2A battery/base/D415 self-collision poses have > 0 penetration & push-out grad.
  mj_model = mujoco.MjSpec.from_file(str(GO2_PIPER_XML)).compile()
  mj_data = mujoco.MjData(mj_model)
  battery_hit = None
  base_hit = None
  for j2 in (1.4, 1.8, 2.2, 2.6):
    for j3 in (-0.2, -0.7, -1.2, -2.0, -2.7):
      for j5 in (-1.2, 0.0, 1.2):
        arm_pose = [0.0, j2, j3, 0.0, j5, 0.0]
        mujoco.mj_resetData(mj_model, mj_data)
        mj_data.qpos[2] = 0.35
        mj_data.qpos[3] = 1.0
        mj_data.qpos[7:25] = list(DEFAULT_JOINT_POS[:12]) + arm_pose
        mujoco.mj_forward(mj_model, mj_data)
        for c in range(mj_data.ncon):
          g1 = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, mj_data.contact[c].geom1)
          g2 = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, mj_data.contact[c].geom2)
          pair = {g1, g2}
          if "front_battery_collision_box" in pair and mj_data.contact[c].dist < -0.002 and battery_hit is None:
            battery_hit = arm_pose
          if (
            {"base_collision", "head_collision1", "d415_head_camera_collision_camera_main"} & pair
            and {"link4_collision", "link5_collision", "link6_collision", "end_effector_collision"} & pair
            and mj_data.contact[c].dist < -0.002
            and base_hit is None
          ):
            base_hit = arm_pose
  assert battery_hit is not None and base_hit is not None

  q_safe_0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  q_safe_work = [0.2, 1.57, -1.35, -0.3, -0.5, 0.4]
  q_link0_hit = [-1.742942, 3.136042, -0.012333, -0.542687, 0.251691, -0.897018]
  q_batch = torch.tensor(
    [q_safe_0, q_safe_work, battery_hit, q_link0_hit], dtype=torch.float32, requires_grad=True
  )
  fk = piper_fk(q_batch)
  s = _make_standing_state(4, "cpu", base_z=0.30)
  d, pens, (d_ori, d_base, d_col) = imag.termination_prob(s, fk)
  assert (d >= 0.0).all() and (d <= 1.0).all()
  assert (d_ori >= 0.0).all() and (d_base >= 0.0).all() and (d_col >= 0.0).all()

  # Unconditional assertions: q_arm = 0 and safe workspace have exact zero penetration and d_col < 1e-4
  assert pens[0].sum().item() == 0.0
  assert pens[1].sum().item() == 0.0
  assert d_col[0].item() < 1e-4
  assert d_col[1].item() < 1e-4

  # Real battery & base collision poses must register positive penetration and elevated d_col
  assert pens[2].sum().item() > 0.0
  assert d_col[2].item() > 0.01

  # Real link0_collision pose (link4/link5 hitting link0) must also register > 0, elevated d_col, and push-out grad
  assert pens[3].sum().item() > 0.02
  assert d_col[3].item() > 0.10
  q_l0_t = torch.tensor([q_link0_hit], dtype=torch.float32, requires_grad=True)
  l0_pen_before = float(imag.collision_penetration(piper_fk(q_l0_t)).sum().item())
  (-imag.collision_penetration(piper_fk(q_l0_t)).sum()).backward()
  assert q_l0_t.grad is not None and torch.isfinite(q_l0_t.grad).all()
  g_l0 = q_l0_t.grad[0].detach()
  assert g_l0.norm().item() > 0.0
  q_l0_nudged = torch.tensor([q_link0_hit], dtype=torch.float32) + 1e-3 * (g_l0 / g_l0.norm())
  l0_pen_after = float(imag.collision_penetration(piper_fk(q_l0_nudged)).sum().item())
  assert l0_pen_after < l0_pen_before

  # Also check base_hit explicitly
  q_base_t = torch.tensor([base_hit], dtype=torch.float32, requires_grad=True)
  pens_base = imag.collision_penetration(piper_fk(q_base_t))
  assert pens_base.sum().item() > 0.0

  # Gradient ascent on -penetration must reduce penetration when stepping q_arm
  pen_before = float(pens[2].sum().item())
  (-pens[2].sum()).backward()
  assert q_batch.grad is not None and torch.isfinite(q_batch.grad[2]).all()
  g = q_batch.grad[2].detach()
  assert g.norm().item() > 0.0
  q_nudged = torch.tensor([battery_hit], dtype=torch.float32) + 1e-3 * (g / g.norm())
  pen_after = float(imag.collision_penetration(piper_fk(q_nudged)).sum().item())
  assert pen_after < pen_before

  # 32-step default q_arm=0 rollout survival must be >= 0.99 (> 0.95)
  class _StandingIdentityWM(nn.Module):
    def forward(self, s, a, a_prev, xi, relief=None, q_ref_arm=None):
      del a, a_prev, xi, relief, q_ref_arm
      return s, torch.zeros(s.shape[0], 3, device=s.device, dtype=s.dtype)

  consts1 = _synthetic_consts(2, "cpu")
  seed0 = _make_rollout_seed(2, "cpu", consts1, stage2_traj=False, q_arm=(0.0,) * 6)
  seed0["s"][:, 0] = 0.30
  actor = Actor(WBC_ACTOR_OBS_DIM, (64, 32), WBC_NUM_JOINTS)
  critic = Critic(imag.critic_dim, (64, 32))
  _, rollout_data = imag.rollout(_StandingIdentityWM(), actor, critic, seed0)
  assert float(rollout_data["surv"][:, -1].min().item()) >= 0.99

  # Critic input dimension 114D flat
  k = imag.rollout_constants(torch.arange(4))
  crit = imag.critic_input(
    s,
    torch.zeros(4, 3),
    torch.zeros(4, 6),
    torch.zeros(4, 3),
    torch.zeros(4, 7),
    torch.ones(4, 9),
    k,
  )
  assert crit.shape == (4, 114)


def test_gae_targets_uses_exact_continuation_without_surv_clamp() -> None:
  """P1 #3: When surv_t drops to 1e-8 and subsequent d_t == 0, local continuation is 1.0."""
  consts = _synthetic_consts(1, "cpu")
  cfg = WbcImagineCfg(horizon=4, gamma=0.99, lam=0.95)
  imag = WbcImagination(cfg, consts, "cpu")

  class _ConstCritic(nn.Module):
    def forward(self, x):
      return torch.full((*x.shape[:-1], 1), 2.0, device=x.device, dtype=x.dtype)

  critic = _ConstCritic()
  rewards = torch.tensor([[1.0, 0.5, 0.25, 0.125]], dtype=torch.float32)
  # Step 0 kills survival down to 1e-8 (d_0 = 1 - 1e-8); steps 1..3 have d_t = 0 -> cont_t = 1.0
  cont = torch.tensor([[1e-8, 1.0, 1.0, 1.0]], dtype=torch.float32)
  surv = torch.tensor([[1.0, 1e-8, 1e-8, 1e-8, 1e-8]], dtype=torch.float32)
  data = {
    "rewards": rewards,
    "cont": cont,
    "surv": surv,
    "critic_in": torch.zeros((1, 5, imag.critic_dim), dtype=torch.float32),
  }
  targets = imag.gae_targets(critic, data)

  # Analytical TD(lambda) with exact per-step cont_t
  expected = torch.zeros_like(rewards)
  nxt = torch.tensor(2.0)
  for t in reversed(range(4)):
    nxt = rewards[0, t] + cfg.gamma * cont[0, t] * ((1.0 - cfg.lam) * 2.0 + cfg.lam * nxt)
    expected[0, t] = nxt
  assert torch.allclose(targets, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# 3. TorchScript export parity (< 1e-6) & CLI export test
# ---------------------------------------------------------------------------


def test_torchscript_export_parity_and_cli(tmp_path: Path) -> None:
  torch.manual_seed(123)
  actor = Actor(WBC_ACTOR_OBS_DIM, (512, 256, 128), WBC_NUM_JOINTS)
  with torch.no_grad():
    actor.norm.mean.uniform_(-2.0, 2.0)
    actor.norm.var.uniform_(0.2, 5.0)
  actor.eval()

  pt_path = tmp_path / "policy.pt"
  out = export_wbc_actor_torchscript(actor, pt_path, verify=True, atol=1e-6)
  assert out.exists()

  ckpt_path = tmp_path / "model_10.pt"
  torch.save({"actor": actor.state_dict(), "obs_dim": WBC_ACTOR_OBS_DIM}, ckpt_path)
  cli_pt = tmp_path / "policy_cli.pt"
  repo_root = Path(__file__).resolve().parents[1]
  res = subprocess.run(
    [
      sys.executable,
      str(repo_root / "scripts" / "export_wbc_policy.py"),
      "--checkpoint",
      str(ckpt_path),
      "--output",
      str(cli_pt),
    ],
    check=True,
    capture_output=True,
    text=True,
  )
  assert cli_pt.exists()
  assert "max_parity_err=" in res.stdout

  loaded = torch.jit.load(str(cli_pt), map_location="cpu").eval()
  for n in (1, 4):
    obs = torch.randn(n, WBC_ACTOR_OBS_DIM) * 2.5
    with torch.no_grad():
      ref = actor(obs)
      got = loaded(obs)
    assert got.shape == (n, WBC_NUM_JOINTS)
    err = float((got - ref).abs().max().item())
    assert err < 1e-6

  if torch.cuda.is_available():
    loaded_gpu = torch.jit.load(str(cli_pt), map_location="cuda:0").eval()
    obs_gpu = torch.randn(4, WBC_ACTOR_OBS_DIM, device="cuda:0")
    out_gpu = loaded_gpu(obs_gpu)
    assert out_gpu.shape == (4, WBC_NUM_JOINTS)
    assert torch.isfinite(out_gpu).all()


# ---------------------------------------------------------------------------
# 4. Task registration, P1 #4 resume curriculum clock, & MicroRough GPU smoke
# ---------------------------------------------------------------------------


def test_task_registration_and_defaults() -> None:
  import go2_supermimic.env  # noqa: F401
  from mjlab.tasks.registry import list_tasks, load_env_cfg

  tasks = set(list_tasks())
  assert "Go2Piper-WBC-Flat" in tasks
  assert "Go2Piper-WBC-MicroRough" in tasks
  assert "Go2-Chase-Flat" in tasks
  assert "Go2-Chase-Obstacles" in tasks
  assert "Go2-Chase-Terrain" in tasks

  flat_cfg = load_env_cfg("Go2Piper-WBC-Flat")
  rough_cfg = load_env_cfg("Go2Piper-WBC-MicroRough")
  assert flat_cfg.scene.terrain.terrain_type == "plane"
  assert rough_cfg.scene.terrain.terrain_type == "generator"

  wbc_defaults = WbcTrainCfg()
  assert wbc_defaults.driver_policy == ""
  assert wbc_defaults.warmup_iterations == 300
  assert wbc_defaults.explore_frac == 0.35
  assert wbc_defaults.explore_noise == 0.35
  chase_defaults = TrainCfg()
  assert chase_defaults.explore_frac == 0.25
  assert chase_defaults.explore_noise == 0.3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA GPU")
def test_resume_restores_common_step_counter_and_curriculum_before_first_reset(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """P1 #4: Resume must restore common_step_counter and command curricula BEFORE first reset()."""
  from mjlab.envs import ManagerBasedRlEnv

  cfg1 = WbcTrainCfg(
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
    target_curriculum_iterations=100,
    wbc_stage=1,
    log_interval=10,
    save_interval=2,
    log_root=str(tmp_path),
    run_name="p1_4_stage1",
  )
  t1 = WbcTrainer(cfg1)
  t1.run()
  ckpt_path = next(tmp_path.glob("*_p1_4_stage1/model_2.pt"))
  ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
  saved_step = int(ckpt["common_step_counter"])
  saved_vel_p = float(ckpt["vel_curriculum_progress"])
  saved_ee_p = float(ckpt["ee_curriculum_progress"])
  # wm_window=4 -> 6 warmup collects + 2 iteration collects = 8 env steps (> iter=2)
  assert saved_step == 8
  assert saved_vel_p == pytest.approx(8.0 / 100.0, rel=1e-5)
  assert saved_ee_p == pytest.approx(8.0 / 100.0, rel=1e-5)

  seen_at_first_reset: list[tuple[int, float, float, int]] = []
  orig_reset = ManagerBasedRlEnv.reset

  def _spy_reset(self, *args, **kwargs):
    if not seen_at_first_reset:
      vel_t = self.command_manager.get_term("base_velocity")
      ee_t = self.command_manager.get_term("ee_pose")
      seen_at_first_reset.append(
        (
          int(self.common_step_counter),
          float(vel_t.cfg.curriculum_progress),
          float(ee_t.cfg.curriculum_progress),
          int(ee_t.cfg.stage),
        )
      )
    return orig_reset(self, *args, **kwargs)

  monkeypatch.setattr(ManagerBasedRlEnv, "reset", _spy_reset)

  cfg2 = WbcTrainCfg(
    task="Go2Piper-WBC-Flat",
    num_envs=8,
    buffer_len=32,
    iterations=4,
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
    target_curriculum_iterations=100,
    wbc_stage=2,
    resume=str(ckpt_path),
    continue_run=True,
    log_root=str(tmp_path),
    run_name="p1_4_stage2",
  )
  t2 = WbcTrainer(cfg2)
  try:
    assert len(seen_at_first_reset) == 1
    pre_step, pre_vel_p, pre_ee_p, pre_stage = seen_at_first_reset[0]
    assert pre_step == saved_step
    assert pre_vel_p == pytest.approx(saved_vel_p, rel=1e-5)
    assert pre_ee_p == pytest.approx(saved_ee_p, rel=1e-5)
    assert pre_stage == 2

    assert int(t2.env.common_step_counter) == saved_step
    vel_t2 = t2.env.command_manager.get_term("base_velocity")
    ee_t2 = t2.env.command_manager.get_term("ee_pose")
    assert float(vel_t2.cfg.curriculum_progress) == pytest.approx(saved_vel_p, rel=1e-5)
    assert float(ee_t2.cfg.curriculum_progress) == pytest.approx(saved_ee_p, rel=1e-5)
    assert ee_t2.cfg.stage == 2
    assert ee_t2.cfg.trajectory_mode_prob == pytest.approx(0.40)
  finally:
    t2.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA GPU")
def test_consecutive_multi_resume_preserves_full_curriculum_curve(tmp_path: Path) -> None:
  """P1 #2: end_step 100 (step 40 -> 40%) -> extend to 200 (step 60 -> 47.5%) -> resume at 200 keeps 47.5%."""
  def _make_cfg(end_step: int, run_name: str, resume: str = "", continue_run: bool = False) -> WbcTrainCfg:
    return WbcTrainCfg(
      task="Go2Piper-WBC-Flat",
      num_envs=8,
      buffer_len=32,
      iterations=1,
      warmup_iterations=10,
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
      target_curriculum_iterations=end_step,
      wbc_stage=2 if resume else 1,
      resume=resume,
      continue_run=continue_run,
      log_root=str(tmp_path),
      run_name=run_name,
    )

  # Phase A: end_step=100, advance common_step_counter to 40 (40% progress) and save
  t1 = WbcTrainer(_make_cfg(100, "curve_p1"))
  try:
    t1.env.common_step_counter = 40
    ckpt1 = t1.save(10)
  finally:
    t1.close()
  d1 = torch.load(ckpt1, map_location="cpu", weights_only=False)
  assert float(d1["vel_curriculum_progress"]) == pytest.approx(0.40, abs=1e-6)
  assert float(d1["ee_curriculum_progress"]) == pytest.approx(0.40, abs=1e-6)

  # Phase B: first resume extending end_step 100 -> 200; step 40 stays 40%, step 60 reaches 47.5%
  t2 = WbcTrainer(_make_cfg(200, "curve_p2", resume=str(ckpt1), continue_run=True))
  try:
    assert int(t2.env.common_step_counter) == 40
    assert float(t2.env.command_manager.get_term("base_velocity").cfg.curriculum_progress) == pytest.approx(0.40, abs=1e-6)
    t2.env.common_step_counter = 60
    ckpt2 = t2.save(20)
  finally:
    t2.close()
  d2 = torch.load(ckpt2, map_location="cpu", weights_only=False)
  assert float(d2["vel_curriculum_progress"]) == pytest.approx(0.475, abs=1e-6)
  assert float(d2["ee_curriculum_progress"]) == pytest.approx(0.475, abs=1e-6)

  # Phase C: second resume keeping end_step=200; must stay strictly at 47.5% (NOT drop to 30%) and advance along same curve
  t3 = WbcTrainer(_make_cfg(200, "curve_p3", resume=str(ckpt2), continue_run=True))
  try:
    assert int(t3.env.common_step_counter) == 60
    assert float(t3.env.command_manager.get_term("base_velocity").cfg.curriculum_progress) == pytest.approx(0.475, abs=1e-6)
    assert float(t3.env.command_manager.get_term("ee_pose").cfg.curriculum_progress) == pytest.approx(0.475, abs=1e-6)
    t3.env.common_step_counter = 80
    t3.env.curriculum_manager.compute()
    assert float(t3.env.command_manager.get_term("base_velocity").cfg.curriculum_progress) == pytest.approx(0.55, abs=1e-6)
    assert float(t3.env.command_manager.get_term("ee_pose").cfg.curriculum_progress) == pytest.approx(0.55, abs=1e-6)
  finally:
    t3.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA GPU")
def test_microrough_trainer_init_collect_wm_and_rollout(tmp_path: Path) -> None:
  cfg = WbcTrainCfg(
    task="Go2Piper-WBC-MicroRough",
    num_envs=16,
    buffer_len=32,
    iterations=2,
    warmup_iterations=0,
    wm_window=4,
    wm_batch=16,
    wm_updates=1,
    imag_batch=16,
    horizon=8,
    wm_hidden=(128, 128),
    actor_hidden=(128, 64),
    critic_hidden=(128, 64),
    critic_epochs=1,
    critic_minibatches=2,
    wbc_stage=2,
    log_root=str(tmp_path),
    run_name="microrough_smoke",
  )
  trainer = WbcTrainer(cfg)
  try:
    assert trainer.scan_dim == SCAN_POINTS
    assert trainer.field is not None
    assert trainer.imag.critic_dim == WBC_CRITIC_BASE_DIM + SCAN_POINTS  # 191D
    assert trainer.wm.in_norm.mean.shape[0] == (
      WBC_STATE_DIM + 2 * WBC_NUM_JOINTS + WBC_Q_REF_ARM_DIM + WBC_CONTEXT_DIM + SCAN_POINTS
    )
    while trainer.buf.size <= cfg.wm_window + 1:
      trainer.collect()
    wm_loss = trainer.fit_world_model()
    pol_stats = trainer.improve_policy()
    assert math.isfinite(wm_loss)
    assert all(math.isfinite(v) for v in pol_stats.values())
    assert trainer.nonfinite_steps == 0
  finally:
    trainer.close()


def test_wbc_curriculum_defaults_3000() -> None:
  from go2_supermimic.supermimic.train import TrainCfg
  from go2_supermimic.wbc.env_cfg import go2_piper_wbc_env_cfg

  assert TrainCfg().target_curriculum_iterations == 3_000
  env_cfg = go2_piper_wbc_env_cfg()
  assert env_cfg.curriculum["base_velocity_levels"].params["end_step"] == 3_000
  assert env_cfg.curriculum["ee_pose_levels"].params["end_step"] == 3_000
