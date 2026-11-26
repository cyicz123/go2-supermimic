"""Phase 2C tests: differentiable WBC imagination, rewards, soft termination,
32-step BPTT gradients, Critic 114D/191D inputs, TorchScript export parity,
task registration, and MicroRough end-to-end pipeline verification.
"""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from go2_supermimic.supermimic.buffer import WbcReplayBuffer
from go2_supermimic.supermimic.train import Actor, Critic, TrainCfg
from go2_supermimic.supermimic.world_model import WorldModel
from go2_supermimic.target.terrain import SCAN_POINTS, HeightField
from go2_supermimic.wbc import (
  LINK0_BODY_BOX_MAX,
  LINK0_BODY_BOX_MIN,
  TRAJ_STATE_DIM,
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
) -> dict[str, torch.Tensor]:
  dev = torch.device(device)
  s0 = _make_standing_state(batch, dev, base_z=0.31, feet_z_rel_ground=0.015)
  s0[:, wbc_st.Q_ARM] = torch.tensor([0.0, 0.6, -0.7, 0.0, 0.3, 0.0], device=dev)
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
# 1. 32-step BPTT full-parameter finite non-zero gradient gate
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
  for p in wm.parameters():
    p.requires_grad_(False)

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

  seed = _make_rollout_seed(B, device, consts, stage2_traj=stage2_traj)
  objective, data = imag.rollout(wm, actor, critic, seed)
  assert objective.shape == (B,)
  assert data["rewards"].shape == (B, 32)
  assert data["surv"].shape == (B, 33)
  assert data["critic_in"].shape == (B, 33, imag.critic_dim)
  assert torch.isfinite(objective).all()
  assert torch.isfinite(data["rewards"]).all()
  assert torch.isfinite(data["surv"]).all()
  assert torch.isfinite(data["critic_in"]).all()

  loss = -objective.mean()
  actor.zero_grad(set_to_none=True)
  loss.backward()

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
# 2. Differentiable rewards, B1 wz gating, Box-SDF, soft termination, Critic
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

  # Match exact formula sigmoid((0.032 - z_foot) / 0.006)
  p_c, z_foot, v_xy_w, _ = imag.soft_contact_and_foot_vel(s0, pose0, s1, pose1, 0.02)
  assert torch.allclose(p_c, soft_contact_prob(z_foot, 0.032, 0.006), atol=1e-6)
  # Env 0 moved +0.1 m in X over 0.02 s -> v_foot_x_w = 5.0 m/s
  assert torch.allclose(v_xy_w[0, :, 0], torch.full((4,), 5.0), atol=1e-4)
  assert torch.allclose(v_xy_w[1], torch.zeros(4, 2), atol=1e-5)


def test_gait_wz_gating_standing_5x_and_reward_terms() -> None:
  consts = _synthetic_consts(3, "cpu")
  imag = WbcImagination(WbcImagineCfg(dt=0.02, lin_vel_kernel="smooth_l1"), consts, "cpu")

  pose0 = torch.zeros(3, 3)
  pose1 = torch.zeros(3, 3)
  s0 = _make_standing_state(3, "cpu", base_z=0.30, feet_z_rel_ground=0.01)
  s1 = _make_standing_state(3, "cpu", base_z=0.30, feet_z_rel_ground=0.01)
  # Lift foot 0 to z=0.04 m (below clearance target 0.06 m) and give it horizontal speed
  s1[:, wbc_st.FEET.start + 0] += 0.02
  s1[:, wbc_st.FEET.start + 2] += 0.03
  s1[:, wbc_st.Q_LEG] = 0.1
  s1[:, wbc_st.QD_ARM] = torch.tensor([[2.0] * 6, [3.05] * 6, [0.0] * 6])

  # Env 0: standing; Env 1: linear moving; Env 2: pure in-place yaw turning (wz != 0)
  cmd_vel = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 0.8]])
  a = torch.zeros(3, WBC_NUM_JOINTS)
  a_prev = torch.zeros(3, WBC_NUM_JOINTS)
  q_ref_arm = torch.full((3, WBC_NUM_ARM_JOINTS), 0.25)
  ee_cmd = torch.tensor([[0.5, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0]]).expand(3, 7)
  xi = torch.ones(3, WBC_CONTEXT_DIM)
  xi[:, 1:4] = 0.0
  k = imag.rollout_constants(torch.arange(3))

  _, terms = imag.compute_rewards(s0, s1, pose0, pose1, a, a_prev, q_ref_arm, cmd_vel, ee_cmd, xi, k)

  # B1 wz gating: clearance active for env 1 (vx) and env 2 (pure wz), 0 for env 0 (standing)
  assert terms["feet_clearance_soft"][0].item() == pytest.approx(0.0, abs=1e-7)
  assert terms["feet_clearance_soft"][1].item() > 0.0
  assert terms["feet_clearance_soft"][2].item() > 0.0

  # standing_feet_air_soft active only for standing env 0
  assert terms["standing_feet_air_soft"][0].item() > 0.0
  assert terms["standing_feet_air_soft"][1].item() == pytest.approx(0.0, abs=1e-7)
  assert terms["standing_feet_air_soft"][2].item() == pytest.approx(0.0, abs=1e-7)

  # leg_joint_pos 5x weight on standing env 0 vs moving envs 1, 2
  assert terms["leg_joint_pos"][0].item() / terms["leg_joint_pos"][1].item() == pytest.approx(
    5.0, rel=1e-5
  )

  # arm_vel_hinge (> 2.85 rad/s): 0 for 2.0 and 0.0; >0 for 3.05
  assert terms["arm_vel_hinge"][0].item() == pytest.approx(0.0, abs=1e-7)
  assert terms["arm_vel_hinge"][1].item() > 0.0
  assert terms["arm_vel_hinge"][2].item() == pytest.approx(0.0, abs=1e-7)

  # joint_torques_l2 / joint_power use q_ref_arm (0.25) instead of default_arm + scale*a_arm (0.0)
  expected_arm_tau = (k["kp"][12:] * 0.25).clamp(-k["effort"][12:], k["effort"][12:]).square().sum().item()
  assert terms["joint_torques_l2"][0].item() >= expected_arm_tau - 1e-3

  # Quaternion geodesic angle sign invariance and smoothness at 0 and pi
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
  consts = _synthetic_consts(2, "cpu")
  imag = WbcImagination(WbcImagineCfg(), consts, "cpu")

  # Outside box + radius -> exact 0; inside box -> > 0 and -grad pushes outward
  bmax = torch.tensor(LINK0_BODY_BOX_MAX)
  bmin = torch.tensor(LINK0_BODY_BOX_MIN)
  outside_pt = (bmax + torch.tensor([0.3, 0.3, 0.3])).view(1, 1, 3)
  inside_pt = (0.5 * (bmin + bmax) + torch.tensor([0.05, 0.0, 0.0])).view(1, 1, 3).requires_grad_(True)
  r = torch.tensor([0.05])
  assert box_sdf_penetration(outside_pt, r).item() == 0.0
  pen_in = box_sdf_penetration(inside_pt, r)
  assert pen_in.item() > 0.05
  (-pen_in).backward()
  assert inside_pt.grad is not None and torch.isfinite(inside_pt.grad).all()
  # Gradient of (-pen) should push toward nearest face (+z or +x) away from center
  assert inside_pt.grad[0, 0, 0].item() > 0.0

  # Test FK + termination_prob union bounds [0, 1] and 0 baseline at zero collision
  q_safe = torch.tensor([[0.0, 1.2, -0.5, 0.0, 0.0, 0.0], [0.0, 0.1, -0.1, 0.0, 0.0, 0.0]])
  fk = piper_fk(q_safe)
  s = _make_standing_state(2, "cpu", base_z=0.30)
  d, pens, (d_ori, d_base, d_col) = imag.termination_prob(s, fk)
  assert (d >= 0.0).all() and (d <= 1.0).all()
  assert (d_ori >= 0.0).all() and (d_base >= 0.0).all() and (d_col >= 0.0).all()
  # Safe pose outside LINK0_BODY_BOX has zero penetration & zero d_col baseline pollution
  if pens[0].max().item() == 0.0:
    assert d_col[0].item() == pytest.approx(0.0, abs=1e-7)

  # Critic input dimension 114D flat / 191D micro-rough
  k = imag.rollout_constants(torch.arange(2))
  crit = imag.critic_input(
    s,
    torch.zeros(2, 3),
    torch.zeros(2, 6),
    torch.zeros(2, 3),
    torch.zeros(2, 7),
    torch.ones(2, 9),
    k,
  )
  assert crit.shape == (2, 114)


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
# 4. Task registration & MicroRough scan_dim=77 GPU trainer smoke test
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
