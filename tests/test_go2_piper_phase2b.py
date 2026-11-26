"""Phase 2B tests: 82D WBC state, analytic Piper FK, 210D obs, 9D DR context, WorldModel & Buffer."""

from __future__ import annotations

import math

import pytest
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.lab_api.math import quat_apply_inverse, quat_conjugate, quat_mul

from go2_supermimic.robot.go2_piper import (
  ARM_JOINT_NAMES,
  FOOT_GEOM_NAMES,
  LEG_JOINT_NAMES,
)
from go2_supermimic.supermimic import state as chase_st
from go2_supermimic.supermimic.buffer import (
  GEN_DIM,
  ReplayBuffer,
  WbcReplayBuffer,
  WbcWindowArchive,
  WindowArchive,
)
from go2_supermimic.supermimic.train import expand_world_model_scan
from go2_supermimic.supermimic.world_model import StateLayout, WorldModel
from go2_supermimic.target.terrain import SCAN_POINTS
from go2_supermimic.wbc import (
  TRAJ_STATE_DIM,
  WBC_ACTOR_OBS_DIM,
  WBC_CONTEXT_DIM,
  WBC_NUM_ARM_JOINTS,
  WBC_NUM_JOINTS,
  WBC_OBS_FRAME_DIM,
  WBC_STATE_DIM,
  WbcContextReader,
  ee_world_height,
  go2_piper_wbc_env_cfg,
  piper_fk,
  slew_step_arm,
  slew_step_arm_diff,
  wbc_apply_pose_delta,
  wbc_init_history,
  wbc_obs_frame,
  wbc_planar_pose,
  wbc_pose_delta,
  wbc_previous_action,
  wbc_push_history,
  wbc_robot_consts,
  wbc_robot_state,
)
from go2_supermimic.wbc import state as wbc_st

WBC_STATE_LAYOUT = StateLayout(
  state_dim=WBC_STATE_DIM,
  z=wbc_st.Z,
  grav=wbc_st.GRAV,
  v_yaw=wbc_st.V_YAW,
  w_b=wbc_st.W_B,
  q=wbc_st.Q,
  qd=wbc_st.QD,
  hips=wbc_st.HIPS,
  knees=wbc_st.KNEES,
  feet=wbc_st.FEET,
)


def _quat_dist_and_angle(q1: torch.Tensor, q2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
  """Sign-invariant WXYZ quaternion Euclidean distance and geodesic angle [rad]."""
  d_pos = torch.norm(q1 - q2, dim=-1)
  d_neg = torch.norm(q1 + q2, dim=-1)
  dist = torch.minimum(d_pos, d_neg)
  # Use half-chord asin instead of acos(|dot|) to avoid float32 precision loss near 0 rad.
  angle = 2.0 * torch.asin((0.5 * dist).clamp(0.0, 1.0))
  return dist, angle


# ---------------------------------------------------------------------------
# 1. 50-step real mjlab alignment (Stage 1 & Stage 2, including partial reset)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA GPU")
@pytest.mark.parametrize("stage", [1, 2])
def test_50step_mjlab_alignment_obs_fk_height_slew(stage: int) -> None:
  torch.manual_seed(42 + stage)
  cfg = go2_piper_wbc_env_cfg(stage=stage, micro_rough=False, play=True)
  cfg.scene.num_envs = 8
  cfg.observations["actor"].enable_corruption = False
  env = ManagerBasedRlEnv(cfg, device="cuda:0")
  try:
    consts = wbc_robot_consts(env)
    robot = env.scene["robot"]
    action_term = env.action_manager.get_term("joint_pos")
    vel_cmd_term = env.command_manager.get_term("base_velocity")
    pose_cmd_term = env.command_manager.get_term("ee_pose")

    link0_body_idx = robot.find_bodies("link0")[0][0]
    ee_body_idx = robot.find_bodies("end_effector")[0][0]
    link4_idx = robot.find_bodies("link4")[0][0]
    link5_idx = robot.find_bodies("link5")[0][0]
    link6_idx = robot.find_bodies("link6")[0][0]
    link0_site_idx = robot.find_sites("link0")[0][0]
    ee_site_idx = robot.find_sites("end_effector")[0][0]

    obs_dict, _ = env.reset(seed=100 + stage)
    s = wbc_robot_state(env)
    assert s.shape == (8, WBC_STATE_DIM)
    frame0 = wbc_obs_frame(
      s,
      wbc_previous_action(env),
      vel_cmd_term.command,
      pose_cmd_term.command,
      consts=consts,
      add_noise=False,
    )
    assert frame0.shape == (8, WBC_OBS_FRAME_DIM)
    obs_hist = wbc_init_history(frame0)
    assert obs_hist.shape == (8, WBC_ACTOR_OBS_DIM)

    max_obs_err = float((obs_hist - obs_dict["actor"]).abs().max().item())
    max_fk_pos_err = 0.0
    max_fk_site_err = 0.0
    max_fk_quat_err = 0.0
    max_fk_angle_err = 0.0
    max_link456_err = 0.0
    max_z_ee_err = 0.0
    max_slew_err = 0.0

    def _check_fk_and_height(s_cur: torch.Tensor) -> None:
      nonlocal max_fk_pos_err, max_fk_site_err, max_fk_quat_err
      nonlocal max_fk_angle_err, max_link456_err, max_z_ee_err
      q_arm = s_cur[:, wbc_st.Q_ARM] + consts.default_arm
      fk = piper_fk(q_arm)

      l0_pos = robot.data.body_link_pos_w[:, link0_body_idx]
      l0_quat = robot.data.body_link_quat_w[:, link0_body_idx]
      ee_pos = robot.data.body_link_pos_w[:, ee_body_idx]
      ee_quat = robot.data.body_link_quat_w[:, ee_body_idx]

      gt_pos_l0 = quat_apply_inverse(l0_quat, ee_pos - l0_pos)
      gt_quat_l0 = quat_mul(quat_conjugate(l0_quat), ee_quat)

      l0_spos = robot.data.site_pos_w[:, link0_site_idx]
      ee_spos = robot.data.site_pos_w[:, ee_site_idx]
      gt_spos_l0 = quat_apply_inverse(l0_quat, ee_spos - l0_spos)

      pos_err = float((fk.p_ee_link0 - gt_pos_l0).abs().max().item())
      site_err = float((fk.p_ee_link0 - gt_spos_l0).abs().max().item())
      q_dist, q_ang = _quat_dist_and_angle(fk.q_ee_link0_wxyz, gt_quat_l0)

      for name, b_idx in (("link4", link4_idx), ("link5", link5_idx), ("link6", link6_idx)):
        b_pos_l0 = quat_apply_inverse(l0_quat, robot.data.body_link_pos_w[:, b_idx] - l0_pos)
        max_link456_err = max(
          max_link456_err, float((fk.link_pos_link0[name] - b_pos_l0).abs().max().item())
        )

      z_ee = ee_world_height(s_cur[:, 0], s_cur[:, wbc_st.GRAV], fk.p_ee_base)
      z_err = float((z_ee - ee_pos[:, 2]).abs().max().item())

      max_fk_pos_err = max(max_fk_pos_err, pos_err)
      max_fk_site_err = max(max_fk_site_err, site_err)
      max_fk_quat_err = max(max_fk_quat_err, float(q_dist.max().item()))
      max_fk_angle_err = max(max_fk_angle_err, float(q_ang.max().item()))
      max_z_ee_err = max(max_z_ee_err, z_err)

    _check_fk_and_height(s)

    for step_idx in range(50):
      raw_act = (torch.rand(8, WBC_NUM_JOINTS, device="cuda:0") * 2.0 - 1.0) * 4.0
      q_ref_prev = action_term.q_ref_arm.clone()
      q_ref_pred = slew_step_arm_diff(
        q_ref_prev=q_ref_prev,
        raw_arm=raw_act[:, consts.arm_indices],
        default_arm=consts.default_arm,
        scale_arm=consts.scale_arm,
        arm_lower=consts.arm_lower,
        arm_upper=consts.arm_upper,
        max_delta=consts.max_arm_delta,
      )

      obs_dict, _, term, trunc, _ = env.step(raw_act)
      done = term | trunc

      keep = ~done
      if bool(keep.any()):
        slew_err = float((q_ref_pred[keep] - action_term.q_ref_arm[keep]).abs().max().item())
        max_slew_err = max(max_slew_err, slew_err)

      s = wbc_robot_state(env)
      frame = wbc_obs_frame(
        s,
        wbc_previous_action(env),
        vel_cmd_term.command,
        pose_cmd_term.command,
        consts=consts,
        add_noise=False,
      )
      obs_hist = wbc_push_history(obs_hist, frame)
      if bool(done.any()):
        obs_hist[done] = wbc_init_history(frame[done])

      max_obs_err = max(max_obs_err, float((obs_hist - obs_dict["actor"]).abs().max().item()))
      _check_fk_and_height(s)

      # Trigger an explicit partial reset mid-rollout (step 24) and verify immediate re-sync.
      if step_idx == 24:
        partial_ids = torch.tensor([1, 4, 6], device="cuda:0", dtype=torch.long)
        obs_reset_dict, _ = env.reset(env_ids=partial_ids)
        s = wbc_robot_state(env)
        frame_reset = wbc_obs_frame(
          s,
          wbc_previous_action(env),
          vel_cmd_term.command,
          pose_cmd_term.command,
          consts=consts,
          add_noise=False,
        )
        obs_hist[partial_ids] = wbc_init_history(frame_reset[partial_ids])
        max_obs_err = max(
          max_obs_err, float((obs_hist - obs_reset_dict["actor"]).abs().max().item())
        )
        _check_fk_and_height(s)

    print(
      f"[Stage {stage} 50-step alignment] "
      f"max_obs_err={max_obs_err:.3e}, "
      f"max_fk_pos_err={max_fk_pos_err:.3e}, "
      f"max_fk_site_err={max_fk_site_err:.3e}, "
      f"max_fk_quat_err={max_fk_quat_err:.3e}, "
      f"max_fk_angle_err={max_fk_angle_err:.3e}, "
      f"max_link456_err={max_link456_err:.3e}, "
      f"max_z_ee_err={max_z_ee_err:.3e}, "
      f"max_slew_err={max_slew_err:.3e}"
    )

    assert max_obs_err < 1e-5
    assert max_fk_pos_err < 1e-4
    assert max_fk_site_err < 1e-4
    assert max_fk_quat_err < 1e-4
    assert max_fk_angle_err < 1e-4
    assert max_link456_err < 1e-4
    assert max_z_ee_err < 1e-4
    assert max_slew_err < 1e-6
  finally:
    env.close()


# ---------------------------------------------------------------------------
# 2. Differentiability & finite-difference gradient checks
# ---------------------------------------------------------------------------


def test_piper_fk_shapes_and_gradcheck() -> None:
  # Leading shape support: 1D (6,), 2D (B, 6), 3D (T, B, 6)
  q1 = torch.zeros(6, dtype=torch.float32)
  q2 = torch.randn(5, 6, dtype=torch.float32)
  q3 = torch.randn(3, 4, 6, dtype=torch.float32)
  r1, r2, r3 = piper_fk(q1), piper_fk(q2), piper_fk(q3)
  assert r1.p_ee_link0.shape == (3,)
  assert r1.q_ee_link0_wxyz.shape == (4,)
  assert r1.p_ee_base.shape == (3,)
  assert r2.p_ee_link0.shape == (5, 3)
  assert r3.p_ee_link0.shape == (3, 4, 3)
  assert r3.q_ee_link0_wxyz.shape == (3, 4, 4)
  for k in ("link4", "link5", "link6", "end_effector"):
    assert r3.link_pos_link0[k].shape == (3, 4, 3)

  # Finite & gradcheck-verified gradients across 0, 90, 180 deg configs in float64
  q_special = torch.tensor(
    [
      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      [math.pi / 2, math.pi / 2, -math.pi / 2, math.pi / 2, 0.5, -math.pi / 2],
      [math.pi, math.pi, -math.pi, 0.0, 0.0, math.pi],
      [-0.7, 1.3, -1.9, 1.1, -0.8, 1.5],
    ],
    dtype=torch.float64,
    requires_grad=True,
  )

  def _fk_fn(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    res = piper_fk(q)
    return res.p_ee_link0, res.q_ee_link0_wxyz, res.p_ee_base

  assert torch.autograd.gradcheck(_fk_fn, (q_special,), eps=1e-6, atol=1e-4, rtol=1e-4)

  # Verify float32 gradients are finite & non-zero at special angles too
  q_f32 = q_special.detach().float().requires_grad_(True)
  res = piper_fk(q_f32)
  loss = res.p_ee_link0.square().sum() + res.q_ee_link0_wxyz.square().sum() + res.link_pos_link0["link6"].sum()
  loss.backward()
  assert q_f32.grad is not None
  assert torch.isfinite(q_f32.grad).all()
  assert q_f32.grad.abs().sum() > 0.0


def test_ee_world_height_differentiability_and_tilts() -> None:
  # Wide tilt angles including 0, 90, 180 deg pitch/roll
  angles = torch.tensor([0.0, math.pi / 4, math.pi / 2, math.pi], dtype=torch.float64)
  grav_b = torch.stack([-torch.sin(angles), torch.zeros_like(angles), -torch.cos(angles)], dim=-1)
  base_z = torch.full((4,), 0.35, dtype=torch.float64, requires_grad=True)
  grav_req = grav_b.clone().requires_grad_(True)
  q_arm = torch.tensor(
    [
      [0.0, 0.5, -0.5, 0.0, 0.2, 0.0],
      [0.4, 1.2, -1.5, 0.3, -0.4, 0.7],
      [-0.8, 2.0, -1.0, -0.5, 0.6, -1.1],
      [1.1, 0.8, -2.0, 0.9, -0.3, 0.4],
    ],
    dtype=torch.float64,
    requires_grad=True,
  )

  def _h_fn(bz: torch.Tensor, g: torch.Tensor, qa: torch.Tensor) -> torch.Tensor:
    return ee_world_height(bz, g, piper_fk(qa).p_ee_base)

  assert torch.autograd.gradcheck(_h_fn, (base_z, grav_req, q_arm), eps=1e-6, atol=1e-4, rtol=1e-4)

  # Also check (B, 1) base_z shape compatibility without unintended broadcasting
  z_col = base_z.detach().float().unsqueeze(-1).requires_grad_(True)
  h_col = ee_world_height(z_col, grav_b.float(), piper_fk(q_arm.detach().float()).p_ee_base)
  assert h_col.shape == (4,)
  h_col.sum().backward()
  assert z_col.grad is not None and torch.allclose(z_col.grad, torch.ones_like(z_col))


def test_slew_step_arm_diff_exact_and_saturated_gradients() -> None:
  lower = torch.tensor([-2.168, 0.0, -2.967, -1.745, -1.22, -2.0944])
  upper = torch.tensor([2.168, 3.14, 0.0, 1.745, 1.22, 2.0944])
  default_arm = torch.zeros(6)
  scale_arm = torch.full((6,), 0.25)
  max_delta = 0.06

  # Row 0: unsaturated; Row 1: slew-saturated (+); Row 2: slew-saturated (large raw=10);
  # Row 3: joint-limit saturated (at lower/upper bounds)
  q_ref_prev = torch.tensor(
    [
      [0.1, 0.5, -0.5, 0.0, 0.0, 0.0],
      [0.0, 0.5, -0.5, 0.0, 0.0, 0.0],
      [0.0, 0.5, -0.5, 0.0, 0.0, 0.0],
      [-2.15, 0.02, -0.02, 1.73, -1.20, 2.08],
    ],
    dtype=torch.float32,
    requires_grad=True,
  )
  raw_arm = torch.tensor(
    [
      [0.44, 2.1, -2.1, 0.1, -0.1, 0.12],  # small delta < 0.06
      [2.0, 4.0, -4.0, 2.0, -2.0, 2.0],    # slew clipped
      [10.0, 10.0, -10.0, 10.0, -10.0, 10.0],  # heavily saturated
      [-10.0, -1.0, 1.0, 10.0, -10.0, 10.0],   # hits joint limits
    ],
    dtype=torch.float32,
    requires_grad=True,
  )

  q_diff = slew_step_arm_diff(
    q_ref_prev, raw_arm, default_arm, scale_arm, lower, upper, max_delta
  )
  q_ref = slew_step_arm(
    q_ref_prev.detach(), raw_arm.detach(), default_arm, scale_arm, lower, upper, max_delta
  )
  assert float((q_diff - q_ref).abs().max().item()) < 1e-7

  q_diff.sum().backward()
  assert raw_arm.grad is not None and q_ref_prev.grad is not None
  assert torch.isfinite(raw_arm.grad).all()
  assert torch.isfinite(q_ref_prev.grad).all()
  # All rows (including slew-saturated and joint-limit-saturated) get positive finite grad w.r.t. raw_arm
  assert (raw_arm.grad > 0.0).all()
  # Slew-saturated rows (1, 2) get positive finite grad w.r.t. q_ref_prev
  assert (q_ref_prev.grad[1:3] > 0.0).all()


def test_wbc_obs_frame_and_push_history_gradients_and_noise() -> None:
  s = torch.randn(4, WBC_STATE_DIM, requires_grad=True)
  a_prev = torch.randn(4, WBC_NUM_JOINTS, requires_grad=True)
  cmd_vel = torch.randn(4, 3, requires_grad=True)
  ee_cmd = torch.randn(4, 7, requires_grad=True)

  f0 = wbc_obs_frame(s, a_prev, cmd_vel, ee_cmd, add_noise=False)
  hist = wbc_init_history(f0)
  f1 = wbc_obs_frame(s * 1.1, a_prev * 0.9, cmd_vel, ee_cmd, add_noise=True)
  hist1 = wbc_push_history(hist, f1)

  hist1.square().sum().backward()
  for tensor in (s, a_prev, cmd_vel, ee_cmd):
    assert tensor.grad is not None
    assert torch.isfinite(tensor.grad).all()
  assert s.grad[:, :1].abs().sum() == 0.0  # base_z not in actor obs
  assert s.grad[:, wbc_st.GRAV].abs().sum() > 0.0
  assert s.grad[:, wbc_st.W_B].abs().sum() > 0.0
  assert s.grad[:, wbc_st.Q].abs().sum() > 0.0
  assert s.grad[:, wbc_st.QD].abs().sum() > 0.0
  assert a_prev.grad.abs().sum() > 0.0
  assert cmd_vel.grad.abs().sum() > 0.0
  assert ee_cmd.grad.abs().sum() > 0.0

  # Verify noise is applied BEFORE scale (ang_vel: ±0.2*0.2=±0.04, joint_vel: ±1.5*0.05=±0.075)
  torch.manual_seed(0)
  zero_s = torch.zeros(2000, WBC_STATE_DIM)
  zero_a = torch.zeros(2000, WBC_NUM_JOINTS)
  noisy = wbc_obs_frame(
    zero_s, zero_a, torch.zeros(2000, 3), torch.zeros(2000, 7), add_noise=True
  )
  assert noisy[:, 0:3].abs().max().item() <= 0.2 * 0.2 + 1e-6
  assert noisy[:, 0:3].abs().max().item() > 0.03
  assert noisy[:, 3:6].abs().max().item() <= 0.05 + 1e-6
  assert noisy[:, 6:24].abs().max().item() <= 0.01 + 1e-6
  assert noisy[:, 24:42].abs().max().item() <= 1.5 * 0.05 + 1e-6
  assert noisy[:, 24:42].abs().max().item() > 0.06


# ---------------------------------------------------------------------------
# 3. 9D WbcContextReader real train-env verification
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA GPU")
def test_wbc_context_reader_9d_real_train_env() -> None:
  cfg = go2_piper_wbc_env_cfg(stage=2, micro_rough=False, play=False)
  assert "push_robot" not in cfg.events
  cfg.scene.num_envs = 16
  env = ManagerBasedRlEnv(cfg, device="cuda:0")
  try:
    env.reset(seed=2026)
    reader = WbcContextReader(env)
    xi = reader.read()
    assert xi.shape == (16, WBC_CONTEXT_DIM)
    assert torch.isfinite(xi).all()

    robot = env.scene["robot"]
    local_geoms, _ = robot.find_geoms(FOOT_GEOM_NAMES, preserve_order=True)
    foot_geoms = robot.indexing.geom_ids[local_geoms].to(dtype=torch.long)
    base_id = int(robot.indexing.body_ids[robot.find_bodies("base")[0][0]])

    gt_mu = env.sim.model.geom_friction[:, foot_geoms, 0].mean(dim=-1)
    gt_com = env.sim.model.body_ipos[:, base_id] - torch.as_tensor(
      env.sim.mj_model.body_ipos[base_id], device="cuda:0", dtype=torch.float32
    )
    gt_mass = env.sim.model.body_mass[:, base_id] / float(env.sim.mj_model.body_mass[base_id])

    leg_set, arm_set = set(LEG_JOINT_NAMES), set(ARM_JOINT_NAMES)
    leg_kp_list, leg_kd_list, arm_kp_list, arm_kd_list = [], [], [], []
    for act in robot.actuators:
      l_idx = [i for i, n in enumerate(act.target_names) if n in leg_set]
      a_idx = [i for i, n in enumerate(act.target_names) if n in arm_set]
      if l_idx:
        leg_kp_list.append(act.stiffness[:, l_idx] / act.default_stiffness[:, l_idx])
        leg_kd_list.append(act.damping[:, l_idx] / act.default_damping[:, l_idx])
      if a_idx:
        arm_kp_list.append(act.stiffness[:, a_idx] / act.default_stiffness[:, a_idx])
        arm_kd_list.append(act.damping[:, a_idx] / act.default_damping[:, a_idx])

    gt_leg_kp = torch.cat(leg_kp_list, dim=-1).mean(dim=-1)
    gt_leg_kd = torch.cat(leg_kd_list, dim=-1).mean(dim=-1)
    gt_arm_kp = torch.cat(arm_kp_list, dim=-1).mean(dim=-1)
    gt_arm_kd = torch.cat(arm_kd_list, dim=-1).mean(dim=-1)

    torch.testing.assert_close(xi[:, 0], gt_mu, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(xi[:, 1:4], gt_com, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(xi[:, 4], gt_mass, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(xi[:, 5], gt_leg_kp, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(xi[:, 6], gt_leg_kd, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(xi[:, 7], gt_arm_kp, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(xi[:, 8], gt_arm_kd, atol=1e-5, rtol=1e-5)

    # Confirm all 9 DR dims actually vary across envs
    assert xi[:, 0].std() > 1e-2
    assert (xi[:, 1:4].std(dim=0) > 1e-3).all()
    assert xi[:, 4].std() > 1e-3
    assert (xi[:, 5:9].std(dim=0) > 1e-2).all()
  finally:
    env.close()


# ---------------------------------------------------------------------------
# 4. WorldModel, ReplayBuffer, WindowArchive, and Go2-Chase compatibility
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scan_dim", [0, SCAN_POINTS])
def test_wbc_world_model_forward_loss_and_grad(scan_dim: int) -> None:
  torch.manual_seed(0)
  wm = WorldModel(
    hidden=(64, 64),
    state_dim=WBC_STATE_DIM,
    act_dim=WBC_NUM_JOINTS,
    context_dim=WBC_CONTEXT_DIM,
    scan_dim=scan_dim,
    q_ref_arm_dim=WBC_NUM_ARM_JOINTS,
    state_slices=WBC_STATE_LAYOUT,
    loss_weights={"z": 2.0, "grav": 2.0, "feet": 1.5, "pose": 0.5},
  )
  B, k = 8, 4
  s0 = torch.randn(B, WBC_STATE_DIM, requires_grad=True)
  actions = torch.randn(B, k, WBC_NUM_JOINTS, requires_grad=True)
  actions_prev = torch.randn(B, k, WBC_NUM_JOINTS)
  q_ref_arms = torch.randn(B, k, WBC_NUM_ARM_JOINTS, requires_grad=True)
  xi = torch.randn(B, WBC_CONTEXT_DIM)
  reliefs = torch.randn(B, k, scan_dim) if scan_dim > 0 else None
  s_targets = torch.randn(B, k, WBC_STATE_DIM)
  pose_targets = torch.randn(B, k, 3)
  valid = torch.ones(B, k)
  valid[0, 2:] = 0.0

  wm.update_stats(
    s0.detach(),
    actions[:, 0].detach(),
    actions_prev[:, 0],
    xi,
    None if reliefs is None else reliefs[:, 0],
    s_targets[:, 0],
    pose_targets[:, 0],
    q_ref_arm=q_ref_arms[:, 0].detach(),
  )
  s_pred, dpose = wm(
    s0,
    actions[:, 0],
    actions_prev[:, 0],
    xi,
    None if reliefs is None else reliefs[:, 0],
    q_ref_arm=q_ref_arms[:, 0],
  )
  assert s_pred.shape == (B, WBC_STATE_DIM)
  assert dpose.shape == (B, 3)
  if scan_dim == 0:
    s_pred_empty, _ = wm(
      s0,
      actions[:, 0],
      actions_prev[:, 0],
      xi,
      torch.empty(B, 0),
      q_ref_arm=q_ref_arms[:, 0],
    )
    torch.testing.assert_close(s_pred_empty, s_pred)

  loss = wm.loss(
    s0, actions, actions_prev, xi, reliefs, s_targets, pose_targets, valid, q_ref_arms=q_ref_arms
  )
  loss.backward()
  assert torch.isfinite(loss)
  assert s0.grad is not None and torch.isfinite(s0.grad).all() and s0.grad.abs().sum() > 0
  assert actions.grad is not None and torch.isfinite(actions.grad).all() and actions.grad.abs().sum() > 0
  assert (
    q_ref_arms.grad is not None
    and torch.isfinite(q_ref_arms.grad).all()
    and q_ref_arms.grad.abs().sum() > 0
  )
  for p in wm.parameters():
    assert p.grad is not None and torch.isfinite(p.grad).all()


def test_expand_world_model_scan_works_for_82d_wbc() -> None:
  torch.manual_seed(1)
  wm_flat = WorldModel(
    hidden=(64, 64),
    state_dim=WBC_STATE_DIM,
    act_dim=WBC_NUM_JOINTS,
    context_dim=WBC_CONTEXT_DIM,
    scan_dim=0,
    q_ref_arm_dim=WBC_NUM_ARM_JOINTS,
  )
  wm_rough = WorldModel(
    hidden=(64, 64),
    state_dim=WBC_STATE_DIM,
    act_dim=WBC_NUM_JOINTS,
    context_dim=WBC_CONTEXT_DIM,
    scan_dim=SCAN_POINTS,
    q_ref_arm_dim=WBC_NUM_ARM_JOINTS,
  )
  assert expand_world_model_scan(wm_flat.state_dict(), wm_rough)

  B = 12
  s = torch.randn(B, WBC_STATE_DIM)
  a = torch.randn(B, WBC_NUM_JOINTS)
  ap = torch.randn(B, WBC_NUM_JOINTS)
  qr = torch.randn(B, WBC_NUM_ARM_JOINTS)
  xi = torch.randn(B, WBC_CONTEXT_DIM)
  with torch.no_grad():
    s1_f, p1_f = wm_flat(s, a, ap, xi, None, q_ref_arm=qr)
    for relief in (torch.zeros(B, SCAN_POINTS), torch.randn(B, SCAN_POINTS)):
      s1_r, p1_r = wm_rough(s, a, ap, xi, relief, q_ref_arm=qr)
      torch.testing.assert_close(s1_r, s1_f, atol=1e-5, rtol=1e-5)
      torch.testing.assert_close(p1_r, p1_f, atol=1e-5, rtol=1e-5)


def test_wbc_replay_buffer_and_window_archive() -> None:
  N, L, H = 3, 12, 4
  buf = WbcReplayBuffer(num_envs=N, length=L, obs_dim=WBC_ACTOR_OBS_DIM, device="cpu", scan_dim=0)
  assert buf.relief.numel() == 0

  # Write 15 steps (> L=12 to test ring wraparound) with done=1 at logical step 5 -> slot 5
  for step in range(15):
    val = float(step + 1)
    s = torch.full((N, WBC_STATE_DIM), val)
    pose = torch.full((N, 3), val)
    a = torch.full((N, WBC_NUM_JOINTS), val)
    a_prev = torch.full((N, WBC_NUM_JOINTS), val - 1.0)
    q_ref = torch.full((N, WBC_NUM_ARM_JOINTS), val * 0.1)
    xi = torch.full((N, WBC_CONTEXT_DIM), val * 0.01)
    obs = torch.full((N, WBC_ACTOR_OBS_DIM), val)
    cmd_vel = torch.full((N, 3), val * 0.2)
    ee_cmd = torch.full((N, 7), val * 0.3)
    traj = torch.full((N, TRAJ_STATE_DIM), val * 0.4)
    done = torch.tensor([1.0 if step in (6, 10) else 0.0] * N)
    buf.add_wbc(s, pose, a, a_prev, q_ref, xi, obs, cmd_vel, ee_cmd, traj, done)

  w = buf.sample_windows(batch_size=32, horizon=H)
  assert w["s"].shape == (32, H + 1, WBC_STATE_DIM)
  assert w["pose"].shape == (32, H + 1, 3)
  assert w["a"].shape == (32, H, WBC_NUM_JOINTS)
  assert w["a_prev"].shape == (32, H, WBC_NUM_JOINTS)
  assert w["q_ref_arm"].shape == (32, H, WBC_NUM_ARM_JOINTS)
  assert w["xi"].shape == (32, WBC_CONTEXT_DIM)
  assert w["cmd_vel"].shape == (32, H + 1, 3)
  assert w["ee_pose_cmd"].shape == (32, H + 1, 7)
  assert w["ee_traj_state"].shape == (32, H + 1, TRAJ_STATE_DIM)
  assert w["valid"].shape == (32, H)
  assert w["relief"].shape == (32, H, 0)

  # Verify exact value restoration and done-boundary masking
  for b in range(32):
    step_vals = w["s"][b, :, 0]
    assert torch.allclose(step_vals[1:] - step_vals[:-1], torch.ones(H))
    dones = buf.done[w["env"][b], w["idx"][b, :H]]
    expected_valid = torch.cumprod(1.0 - dones, dim=0)
    torch.testing.assert_close(w["valid"][b], expected_valid)

  seed = buf.sample_seed(16)
  assert seed["s"].shape == (16, WBC_STATE_DIM)
  assert seed["ee_traj_state"].shape == (16, TRAJ_STATE_DIM)

  archive = WbcWindowArchive(capacity=64, k=H, device="cpu", scan_dim=0)
  archive.add(w)
  sampled = archive.sample(10)
  assert sampled["s"].shape == (10, H + 1, WBC_STATE_DIM)
  assert sampled["q_ref_arm"].shape == (10, H, WBC_NUM_ARM_JOINTS)
  assert sampled["xi"].shape == (10, WBC_CONTEXT_DIM)


def test_chase_70d_backward_compatibility() -> None:
  torch.manual_seed(7)
  wm = WorldModel(hidden=(32, 32))
  B, k = 4, 3
  s = torch.randn(B, chase_st.STATE_DIM)
  a = torch.randn(B, chase_st.NUM_JOINTS)
  ap = torch.randn(B, chase_st.NUM_JOINTS)
  xi = torch.randn(B, chase_st.CONTEXT_DIM)
  rel = torch.randn(B, k, SCAN_POINTS)
  s_tgt = torch.randn(B, k, chase_st.STATE_DIM)
  p_tgt = torch.randn(B, k, 3)
  valid = torch.ones(B, k)
  wm.update_stats(s, a, ap, xi, rel[:, 0], s_tgt[:, 0], p_tgt[:, 0])
  s1, dp = wm(s, a, ap, xi, rel[:, 0])
  assert s1.shape == (B, chase_st.STATE_DIM) and dp.shape == (B, 3)
  loss = wm.loss(
    s,
    a.unsqueeze(1).expand(B, k, -1),
    ap.unsqueeze(1).expand(B, k, -1),
    xi,
    rel,
    s_tgt,
    p_tgt,
    valid,
  )
  assert torch.isfinite(loss)

  buf = ReplayBuffer(num_envs=2, length=10, obs_dim=45, device="cpu")
  for i in range(6):
    buf.add(
      torch.randn(2, chase_st.STATE_DIM),
      torch.randn(2, 3),
      torch.randn(2, chase_st.NUM_JOINTS),
      torch.randn(2, chase_st.NUM_JOINTS),
      torch.randn(2, 45),
      torch.randn(2, GEN_DIM),
      torch.tensor([1.0 if i == 2 else 0.0, 0.0]),
      torch.randn(2, SCAN_POINTS),
    )
  env, idx, v = buf.sample_windows(8, 3)
  assert env.shape == (8,) and idx.shape == (8, 4) and v.shape == (8, 3)

  arch = WindowArchive(capacity=16, k=3, context_dim=chase_st.CONTEXT_DIM, device="cpu")
  arch.add(
    {
      "s": buf.s[env.unsqueeze(-1), idx],
      "pose": buf.pose[env.unsqueeze(-1), idx],
      "a": buf.a[env.unsqueeze(-1), idx[:, :3]],
      "a_prev": buf.a_prev[env.unsqueeze(-1), idx[:, :3]],
      "xi": xi[:8] if False else torch.randn(8, chase_st.CONTEXT_DIM),
      "valid": v,
      "relief": buf.relief[env.unsqueeze(-1), idx[:, :3]],
    }
  )
  out = arch.sample(4)
  assert set(out.keys()) == {"s", "pose", "a", "a_prev", "xi", "valid", "relief"}
  assert out["s"].shape == (4, 4, chase_st.STATE_DIM)
