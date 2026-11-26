"""SuperMimic training loop for 18-DoF Go2+Piper WBC (Flat & MicroRough, Stage 1 & Stage 2)."""

from __future__ import annotations

import os
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from go2_supermimic.supermimic.buffer import WbcReplayBuffer, WbcWindowArchive
from go2_supermimic.supermimic.train import (
  Actor,
  Critic,
  TrainCfg,
  clip_grads,
  expand_world_model_scan,
  lr_at,
)
from go2_supermimic.supermimic.world_model import WorldModel
from go2_supermimic.target import terrain as tr
from go2_supermimic.wbc import state as wbc_st
from go2_supermimic.wbc.commands import WbcEePoseCommandCfg
from go2_supermimic.wbc.imagine import (
  WBC_STATE_LAYOUT,
  WbcImagineCfg,
  WbcImagination,
  WbcRewardWeights,
)


@dataclass(kw_only=True)
class WbcTrainCfg(TrainCfg):
  """Training defaults tailored for pure from-scratch 18-DoF Go2+Piper WBC."""

  task: str = "Go2Piper-WBC-Flat"
  driver_policy: str = ""
  warmup_iterations: int = 300
  explore_frac: float = 0.35
  explore_noise: float = 0.35
  wbc_stage: int = 1
  compile: bool = False


class WbcTrainer:
  """Three-phase SuperMimic trainer for Go2+Piper WBC (210D -> 18D)."""

  def __init__(self, cfg: TrainCfg):
    import go2_supermimic.env  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    if cfg.wbc_stage not in (1, 2):
      raise ValueError(f"wbc_stage must be 1 or 2, got {cfg.wbc_stage}")

    self.cfg = cfg
    torch.manual_seed(cfg.seed)

    resume_ckpt = (
      torch.load(cfg.resume, map_location="cpu", weights_only=False)
      if cfg.resume
      else None
    )

    env_cfg = load_env_cfg(cfg.task)
    is_micro_rough = env_cfg.scene.terrain.terrain_type == "generator"
    self.scan_dim = tr.SCAN_POINTS if is_micro_rough else 0

    if env_cfg.commands["ee_pose"].stage != cfg.wbc_stage:
      env_cfg.commands["ee_pose"] = WbcEePoseCommandCfg(
        entity_name="robot",
        body_name="end_effector",
        link_name="link0",
        resampling_time_range=(8.0, 10.0),
        stage=cfg.wbc_stage,
        curriculum_progress=env_cfg.commands["ee_pose"].curriculum_progress,
        trajectory_step_dt=0.02,
        debug_vis=True,
      )

    env_cfg.scene.num_envs = cfg.num_envs
    env_cfg.seed = cfg.seed

    saved_step = 0
    new_end_step = int(cfg.target_curriculum_iterations)
    if resume_ckpt is not None and cfg.continue_run:
      saved_step = int(
        resume_ckpt.get("common_step_counter", resume_ckpt.get("iteration", 0))
      )
      saved_cur_state = resume_ckpt.get("curriculum_state", {})
      saved_end_fallback = int(
        resume_ckpt.get("cfg", {}).get("target_curriculum_iterations", new_end_step)
      )
      for cname, pkey in (
        ("base_velocity_levels", "vel_curriculum_progress"),
        ("ee_pose_levels", "ee_curriculum_progress"),
      ):
        if cname not in env_cfg.curriculum:
          continue
        params = env_cfg.curriculum[cname].params
        c_state = saved_cur_state.get(cname) if isinstance(saved_cur_state, dict) else None
        if isinstance(c_state, dict):
          saved_end_step = int(c_state["end_step"])
          if new_end_step == saved_end_step:
            params["start_step"] = int(c_state["start_step"])
            params["end_step"] = saved_end_step
            params["init_progress"] = float(c_state["init_progress"])
          else:
            params["start_step"] = saved_step
            params["end_step"] = max(saved_step + 1, new_end_step)
            params["init_progress"] = float(c_state["progress"])
        else:
          params["end_step"] = new_end_step
          if pkey in resume_ckpt and saved_end_fallback != new_end_step:
            params["start_step"] = saved_step
            params["end_step"] = max(saved_step + 1, new_end_step)
            params["init_progress"] = float(resume_ckpt[pkey])
    else:
      for cname in ("base_velocity_levels", "ee_pose_levels"):
        if cname in env_cfg.curriculum:
          env_cfg.curriculum[cname].params["end_step"] = new_end_step

    self.env_cfg = env_cfg
    self.env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
    self.dev = self.env.device

    if resume_ckpt is not None and cfg.continue_run:
      # Restore common_step_counter and sync command curricula BEFORE the very first reset().
      self.env.common_step_counter = saved_step
      self.env.curriculum_manager.compute()

    obs, _ = self.env.reset()
    self.obs = obs["actor"]
    obs_dim = int(self.obs.shape[-1])
    self.consts = wbc_st.wbc_robot_consts(self.env)
    self.ctx = wbc_st.WbcContextReader(self.env)
    self.xi = self.ctx.read()

    robot_data = self.env.scene["robot"].data
    enc_bias = getattr(robot_data, "encoder_bias", None)
    self.enc_bias = (
      enc_bias.detach().clone()
      if enc_bias is not None
      else torch.zeros((cfg.num_envs, wbc_st.WBC_NUM_JOINTS), device=self.dev)
    )

    if is_micro_rough:
      cache_root = os.environ.get("XDG_CACHE_HOME", cfg.log_root)
      exclude = {int(g) for g in self.env.scene["robot"].indexing.geom_ids}
      self.field: tr.HeightField | None = tr.env_height_field(
        self.env,
        exclude,
        half=8.0,
        res=0.05,
        cache_dir=str(Path(cache_root) / "heightfields"),
      )
    else:
      self.field = None

    self.wm = WorldModel(
      cfg.wm_hidden,
      state_dim=wbc_st.WBC_STATE_DIM,
      act_dim=wbc_st.WBC_NUM_JOINTS,
      context_dim=wbc_st.WBC_CONTEXT_DIM,
      scan_dim=self.scan_dim,
      q_ref_arm_dim=wbc_st.WBC_Q_REF_ARM_DIM,
      pose_delta_dim=wbc_st.WBC_POSE_DELTA_DIM,
      z_slice=wbc_st.Z,
      state_slices=WBC_STATE_LAYOUT,
    ).to(self.dev)
    self.actor = Actor(obs_dim, cfg.actor_hidden, wbc_st.WBC_NUM_JOINTS).to(self.dev)
    self.opt_wm = torch.optim.Adam(self.wm.parameters(), lr=cfg.lr)
    self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr, betas=(0.9, 0.95))

    self.buf = WbcReplayBuffer(
      cfg.num_envs,
      cfg.buffer_len,
      obs_dim,
      self.dev,
      scan_dim=self.scan_dim,
    )
    self.archive = WbcWindowArchive(
      cfg.archive_size,
      cfg.wm_window,
      self.dev,
      scan_dim=self.scan_dim,
    )
    self.explore = (
      torch.arange(cfg.num_envs, device=self.dev) < int(cfg.explore_frac * cfg.num_envs)
    )
    self.ou = torch.zeros(cfg.num_envs, wbc_st.WBC_NUM_JOINTS, device=self.dev)

    imag_cfg = WbcImagineCfg(
      horizon=cfg.horizon,
      compile=cfg.compile,
      sensor_grad=cfg.sensor_grad,
      weights=WbcRewardWeights(),
    )
    self.imag = WbcImagination(
      imag_cfg,
      self.consts,
      self.dev,
      scan_dim=self.scan_dim,
      field=self.field,
    )
    self.critic = Critic(self.imag.critic_dim, cfg.critic_hidden).to(self.dev)
    self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr)

    self.a_prev = torch.zeros(cfg.num_envs, wbc_st.WBC_NUM_JOINTS, device=self.dev)

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    self.log_dir = Path(cfg.log_root) / f"{stamp}_{cfg.run_name}"
    self.log_dir.mkdir(parents=True, exist_ok=True)
    self.writer = SummaryWriter(str(self.log_dir))
    self.ep_stats: dict[str, list[float]] = defaultdict(list)
    self.env_steps = 0
    self.actor_frozen_until = 0
    self.policy_start = cfg.warmup_iterations
    self.lr_scale = 1.0
    self.ramp_start = 0
    self.last_ckpt: Path | None = None
    self.collapse_strikes = 0
    self.nonfinite_steps = 0
    self.start_it = 0
    self.it = 0
    if resume_ckpt is not None:
      self.load(cfg.resume, ckpt=resume_ckpt)

  def load(self, path: str, *, ckpt: dict | None = None) -> None:
    if ckpt is None:
      ckpt = torch.load(path, map_location=self.dev, weights_only=False)
    else:
      # Move loaded tensors onto target device if peeked on CPU.
      ckpt = {
        k: (
          {sk: sv.to(self.dev) if torch.is_tensor(sv) else sv for sk, sv in v.items()}
          if isinstance(v, dict)
          else v
        )
        for k, v in ckpt.items()
      }

    wm_sd = self.wm.state_dict()
    same_wm = all(
      k in ckpt["world_model"] and ckpt["world_model"][k].shape == v.shape
      for k, v in wm_sd.items()
    )
    if same_wm:
      self.wm.load_state_dict(ckpt["world_model"])
      if "opt_wm" in ckpt:
        self.opt_wm.load_state_dict(ckpt["opt_wm"])
    elif expand_world_model_scan(ckpt["world_model"], self.wm):
      print("world model: ground-scan inputs added (zero weight, same predictions)", flush=True)
    else:
      self.policy_start = max(self.policy_start, self.cfg.wm_rewarmup_iterations)
      self.actor_frozen_until = max(
        self.actor_frozen_until, self.policy_start + self.cfg.critic_warmup_iterations
      )
      print(
        f"world model re-initialised (shape changed); policy updates from iteration "
        f"{self.policy_start}",
        flush=True,
      )

    old_obs_dim = int(ckpt["actor"]["norm.mean"].shape[0])
    new_obs_dim = int(self.obs.shape[-1])
    if old_obs_dim != new_obs_dim:
      raise ValueError(f"WBC actor obs_dim mismatch on resume: {old_obs_dim} -> {new_obs_dim}")
    self.actor.load_state_dict(ckpt["actor"])
    if "opt_actor" in ckpt:
      self.opt_actor.load_state_dict(ckpt["opt_actor"])

    critic_sd = self.critic.state_dict()
    same_critic = all(
      k in ckpt["critic"] and ckpt["critic"][k].shape == v.shape
      for k, v in critic_sd.items()
    )
    if same_critic and not self.cfg.reset_critic:
      self.critic.load_state_dict(ckpt["critic"])
      if "opt_critic" in ckpt:
        self.opt_critic.load_state_dict(ckpt["opt_critic"])
    else:
      self.actor_frozen_until = max(
        self.actor_frozen_until, self.policy_start + self.cfg.critic_warmup_iterations
      )
      print(
        f"critic re-initialised; actor frozen until iteration {self.actor_frozen_until}",
        flush=True,
      )

    print(f"resumed from {path} (iteration {ckpt.get('iteration')})", flush=True)
    if self.cfg.continue_run:
      self.start_it = int(ckpt.get("iteration", 0))
      self.it = self.start_it
      saved_step = int(ckpt.get("common_step_counter", self.start_it))
      self.env.common_step_counter = saved_step
      self.env.curriculum_manager.compute()
      print(
        f"continuing at iteration {self.start_it}, common_step_counter={saved_step} "
        f"(wbc_stage={self.cfg.wbc_stage})",
        flush=True,
      )

  @torch.inference_mode()
  def collect(self) -> None:
    env, cfg = self.env, self.cfg
    action_term = env.action_manager.get_term("joint_pos")
    vel_cmd_term = env.command_manager.get_term("base_velocity")
    pose_cmd_term = env.command_manager.get_term("ee_pose")
    clip = self.consts.action_clip
    for _ in range(cfg.collect_steps):
      s = wbc_st.wbc_robot_state(env, field=self.field)
      pose = wbc_st.wbc_planar_pose(env)
      q_ref_arm = action_term.q_ref_arm.clone()
      cmd_vel = vel_cmd_term.command.clone()
      ee_pose_cmd = pose_cmd_term.command.clone()
      ee_traj_state = pose_cmd_term.traj_state.clone()
      xi = self.xi
      relief = self._relief(pose)

      a = self.actor(self.obs)
      dt = env.step_dt
      sigma_ou = cfg.explore_noise * (2 * cfg.explore_theta) ** 0.5
      self.ou = self.ou * (1 - cfg.explore_theta * dt) + sigma_ou * dt**0.5 * torch.randn_like(a)
      noise = torch.where(
        self.explore.unsqueeze(-1), self.ou, cfg.action_noise * torch.randn_like(a)
      )
      a = a + noise
      a_exec = torch.clamp(a, clip[:, :, 0], clip[:, :, 1]) if clip is not None else a

      obs, _, term, trunc, extras = env.step(a_exec)
      done = term | trunc
      self.ou = torch.where(done.unsqueeze(-1), torch.zeros_like(self.ou), self.ou)

      self.buf.add_wbc(
        s=s,
        pose=pose,
        a=a_exec,
        a_prev=self.a_prev,
        q_ref_arm=q_ref_arm,
        xi=xi,
        obs=self.obs,
        cmd_vel=cmd_vel,
        ee_pose_cmd=ee_pose_cmd,
        ee_traj_state=ee_traj_state,
        done=done,
        relief=relief,
      )

      keep = ~done
      if bool(keep.any()):
        s1 = wbc_st.wbc_robot_state(env, field=self.field)
        dpose = wbc_st.wbc_pose_delta(pose, wbc_st.wbc_planar_pose(env))
        self.wm.update_stats(
          s[keep],
          a_exec[keep],
          self.a_prev[keep],
          xi[keep],
          relief[keep] if self.scan_dim > 0 else None,
          s1[keep],
          dpose[keep],
          q_ref_arm=q_ref_arm[keep],
        )

      self.actor.norm.update(self.obs)
      if bool(done.any()):
        # Leg & arm PD gains are randomized on reset, so refresh xi.
        self.xi = self.ctx.read()
      self.a_prev = torch.where(done.unsqueeze(-1), torch.zeros_like(a_exec), a_exec)
      self.obs = obs["actor"]
      self.env_steps += env.num_envs
      for k_log, v_log in extras.get("log", {}).items():
        if k_log.startswith(("Episode_Reward", "Episode_Termination", "Metrics/", "Curriculum/")):
          self.ep_stats[k_log].append(float(v_log))

  def _relief(self, pose: torch.Tensor) -> torch.Tensor | None:
    if self.scan_dim <= 0:
      return None
    return tr.relief(pose[:, :2], pose[:, 2], self.field)

  @staticmethod
  def _archive_keys(w: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
      "s": w["s"],
      "pose": w["pose"],
      "a": w["a"],
      "a_prev": w["a_prev"],
      "q_ref_arm": w["q_ref_arm"],
      "xi": w["xi"],
      "valid": w["valid"],
      "relief": w["relief"],
    }

  @staticmethod
  def _rarity(w: dict[str, torch.Tensor]) -> torch.Tensor:
    s = w["s"]
    contact = wbc_st.wbc_feet_world_z(s.reshape(-1, wbc_st.WBC_STATE_DIM)) < 0.035
    contact = contact.reshape(s.shape[0], -1, 4).float()
    changes = (contact[:, 1:] - contact[:, :-1]).abs().sum((1, 2))
    airborne = (contact.sum(-1) == 0).float().sum(-1)
    ended = (w["valid"][:, -1] < 0.5).float() * 4.0
    return changes + airborne + ended

  @staticmethod
  def _archive_worthy(w: dict[str, torch.Tensor]) -> torch.Tensor:
    s = w["s"]
    feet_z = wbc_st.wbc_feet_world_z(s.reshape(-1, wbc_st.WBC_STATE_DIM)).reshape(
      s.shape[0], -1, 4
    )
    flight = ((feet_z >= 0.035).all(-1).float().sum(-1)) >= 2
    hard_vz = s[:, :, wbc_st.V_YAW][..., 2].abs().amax(-1) > 0.6
    fell = w["valid"][:, -1] < 0.5
    return flight | hard_vz | fell

  def _wm_batch(self) -> dict[str, torch.Tensor]:
    cfg, b = self.cfg, self.buf
    n_prio = int(cfg.wm_batch * cfg.wm_prioritized_frac)
    n_arch = int(cfg.wm_batch * cfg.wm_archive_frac) if self.archive.size > 0 else 0
    n_uni = cfg.wm_batch - n_prio - n_arch

    parts = [self._archive_keys(b.sample_windows(n_uni, cfg.wm_window))]
    cand = self._archive_keys(b.sample_windows(4 * max(n_prio, 1), cfg.wm_window))
    score = self._rarity(cand)
    keep = self._archive_worthy(cand) & (torch.rand_like(score) < cfg.archive_keep_prob)
    self.archive.add({k: v[keep] for k, v in cand.items()})
    top = (score + 1e-3 * torch.rand_like(score)).topk(n_prio).indices
    parts.append({k: v[top] for k, v in cand.items()})
    if n_arch:
      parts.append(self.archive.sample(n_arch))
    return {k: torch.cat([p[k] for p in parts]) for k in parts[0]}

  def fit_world_model(self) -> float:
    cfg = self.cfg
    k = cfg.wm_window
    losses = []
    for _ in range(cfg.wm_updates):
      w = self._wm_batch()
      pose_tgt = torch.stack(
        [wbc_st.wbc_pose_delta(w["pose"][:, j], w["pose"][:, j + 1]) for j in range(k)], 1
      )
      reliefs = w["relief"].float() if self.scan_dim > 0 else None
      loss = self.wm.loss(
        w["s"][:, 0],
        w["a"],
        w["a_prev"],
        w["xi"],
        reliefs,
        w["s"][:, 1:],
        pose_tgt,
        w["valid"],
        q_ref_arms=w["q_ref_arm"],
      )
      self.opt_wm.zero_grad(set_to_none=True)
      loss.backward()
      gn = clip_grads(self.wm.parameters(), cfg.grad_clip, float("inf"))
      if bool(torch.isfinite(gn)) and bool(torch.isfinite(loss)):
        self.opt_wm.step()
      else:
        self.nonfinite_steps += 1
      losses.append(float(loss.item()))
    return sum(losses) / len(losses)

  def improve_policy(self) -> dict[str, float]:
    cfg, b = self.cfg, self.buf
    seed = b.sample_seed(cfg.imag_batch)
    seed["enc_bias"] = self.enc_bias[seed["env"]]

    wm_params = list(self.wm.parameters())
    crit_params = list(self.critic.parameters())
    wm_req = [p.requires_grad for p in wm_params]
    crit_req = [p.requires_grad for p in crit_params]
    for p in wm_params:
      p.requires_grad_(False)
    for p in crit_params:
      p.requires_grad_(False)
    try:
      objective, data = self.imag.rollout(self.wm, self.actor, self.critic, seed)
      loss = -objective.mean()
      self.opt_actor.zero_grad(set_to_none=True)
      loss.backward()
      gnorm = clip_grads(self.actor.parameters(), cfg.grad_clip, cfg.grad_clip_elem)
      finite = bool(torch.isfinite(gnorm)) and bool(torch.isfinite(loss))
      if not finite:
        self.nonfinite_steps += 1
      if self.it > self.actor_frozen_until and finite:
        self.opt_actor.step()
    finally:
      for p, req in zip(wm_params, wm_req, strict=True):
        p.requires_grad_(req)
      for p, req in zip(crit_params, crit_req, strict=True):
        p.requires_grad_(req)

    targets = self.imag.gae_targets(self.critic, data)
    x = data["critic_in"][:, :-1].detach().reshape(-1, data["critic_in"].shape[-1])
    y = targets.reshape(-1)
    self.critic.norm.update(x)
    n = x.shape[0]
    mb = max(1, n // cfg.critic_minibatches)
    c_losses = []
    for _ in range(cfg.critic_epochs):
      perm = torch.randperm(n, device=self.dev)
      for i in range(cfg.critic_minibatches):
        sel = perm[i * mb : (i + 1) * mb]
        if sel.numel() == 0:
          continue
        c_loss = (self.critic(x[sel]).squeeze(-1) - y[sel]).square().mean()
        self.opt_critic.zero_grad(set_to_none=True)
        c_loss.backward()
        gn = clip_grads(self.critic.parameters(), cfg.grad_clip, float("inf"))
        if bool(torch.isfinite(gn)) and bool(torch.isfinite(c_loss)):
          self.opt_critic.step()
        c_losses.append(float(c_loss.item()))
    return {
      "imagined_return": float(objective.mean().item()),
      "imagined_reward": float(data["rewards"].mean().item()),
      "imagined_survival": float(data["surv"][:, -1].mean().item()),
      "actor_grad_norm": float(gnorm),
      "critic_loss": sum(c_losses) / max(1, len(c_losses)),
    }

  def save(self, it: int) -> Path:
    path = self.log_dir / f"model_{it}.pt"
    self.env.curriculum_manager.compute()
    vel_term = self.env.command_manager.get_term("base_velocity")
    ee_term = self.env.command_manager.get_term("ee_pose")
    curriculum_state = {}
    for cname, term in (
      ("base_velocity_levels", vel_term),
      ("ee_pose_levels", ee_term),
    ):
      if cname in self.env_cfg.curriculum:
        params = self.env_cfg.curriculum[cname].params
        curriculum_state[cname] = {
          "start_step": int(params.get("start_step", 0)),
          "end_step": int(params.get("end_step", self.cfg.target_curriculum_iterations)),
          "init_progress": float(params.get("init_progress", 0.0)),
          "progress": float(term.cfg.curriculum_progress),
        }
    torch.save(
      {
        "supermimic": True,
        "wbc": True,
        "wbc_stage": self.cfg.wbc_stage,
        "iteration": it,
        "common_step_counter": int(self.env.common_step_counter),
        "vel_curriculum_progress": float(vel_term.cfg.curriculum_progress),
        "ee_curriculum_progress": float(ee_term.cfg.curriculum_progress),
        "curriculum_state": curriculum_state,
        "cfg": asdict(self.cfg),
        "actor": self.actor.state_dict(),
        "critic": self.critic.state_dict(),
        "world_model": self.wm.state_dict(),
        "opt_wm": self.opt_wm.state_dict(),
        "opt_actor": self.opt_actor.state_dict(),
        "opt_critic": self.opt_critic.state_dict(),
        "obs_dim": int(self.obs.shape[-1]),
        "act_dim": wbc_st.WBC_NUM_JOINTS,
        "actor_hidden": tuple(self.cfg.actor_hidden),
      },
      path,
    )
    return path

  def close(self) -> None:
    self.writer.close()
    self.env.close()

  def run(self) -> None:
    cfg = self.cfg
    t_start = time.time()
    try:
      while self.buf.size <= cfg.wm_window + 1:
        self.collect()
      for it in range(self.start_it + 1, cfg.iterations + 1):
        self.it = it
        lr = lr_at(cfg, it) * self.lr_scale
        start = max(self.ramp_start, self.actor_frozen_until, self.policy_start)
        ramp = min(1.0, max(0.0, it - start) / max(1, cfg.actor_ramp_iterations))
        actor_scale = ramp * cfg.actor_lr_scale
        for opt, scale in (
          (self.opt_wm, 1.0),
          (self.opt_actor, actor_scale),
          (self.opt_critic, 1.0),
        ):
          for g in opt.param_groups:
            g["lr"] = lr * scale
        self.collect()
        wm_loss = self.fit_world_model()
        stats: dict[str, float] = {"wm_loss": wm_loss}
        if it > self.policy_start:
          stats.update(self.improve_policy())
        if it % cfg.log_interval == 0:
          elapsed = time.time() - t_start
          for k_stat, v_stat in stats.items():
            self.writer.add_scalar(f"train/{k_stat}", v_stat, it)
          line = [
            f"it {it}/{cfg.iterations}",
            f"{elapsed / 60:.1f} min",
            f"env_steps {self.env_steps / 1e6:.2f}M",
            f"step_cnt {int(self.env.common_step_counter)}",
          ]
          line += [f"{k_stat} {v_stat:.4g}" for k_stat, v_stat in stats.items()]
          line.append(f"archive {self.archive.size}")
          if self.nonfinite_steps:
            line.append(f"nonfinite_skipped {self.nonfinite_steps}")
          falls = self.ep_stats.get("Episode_Termination/fell_over", [])
          falls_mean = sum(falls) / len(falls) if falls else 0.0
          for k_ep, vals in self.ep_stats.items():
            mean = sum(vals) / len(vals)
            self.writer.add_scalar(k_ep, mean, it)
            if k_ep in (
              "Episode_Termination/fell_over",
              "Metrics/base_velocity/error_vel_xy",
              "Metrics/base_velocity/error_vel_yaw",
              "Metrics/ee_pose/position_error",
              "Metrics/ee_pose/orientation_error",
            ):
              line.append(f"{k_ep.split('/')[-1]} {mean:.3g}")
          self.ep_stats.clear()
          self.writer.add_scalar("time/minutes", elapsed / 60, it)
          print(" | ".join(line), flush=True)
          self._collapse_guard(falls_mean)
        if it % cfg.save_interval == 0 or it == cfg.iterations:
          if self.collapse_strikes == 0:
            self.last_ckpt = self.save(it)
            print(f"saved {self.last_ckpt}", flush=True)
    finally:
      self.close()

  def _collapse_guard(self, falls: float) -> None:
    if self.it <= self.actor_frozen_until or falls <= self.cfg.collapse_falls:
      self.collapse_strikes = 0
      return
    self.collapse_strikes += 1
    if self.collapse_strikes >= 2 and self.last_ckpt is not None:
      ckpt = torch.load(self.last_ckpt, map_location=self.dev, weights_only=False)
      self.actor.load_state_dict(ckpt["actor"])
      self.critic.load_state_dict(ckpt["critic"])
      self.lr_scale *= 0.5
      self.collapse_strikes = 0
      self.ramp_start = self.it
      self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.lr, betas=(0.9, 0.95))
      self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=self.cfg.lr)
      print(
        f"collapse guard: {falls:.1f} falls, rolled back to {self.last_ckpt}, "
        f"lr scale {self.lr_scale:.3g}",
        flush=True,
      )
