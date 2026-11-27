"""Fixed-size per-env replay buffer of simulator transitions (oldest replaced first).

Each slot stores the state *before* the action and the executed (noisy) action.
The next state is the following slot of the same env unless `done` is set, in
which case the following slot starts a new episode.
"""

from __future__ import annotations

import torch

from go2_supermimic.supermimic import state as st
from go2_supermimic.target.generator import BOOL_FIELDS, STATE_DIMS
from go2_supermimic.target.terrain import SCAN_POINTS

GEN_DIM = sum(STATE_DIMS.values())


def pack_generator(gen) -> torch.Tensor:
  parts = [getattr(gen, name).float().reshape(-1, dim) for name, dim in STATE_DIMS.items()]
  return torch.cat(parts, dim=-1)


def unpack_generator(gen, packed: torch.Tensor) -> torch.Tensor:
  """Write packed generator state into `gen` (batch must match)."""
  i = 0
  for name, dim in STATE_DIMS.items():
    x = packed[:, i : i + dim]
    if name in BOOL_FIELDS:
      x = x[:, 0] > 0.5
    elif dim == 1:
      x = x[:, 0]
    setattr(gen, name, x.clone())
    i += dim


class ReplayBuffer:
  def __init__(
    self,
    num_envs: int,
    length: int,
    obs_dim: int,
    device: str,
    gen_dim: int = GEN_DIM,
    *,
    state_dim: int = st.STATE_DIM,
    act_dim: int = st.NUM_JOINTS,
    scan_dim: int = SCAN_POINTS,
    q_ref_arm_dim: int = 0,
    context_dim: int = 0,
    cmd_vel_dim: int = 0,
    ee_pose_cmd_dim: int = 0,
    ee_traj_dim: int = 0,
  ):
    self.n, self.L, self.device = num_envs, length, device
    self.state_dim = int(state_dim)
    self.act_dim = int(act_dim)
    self.scan_dim = int(scan_dim)
    self.gen_dim = int(gen_dim)
    self.q_ref_arm_dim = int(q_ref_arm_dim)
    self.context_dim = int(context_dim)
    self.cmd_vel_dim = int(cmd_vel_dim)
    self.ee_pose_cmd_dim = int(ee_pose_cmd_dim)
    self.ee_traj_dim = int(ee_traj_dim)

    f = lambda *shape: torch.zeros(num_envs, length, *shape, device=device)  # noqa: E731
    self.s = f(self.state_dim)
    self.pose = f(3)
    self.a = f(self.act_dim)
    self.a_prev = f(self.act_dim)
    # Observations dominate memory; half precision is plenty for seeding rollouts.
    self.obs = torch.zeros(num_envs, length, obs_dim, device=device, dtype=torch.float16)
    self.relief = torch.zeros(num_envs, length, self.scan_dim, device=device, dtype=torch.float16)
    self.gen = f(self.gen_dim)
    self.done = f()

    self.q_ref_arm = f(self.q_ref_arm_dim) if self.q_ref_arm_dim > 0 else None
    self.xi = f(self.context_dim) if self.context_dim > 0 else None
    self.cmd_vel = f(self.cmd_vel_dim) if self.cmd_vel_dim > 0 else None
    self.ee_pose_cmd = f(self.ee_pose_cmd_dim) if self.ee_pose_cmd_dim > 0 else None
    self.ee_traj_state = f(self.ee_traj_dim) if self.ee_traj_dim > 0 else None

    self.ptr = 0
    self.size = 0

  def add(
    self,
    s,
    pose,
    a,
    a_prev,
    obs,
    gen,
    done,
    relief=None,
    *,
    q_ref_arm=None,
    xi=None,
    cmd_vel=None,
    ee_pose_cmd=None,
    ee_traj_state=None,
  ) -> None:
    i = self.ptr
    self.s[:, i], self.pose[:, i], self.a[:, i], self.a_prev[:, i] = s, pose, a, a_prev
    if self.scan_dim > 0:
      if relief is None:
        raise ValueError("relief is required when scan_dim > 0")
      self.relief[:, i] = relief.half()
    if self.gen_dim > 0 and gen is not None:
      self.gen[:, i] = gen
    self.obs[:, i] = obs.half()
    self.done[:, i] = done.reshape(-1).float()
    if self.q_ref_arm is not None:
      if q_ref_arm is None:
        raise ValueError("q_ref_arm is required when q_ref_arm_dim > 0")
      self.q_ref_arm[:, i] = q_ref_arm
    if self.xi is not None:
      if xi is None:
        raise ValueError("xi is required when context_dim > 0")
      self.xi[:, i] = xi
    if self.cmd_vel is not None:
      if cmd_vel is None:
        raise ValueError("cmd_vel is required when cmd_vel_dim > 0")
      self.cmd_vel[:, i] = cmd_vel
    if self.ee_pose_cmd is not None:
      if ee_pose_cmd is None:
        raise ValueError("ee_pose_cmd is required when ee_pose_cmd_dim > 0")
      self.ee_pose_cmd[:, i] = ee_pose_cmd
    if self.ee_traj_state is not None:
      if ee_traj_state is None:
        raise ValueError("ee_traj_state is required when ee_traj_dim > 0")
      self.ee_traj_state[:, i] = ee_traj_state
    self.ptr = (self.ptr + 1) % self.L
    self.size = min(self.size + 1, self.L)

  def _phys(self, logical: torch.Tensor) -> torch.Tensor:
    start = (self.ptr - self.size) % self.L
    return (start + logical) % self.L

  def sample_windows(self, batch: int, k: int):
    """Random k-step windows: (env_ids, slot indices (B, k+1), valid (B, k))."""
    assert self.size > k + 1, "buffer too small for windows"
    env = torch.randint(0, self.n, (batch,), device=self.device)
    t0 = torch.randint(0, self.size - k, (batch,), device=self.device)
    idx = self._phys(t0.unsqueeze(-1) + torch.arange(k + 1, device=self.device))
    done = self.done[env.unsqueeze(-1), idx[:, :k]]
    # Transition j (slot j -> j+1) is valid if no slot 0..j ended an episode.
    valid = torch.cumprod(1.0 - done, dim=1)
    return env, idx, valid

  def sample_states(self, batch: int):
    """Random (env_ids, slot) pairs to seed imagined rollouts."""
    env = torch.randint(0, self.n, (batch,), device=self.device)
    t = self._phys(torch.randint(0, self.size, (batch,), device=self.device))
    return env, t


class WbcReplayBuffer(ReplayBuffer):
  """Replay buffer tailored for Go2+Piper 82D WBC world-model & imagination rollouts."""

  def __init__(
    self,
    num_envs: int,
    length: int,
    obs_dim: int = 210,
    device: str = "cpu",
    *,
    state_dim: int = 82,
    act_dim: int = 18,
    q_ref_arm_dim: int = 6,
    context_dim: int = 9,
    cmd_vel_dim: int = 3,
    ee_pose_cmd_dim: int = 7,
    ee_traj_dim: int = 20,
    scan_dim: int = 0,
  ):
    super().__init__(
      num_envs=num_envs,
      length=length,
      obs_dim=obs_dim,
      device=device,
      gen_dim=0,
      state_dim=state_dim,
      act_dim=act_dim,
      scan_dim=scan_dim,
      q_ref_arm_dim=q_ref_arm_dim,
      context_dim=context_dim,
      cmd_vel_dim=cmd_vel_dim,
      ee_pose_cmd_dim=ee_pose_cmd_dim,
      ee_traj_dim=ee_traj_dim,
    )

  def add_wbc(
    self,
    s: torch.Tensor,
    pose: torch.Tensor,
    a: torch.Tensor,
    a_prev: torch.Tensor,
    q_ref_arm: torch.Tensor,
    xi: torch.Tensor,
    obs: torch.Tensor,
    cmd_vel: torch.Tensor,
    ee_pose_cmd: torch.Tensor,
    ee_traj_state: torch.Tensor,
    done: torch.Tensor,
    relief: torch.Tensor | None = None,
  ) -> None:
    """Append one batched WBC step to the ring buffer."""
    self.add(
      s=s,
      pose=pose,
      a=a,
      a_prev=a_prev,
      obs=obs,
      gen=None,
      done=done,
      relief=relief,
      q_ref_arm=q_ref_arm,
      xi=xi,
      cmd_vel=cmd_vel,
      ee_pose_cmd=ee_pose_cmd,
      ee_traj_state=ee_traj_state,
    )

  def sample_windows(self, batch_size: int, horizon: int) -> dict[str, torch.Tensor]:
    """Sample ``horizon``-step WBC windows with full tensors and ``valid`` mask.

    Strictly masks transitions across ``done`` boundaries identically to
    ``ReplayBuffer.sample_windows``: ``valid[:, j] == 0`` for any transition at
    or after a slot with ``done == 1``.
    """
    env, idx, valid = super().sample_windows(batch_size, horizon)
    e = env.unsqueeze(-1)
    idx_k = idx[:, :horizon]
    out: dict[str, torch.Tensor] = {
      "env": env,
      "idx": idx,
      "valid": valid,
      "s": self.s[e, idx],
      "pose": self.pose[e, idx],
      "a": self.a[e, idx_k],
      "a_prev": self.a_prev[e, idx_k],
      "q_ref_arm": self.q_ref_arm[e, idx_k],
      "q_ref_arm_seq": self.q_ref_arm[e, idx],
      "xi": self.xi[env, idx[:, 0]],
      "cmd_vel": self.cmd_vel[e, idx],
      "ee_pose_cmd": self.ee_pose_cmd[e, idx],
      "ee_traj_state": self.ee_traj_state[e, idx],
      "obs": self.obs[e, idx],
    }
    if self.scan_dim > 0:
      out["relief"] = self.relief[e, idx_k]
    else:
      out["relief"] = self.relief[e, idx_k]  # zero-width (B, horizon, 0)
    return out

  def sample_seed(self, batch_size: int) -> dict[str, torch.Tensor]:
    """Sample single-slot initial states to seed WBC imagined rollouts."""
    env, t = self.sample_states(batch_size)
    return {
      "env": env,
      "s": self.s[env, t],
      "pose": self.pose[env, t],
      "a_prev": self.a_prev[env, t],
      "q_ref_arm": self.q_ref_arm[env, t],
      "xi": self.xi[env, t],
      "obs": self.obs[env, t].float(),
      "cmd_vel": self.cmd_vel[env, t],
      "ee_pose_cmd": self.ee_pose_cmd[env, t],
      "ee_traj_state": self.ee_traj_state[env, t],
    }


WINDOW_KEYS = ("s", "pose", "a", "a_prev", "xi", "valid", "relief")


class WindowArchive:
  """Ring store of k-step world-model windows with rare events (contact changes,
  flight, falls). Unlike the replay buffer it is not overwritten every few
  seconds of sim time, so the world model keeps seeing jumps and landings."""

  def __init__(
    self,
    capacity: int,
    k: int,
    context_dim: int,
    device: str,
    *,
    state_dim: int = st.STATE_DIM,
    act_dim: int = st.NUM_JOINTS,
    scan_dim: int = SCAN_POINTS,
    q_ref_arm_dim: int = 0,
  ):
    self.cap, self.device = capacity, device
    self.scan_dim = int(scan_dim)
    self.q_ref_arm_dim = int(q_ref_arm_dim)
    z = lambda *shape: torch.zeros(capacity, *shape, device=device)  # noqa: E731
    self.data: dict[str, torch.Tensor] = {
      "s": z(k + 1, state_dim),
      "pose": z(k + 1, 3),
      "a": z(k, act_dim),
      "a_prev": z(k, act_dim),
      "xi": z(context_dim),
      "valid": z(k),
      "relief": torch.zeros(capacity, k, self.scan_dim, device=device, dtype=torch.float16),
    }
    if self.q_ref_arm_dim > 0:
      self.data["q_ref_arm"] = z(k, self.q_ref_arm_dim)
    self._keys = tuple(self.data.keys())
    self.ptr = 0
    self.size = 0

  def add(self, windows: dict[str, torch.Tensor]) -> None:
    n = windows["s"].shape[0]
    if n == 0:
      return
    n = min(n, self.cap)
    idx = (self.ptr + torch.arange(n, device=self.device)) % self.cap
    for k in self._keys:
      if k == "relief" and self.scan_dim == 0 and k not in windows:
        continue
      self.data[k][idx] = windows[k][:n]
    self.ptr = (self.ptr + n) % self.cap
    self.size = min(self.size + n, self.cap)

  def sample(self, batch: int) -> dict[str, torch.Tensor]:
    idx = torch.randint(0, self.size, (batch,), device=self.device)
    return {k: v[idx] for k, v in self.data.items()}


class WbcWindowArchive(WindowArchive):
  """WindowArchive configured for Go2+Piper 82D WBC world-model windows."""

  def __init__(
    self,
    capacity: int,
    k: int,
    device: str,
    *,
    context_dim: int = 9,
    state_dim: int = 82,
    act_dim: int = 18,
    scan_dim: int = 0,
    q_ref_arm_dim: int = 6,
  ):
    super().__init__(
      capacity=capacity,
      k=k,
      context_dim=context_dim,
      device=device,
      state_dim=state_dim,
      act_dim=act_dim,
      scan_dim=scan_dim,
      q_ref_arm_dim=q_ref_arm_dim,
    )
