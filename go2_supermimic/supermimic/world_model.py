"""Deterministic world model f(s, a, a_prev, [q_ref_arm,] xi[, ground scan]) -> (s', planar pose delta).

Fit on k-step autoregressive windows from the replay buffer (SuperMimic Eq. 1,
state term only; reward and termination are analytic in imagine.py). Inputs
and outputs are normalized with running statistics from collected data.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from go2_supermimic.supermimic import state as st
from go2_supermimic.target.terrain import SCAN_POINTS


@dataclass(frozen=True)
class StateLayout:
  """Slice layout of a world-model state vector for parameterized loss weighting."""

  state_dim: int
  z: slice
  grav: slice
  v_yaw: slice
  w_b: slice
  q: slice
  qd: slice
  hips: slice
  knees: slice
  feet: slice

  def as_dict(self) -> dict[str, slice]:
    return {
      "z": self.z,
      "grav": self.grav,
      "v_yaw": self.v_yaw,
      "w_b": self.w_b,
      "q": self.q,
      "qd": self.qd,
      "hips": self.hips,
      "knees": self.knees,
      "feet": self.feet,
    }


CHASE_STATE_LAYOUT = StateLayout(
  state_dim=st.STATE_DIM,
  z=st.Z,
  grav=st.GRAV,
  v_yaw=st.V_YAW,
  w_b=st.W_B,
  q=st.Q,
  qd=st.QD,
  hips=st.HIPS,
  knees=st.KNEES,
  feet=st.FEET,
)


class RunningNorm(nn.Module):
  """Running mean/std over batches (Welford-style merge), frozen when not updated."""

  def __init__(self, dim: int, eps: float = 1e-4, clip: float = 10.0):
    super().__init__()
    self.register_buffer("mean", torch.zeros(dim))
    self.register_buffer("var", torch.ones(dim))
    self.register_buffer("count", torch.tensor(eps))
    self.clip = clip

  @torch.no_grad()
  def update(self, x: torch.Tensor) -> None:
    x = x.reshape(-1, x.shape[-1])
    b_mean, b_var, b_n = x.mean(0), x.var(0, unbiased=False), x.shape[0]
    delta = b_mean - self.mean
    tot = self.count + b_n
    self.mean += delta * b_n / tot
    self.var = (self.var * self.count + b_var * b_n + delta**2 * self.count * b_n / tot) / tot
    self.count = tot

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return ((x - self.mean) / (self.var.sqrt() + 1e-6)).clamp(-self.clip, self.clip)

  def unnormalize(self, y: torch.Tensor) -> torch.Tensor:
    return y * (self.var.sqrt() + 1e-6) + self.mean


def mlp(in_dim: int, hidden: tuple[int, ...], out_dim: int, act=nn.SiLU) -> nn.Sequential:
  layers, d = [], in_dim
  for h in hidden:
    layers += [nn.Linear(d, h), act()]
    d = h
  layers.append(nn.Linear(d, out_dim))
  return nn.Sequential(*layers)


def _build_loss_weights(
  out_dim: int,
  state_dim: int,
  state_slices: StateLayout | dict[str, slice] | None,
  loss_weights: dict[str, float] | torch.Tensor | None,
) -> torch.Tensor | None:
  if loss_weights is None:
    return None
  if torch.is_tensor(loss_weights):
    w = loss_weights.detach().clone().to(dtype=torch.float32)
    if w.shape != (out_dim,):
      raise ValueError(f"Expected loss_weights shape ({out_dim},), got {tuple(w.shape)}")
    return w
  if state_slices is None:
    raise ValueError("state_slices is required when loss_weights is a dict.")
  slices = state_slices.as_dict() if isinstance(state_slices, StateLayout) else state_slices
  w = torch.ones(out_dim, dtype=torch.float32)
  for key, val in loss_weights.items():
    if key == "pose":
      w[state_dim:] = float(val)
    elif key in slices:
      w[slices[key]] = float(val)
    else:
      raise KeyError(f"Unknown state slice key '{key}' in loss_weights")
  return w


class WorldModel(nn.Module):
  """s' = f(s, a, a_prev, [q_ref_arm,] xi[, scan]).

  Defaults match the original 70D 12-DoF ``Go2-Chase`` configuration.
  For Go2+Piper WBC, pass ``state_dim=82, act_dim=18, context_dim=9,
  q_ref_arm_dim=6`` and ``scan_dim=0`` (flat) or ``scan_dim=77`` (micro-rough).
  """

  def __init__(
    self,
    hidden: tuple[int, ...] = (1024, 1024, 1024),
    state_dim: int = st.STATE_DIM,
    act_dim: int = st.NUM_JOINTS,
    context_dim: int = st.CONTEXT_DIM,
    scan_dim: int = SCAN_POINTS,
    q_ref_arm_dim: int = 0,
    pose_delta_dim: int = st.POSE_DELTA_DIM,
    z_slice: slice = st.Z,
    state_slices: StateLayout | dict[str, slice] | None = None,
    loss_weights: dict[str, float] | torch.Tensor | None = None,
  ):
    super().__init__()
    self.state_dim = int(state_dim)
    self.act_dim = int(act_dim)
    self.context_dim = int(context_dim)
    self.scan_dim = int(scan_dim)
    self.q_ref_arm_dim = int(q_ref_arm_dim)
    self.pose_delta_dim = int(pose_delta_dim)
    self.z_slice = state_slices.z if isinstance(state_slices, StateLayout) else z_slice

    in_dim = self.state_dim + 2 * self.act_dim + self.q_ref_arm_dim + self.context_dim + self.scan_dim
    out_dim = self.state_dim + self.pose_delta_dim
    self.net = mlp(in_dim, hidden, out_dim)
    self.in_norm = RunningNorm(in_dim)
    # Normalizes the per-step change of s and the pose delta.
    self.out_norm = RunningNorm(out_dim)

    w = _build_loss_weights(out_dim, self.state_dim, state_slices, loss_weights)
    # Keep state_dict() keys identical to original checkpoints (persistent=False).
    self.register_buffer("_loss_weights", w, persistent=False)

  def _inputs(
    self,
    s: torch.Tensor,
    a: torch.Tensor,
    a_prev: torch.Tensor,
    xi: torch.Tensor,
    relief: torch.Tensor | None = None,
    q_ref_arm: torch.Tensor | None = None,
  ) -> torch.Tensor:
    parts = [s, a, a_prev]
    if self.q_ref_arm_dim > 0:
      if q_ref_arm is None:
        raise ValueError("q_ref_arm is required when q_ref_arm_dim > 0")
      if q_ref_arm.shape[-1] != self.q_ref_arm_dim:
        raise ValueError(
          f"Expected q_ref_arm trailing dim {self.q_ref_arm_dim}, got {q_ref_arm.shape[-1]}"
        )
      parts.append(q_ref_arm)
    parts.append(xi)
    if self.scan_dim > 0:
      if relief is None:
        raise ValueError("relief is required when scan_dim > 0")
      if relief.shape[-1] != self.scan_dim:
        raise ValueError(f"Expected relief trailing dim {self.scan_dim}, got {relief.shape[-1]}")
      parts.append(s[..., self.z_slice] + relief)
    return torch.cat(parts, dim=-1)

  def update_stats(
    self,
    s: torch.Tensor,
    a: torch.Tensor,
    a_prev: torch.Tensor,
    xi: torch.Tensor,
    relief: torch.Tensor | None,
    s_next: torch.Tensor,
    pose_delta: torch.Tensor,
    q_ref_arm: torch.Tensor | None = None,
  ) -> None:
    self.in_norm.update(self._inputs(s, a, a_prev, xi, relief, q_ref_arm))
    self.out_norm.update(torch.cat([s_next - s, pose_delta], dim=-1))

  def forward(
    self,
    s: torch.Tensor,
    a: torch.Tensor,
    a_prev: torch.Tensor,
    xi: torch.Tensor,
    relief: torch.Tensor | None = None,
    q_ref_arm: torch.Tensor | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Predict ``(s_next, pose_delta)``.

    ``relief`` may be ``None`` or empty ``(..., 0)`` when ``scan_dim == 0``.
    """
    x = self._inputs(s, a, a_prev, xi, relief, q_ref_arm)
    y = self.out_norm.unnormalize(self.net(self.in_norm(x)))
    return s + y[..., : self.state_dim], y[..., self.state_dim :]

  def loss(
    self,
    s0: torch.Tensor,
    actions: torch.Tensor,
    actions_prev: torch.Tensor,
    xi: torch.Tensor,
    reliefs: torch.Tensor | None,
    s_targets: torch.Tensor,
    pose_targets: torch.Tensor,
    valid: torch.Tensor,
    q_ref_arms: torch.Tensor | None = None,
  ) -> torch.Tensor:
    """k-step autoregressive loss in normalized output units."""
    k = actions.shape[1]
    scale = self.out_norm.var.sqrt() + 1e-6
    s = s0
    total, denom = 0.0, valid.sum().clamp(min=1.0)
    for j in range(k):
      r_j = None if reliefs is None else reliefs[:, j]
      q_j = None if q_ref_arms is None else q_ref_arms[:, j]
      s_pred, pose_pred = self(s, actions[:, j], actions_prev[:, j], xi, r_j, q_ref_arm=q_j)
      err = torch.cat([s_pred - s_targets[:, j], pose_pred - pose_targets[:, j]], dim=-1)
      sq = (err / scale) ** 2
      if self._loss_weights is not None:
        sq = sq * self._loss_weights
      total = total + sq.mean(-1).mul(valid[:, j]).sum()
      s = s_pred
    return total / denom
