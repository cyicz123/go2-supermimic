"""TorchScript 210D -> 18D export for Go2+Piper WBC Actor policies.

Produces standalone CPU-loadable, device-movable ``torch.jit.ScriptModule``
checkpoints compatible with ``LeggedManip_Lab@b2cce16`` ``export_wbc_leapp.py``:
  - Input: ``obs`` tensor of shape ``(N, 210)`` (float32, any batch size ``N``)
  - Output: raw pre-clip 18D action tensor of shape ``(N, 18)`` (float32)
"""

from __future__ import annotations

import copy
from pathlib import Path

import torch
from torch import nn

from go2_supermimic.supermimic.train import Actor
from go2_supermimic.supermimic.world_model import mlp
from go2_supermimic.wbc.state import WBC_ACTOR_OBS_DIM, WBC_NUM_JOINTS


class WbcExportActor(nn.Module):
  """Standalone TorchScript-safe 210D -> 18D WBC actor (RunningNorm + ELU MLP)."""

  def __init__(
    self,
    obs_dim: int = WBC_ACTOR_OBS_DIM,
    hidden: tuple[int, ...] = (512, 256, 128),
    act_dim: int = WBC_NUM_JOINTS,
    clip: float = 10.0,
  ):
    super().__init__()
    self.register_buffer("mean", torch.zeros(obs_dim, dtype=torch.float32))
    self.register_buffer("var", torch.ones(obs_dim, dtype=torch.float32))
    self.clip = float(clip)
    self.net = mlp(obs_dim, hidden, act_dim, act=nn.ELU)

  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    normed = ((obs - self.mean) / (self.var.sqrt() + 1e-6)).clamp(-self.clip, self.clip)
    return self.net(normed)


def _infer_actor_arch(actor_sd: dict[str, torch.Tensor]) -> tuple[int, tuple[int, ...], int]:
  obs_dim = int(actor_sd["norm.mean"].shape[0])
  linear_weights = sorted(
    (
      (int(k.split(".")[1]), v)
      for k, v in actor_sd.items()
      if k.startswith("net.") and k.endswith(".weight")
    ),
    key=lambda item: item[0],
  )
  if not linear_weights:
    raise ValueError("Actor state_dict has no net.*.weight layers")
  hidden = tuple(int(w.shape[0]) for _, w in linear_weights[:-1])
  act_dim = int(linear_weights[-1][1].shape[0])
  return obs_dim, hidden, act_dim


def _load_actor_from_checkpoint(ckpt_path: str | Path) -> Actor:
  ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
  if "actor" not in ckpt:
    raise KeyError(f"Checkpoint {ckpt_path} does not contain 'actor' state_dict")
  actor_sd = ckpt["actor"]
  obs_dim, hidden, act_dim = _infer_actor_arch(actor_sd)
  actor = Actor(obs_dim, hidden, act_dim)
  actor.load_state_dict(actor_sd)
  actor.eval()
  return actor


def _actor_to_export_module(actor: Actor) -> tuple[WbcExportActor, Actor]:
  actor_cpu = copy.deepcopy(actor).to(device="cpu", dtype=torch.float32).eval()
  obs_dim, hidden, act_dim = _infer_actor_arch(actor_cpu.state_dict())
  if obs_dim != WBC_ACTOR_OBS_DIM or act_dim != WBC_NUM_JOINTS:
    raise ValueError(
      f"Expected 210D -> 18D WBC actor, got {obs_dim}D -> {act_dim}D"
    )
  export_mod = WbcExportActor(
    obs_dim=obs_dim,
    hidden=hidden,
    act_dim=act_dim,
    clip=float(actor_cpu.norm.clip),
  ).to(device="cpu", dtype=torch.float32).eval()
  with torch.no_grad():
    export_mod.mean.copy_(actor_cpu.norm.mean.to(dtype=torch.float32))
    export_mod.var.copy_(actor_cpu.norm.var.to(dtype=torch.float32))
    export_mod.net.load_state_dict(actor_cpu.net.state_dict())
  return export_mod, actor_cpu


@torch.no_grad()
def verify_wbc_torchscript_parity(
  eager_actor: Actor,
  scripted_path: str | Path,
  *,
  batch_sizes: tuple[int, ...] = (1, 4),
  atol: float = 1e-6,
  seed: int = 0,
) -> float:
  """Verify reloaded TorchScript module matches eager Actor within ``atol``."""
  eager = copy.deepcopy(eager_actor).to(device="cpu", dtype=torch.float32).eval()
  loaded = torch.jit.load(str(scripted_path), map_location="cpu").eval()
  gen = torch.Generator(device="cpu").manual_seed(seed)
  max_err = 0.0
  for n in batch_sizes:
    obs = torch.randn((n, WBC_ACTOR_OBS_DIM), generator=gen, dtype=torch.float32) * 3.0
    ref = eager(obs)
    got = loaded(obs)
    if got.shape != (n, WBC_NUM_JOINTS):
      raise ValueError(f"Expected shape ({n}, {WBC_NUM_JOINTS}), got {tuple(got.shape)}")
    err = float((got - ref).abs().max().item())
    max_err = max(max_err, err)
  if max_err >= atol:
    raise AssertionError(
      f"TorchScript parity check failed: max_abs_err={max_err:.3e} >= atol={atol:.3e}"
    )
  return max_err


def export_wbc_actor_torchscript(
  actor_or_ckpt: Actor | str | Path,
  output_path: str | Path,
  *,
  verify: bool = True,
  atol: float = 1e-6,
) -> Path:
  """Export a WBC 210D -> 18D Actor (or training checkpoint) to TorchScript ``policy.pt``."""
  actor = (
    actor_or_ckpt
    if isinstance(actor_or_ckpt, Actor)
    else _load_actor_from_checkpoint(actor_or_ckpt)
  )
  export_mod, actor_cpu = _actor_to_export_module(actor)
  out_path = Path(output_path)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  scripted = torch.jit.script(export_mod)
  scripted.save(str(out_path))
  if verify:
    verify_wbc_torchscript_parity(actor_cpu, out_path, batch_sizes=(1, 4), atol=atol)
  return out_path
