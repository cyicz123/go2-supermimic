'''Tests for WbcTrainer._collapse_guard inference-tensor buffer rollback.'''

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from go2_supermimic.supermimic.train import Actor, Critic
from go2_supermimic.wbc.trainer import WbcTrainer


def _make_dummy_trainer(ckpt_path: Path) -> WbcTrainer:
  trainer = WbcTrainer.__new__(WbcTrainer)
  trainer.dev = torch.device('cpu')
  trainer.cfg = SimpleNamespace(collapse_falls=3.0, lr=2e-4)
  trainer.it = 600
  trainer.actor_frozen_until = 0
  trainer.collapse_strikes = 1
  trainer.lr_scale = 1.0
  trainer.ramp_start = 0
  trainer.last_ckpt = ckpt_path
  trainer.actor = Actor(obs_dim=16, act_dim=6, hidden=(32,))
  trainer.critic = Critic(in_dim=16, hidden=(32,))
  trainer.opt_actor = torch.optim.Adam(trainer.actor.parameters(), lr=2e-4, betas=(0.9, 0.95))
  trainer.opt_critic = torch.optim.Adam(trainer.critic.parameters(), lr=2e-4)
  return trainer


def test_collapse_guard_rollback_after_inference_mode_updates(tmp_path: Path) -> None:
  ckpt_path = tmp_path / 'model_500.pt'
  trainer = _make_dummy_trainer(ckpt_path)

  saved_actor = {k: v.detach().clone() for k, v in trainer.actor.state_dict().items()}
  saved_critic = {k: v.detach().clone() for k, v in trainer.critic.state_dict().items()}
  torch.save({'actor': saved_actor, 'critic': saved_critic}, ckpt_path)

  with torch.no_grad():
    for p in trainer.actor.parameters():
      p.add_(1.0)
    for p in trainer.critic.parameters():
      p.add_(1.0)

  with torch.inference_mode():
    trainer.actor.norm.update(torch.randn(8, 16) + 3.0)
    trainer.critic.norm.update(torch.randn(8, 16) - 2.0)

  assert trainer.actor.norm.var.is_inference()
  assert trainer.actor.norm.count.is_inference()
  assert trainer.critic.norm.var.is_inference()
  assert trainer.critic.norm.count.is_inference()

  trainer._collapse_guard(falls=5.0)

  assert not trainer.actor.norm.var.is_inference()
  assert not trainer.actor.norm.count.is_inference()
  assert not trainer.critic.norm.var.is_inference()
  assert not trainer.critic.norm.count.is_inference()
  assert trainer.lr_scale == 0.5
  assert trainer.collapse_strikes == 0
  assert trainer.ramp_start == 600

  for k, v in trainer.actor.state_dict().items():
    assert torch.allclose(v, saved_actor[k])
  for k, v in trainer.critic.state_dict().items():
    assert torch.allclose(v, saved_critic[k])

  with torch.inference_mode():
    trainer.actor.norm.update(torch.randn(8, 16))
    trainer.critic.norm.update(torch.randn(8, 16))
  trainer.collapse_strikes = 1
  trainer.it = 700
  trainer._collapse_guard(falls=4.0)
  assert trainer.lr_scale == 0.25
  assert not trainer.actor.norm.var.is_inference()
  assert not trainer.critic.norm.var.is_inference()
