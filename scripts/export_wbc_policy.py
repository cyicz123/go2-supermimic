"""Export a trained Go2+Piper WBC checkpoint to a standalone 210D -> 18D TorchScript policy."""

from __future__ import annotations

import argparse
from pathlib import Path

from go2_supermimic.wbc.export import export_wbc_actor_torchscript, verify_wbc_torchscript_parity
from go2_supermimic.wbc.export import _load_actor_from_checkpoint


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--checkpoint", type=Path, required=True, help="Path to SuperMimic WBC model_*.pt")
  ap.add_argument("--output", type=Path, required=True, help="Destination TorchScript policy.pt path")
  ap.add_argument("--atol", type=float, default=1e-6, help="Parity error threshold vs eager Actor")
  args = ap.parse_args()

  out = export_wbc_actor_torchscript(args.checkpoint, args.output, verify=True, atol=args.atol)
  actor = _load_actor_from_checkpoint(args.checkpoint)
  max_err = verify_wbc_torchscript_parity(actor, out, batch_sizes=(1, 4), atol=args.atol)
  print(f"Exported TorchScript WBC policy to {out} (max_parity_err={max_err:.3e})")


if __name__ == "__main__":
  main()
