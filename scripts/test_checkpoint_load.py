"""Test loader for checkpoints in the `checkpoints/` directory.

For each checkpoint (.pt/.pth) the script will:
- load it with `torch.load(..., map_location='cpu')`
- attempt to construct a `HybridForgeryDetector` using a config found in the checkpoint (fallback to default)
- use `safe_load_state_dict_from_checkpoint` from `utils.checkpoint_utils` to load the weights
- print concise diagnostics (loaded strict, counts and sample missing/unexpected keys)

Run from project root: `python scripts/test_checkpoint_load.py`
"""
from __future__ import annotations

import traceback
from pathlib import Path
import torch

from model.hybrid_forgery_detector import HybridForgeryConfig, HybridForgeryDetector
from utils.checkpoint_utils import safe_load_state_dict_from_checkpoint

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"


def try_build_config_from_ck(ck: object) -> HybridForgeryConfig:
    # mirror logic from model_stats.build_model_from_checkpoint to extract config
    if isinstance(ck, dict):
        if "config" in ck and isinstance(ck["config"], dict):
            mc = ck["config"].get("model_config") or ck["config"].get("model")
            if mc and isinstance(mc, dict):
                try:
                    return HybridForgeryConfig(**mc)
                except Exception:
                    pass
    return HybridForgeryConfig()


def main() -> int:
    if not CHECKPOINT_DIR.exists():
        print(f"Checkpoint directory not found: {CHECKPOINT_DIR}")
        return 2

    ck_files = sorted([p for p in CHECKPOINT_DIR.iterdir() if p.suffix in (".pt", ".pth")])
    if not ck_files:
        print("No checkpoint files found in", CHECKPOINT_DIR)
        return 1

    for ck_path in ck_files:
        print("\n===", ck_path.name, "===")
        try:
            ck = torch.load(ck_path, map_location="cpu")
        except Exception as e:
            print("Failed to load checkpoint file:")
            traceback.print_exc()
            continue

        cfg = try_build_config_from_ck(ck)
        model = HybridForgeryDetector(cfg)

        try:
            incompat, loaded_strictly = safe_load_state_dict_from_checkpoint(
                model, ck, key="model_state", map_location="cpu"
            )
            print(f"Loaded strict: {loaded_strictly}")
            if incompat is not None:
                print(f"Missing keys ({len(incompat.missing_keys)}): {incompat.missing_keys[:8]}")
                print(f"Unexpected keys ({len(incompat.unexpected_keys)}): {incompat.unexpected_keys[:8]}")
            else:
                print("No incompatible keys reported (full match)")
        except Exception as e:
            print("Error while trying to load into model:")
            traceback.print_exc()
            continue

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
