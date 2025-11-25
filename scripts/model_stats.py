"""Utility to compute model parameter counts and approximate FLOPs.

Usage examples:
  python -m scripts.model_stats --input-size 3 384 384
  python -m scripts.model_stats --checkpoint path/to/checkpoint.pt --input-size 3 384 384

The script will try to use `thop` first, then `ptflops`.
If neither is available it will still print parameter counts and advise installing one of them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch


def compute_param_count(model: torch.nn.Module) -> Tuple[int, int]:
    """Return (total_params, trainable_params) as integers."""
    total = 0
    trainable = 0
    for p in model.parameters():
        n = int(p.numel())
        total += n
        if p.requires_grad:
            trainable += n
    return total, trainable


def compute_flops(model: torch.nn.Module, input_size: Tuple[int, ...], device: str = "cpu") -> Optional[float]:
    """Try to compute FLOPs (in GFLOPs) using common libraries.

    Returns GFLOPs (float) or None if unavailable.
    """
    # Try thop
    try:
        from thop import profile

        model_eval = model.to(device)
        model_eval.eval()
        with torch.no_grad():
            dummy = torch.randn((1, *input_size), device=device)
            macs, params = profile(model_eval, inputs=(dummy,), verbose=False)
        # thop returns MACs or FLOPs depending on version; treat as FLOPs/MACs
        gflops = float(macs) / 1e9
        return gflops
    except Exception:
        pass

    # Try ptflops
    try:
        from ptflops import get_model_complexity_info

        # get_model_complexity_info runs on CPU by default; move model to CPU
        model_cpu = model.cpu()
        flops, params = get_model_complexity_info(model_cpu, tuple(input_size), as_strings=False, print_per_layer_stat=False)
        # ptflops returns flops as float (FLOPs)
        gflops = float(flops) / 1e9
        return gflops
    except Exception:
        pass

    return None


def build_model_from_checkpoint(checkpoint_path: Path, device: str = "cpu") -> torch.nn.Module:
    """Attempt to construct the model from a checkpoint."""
    from model.hybrid_forgery_detector import HybridForgeryConfig, HybridForgeryDetector

    ck = torch.load(checkpoint_path, map_location="cpu")
    # Expecting either a dict containing a saved config and model_state, or a raw state_dict
    model = None
    if isinstance(ck, dict):
        # Try to locate model_config
        model_cfg = None
        if "config" in ck and isinstance(ck["config"], dict):
            mc = ck["config"].get("model_config") or ck["config"].get("model")
            if mc:
                try:
                    model_cfg = HybridForgeryConfig(**mc)
                except Exception:
                    model_cfg = None
        # Fallback to default config
        if model_cfg is None:
            model_cfg = HybridForgeryConfig()
        model = HybridForgeryDetector(model_cfg)
        # Load weights if present
        state = None
        if "model_state" in ck:
            state = ck["model_state"]
        elif "state_dict" in ck:
            state = ck["state_dict"]
        else:
            # maybe the dict itself is a state dict
            state = {k: v for k, v in ck.items() if isinstance(v, torch.Tensor)}
        if state:
            try:
                model.load_state_dict(state)
            except Exception:
                # try strict=False
                model.load_state_dict(state, strict=False)
    else:
        # assume it's a bare state_dict
        model_cfg = HybridForgeryConfig()
        model = HybridForgeryDetector(model_cfg)
        try:
            model.load_state_dict(ck)
        except Exception:
            model.load_state_dict(ck, strict=False)

    return model.to(device)


def build_default_model(device: str = "cpu") -> torch.nn.Module:
    from model.hybrid_forgery_detector import HybridForgeryConfig, HybridForgeryDetector

    cfg = HybridForgeryConfig()
    model = HybridForgeryDetector(cfg)
    return model.to(device)


def _parse_input_size(args_input) -> Tuple[int, ...]:
    if not args_input:
        return (3, 384, 384)
    values = tuple(int(x) for x in args_input)
    return values


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Compute model parameters and FLOPs for the HybridForgeryDetector")
    p.add_argument("--checkpoint", type=str, help="Optional checkpoint path to load model weights from")
    p.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--input-size", nargs="*", help="Input channel and spatial size, e.g. --input-size 3 384 384")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    device = args.device
    input_size = _parse_input_size(args.input_size)

    if args.checkpoint:
        ck_path = Path(args.checkpoint)
        if not ck_path.exists():
            print(f"Checkpoint not found: {ck_path}")
            return 2
        model = build_model_from_checkpoint(ck_path, device=device)
    else:
        model = build_default_model(device=device)

    total, trainable = compute_param_count(model)
    print(f"Total parameters : {total:,}")
    print(f"Trainable params : {trainable:,}")

    flops = compute_flops(model, input_size, device=device)
    if flops is None:
        print("Could not compute FLOPs: install 'thop' (pip install thop) or 'ptflops' (pip install ptflops).")
    else:
        print(f"Approx. FLOPs     : {flops:.3f} GFLOPs (input {input_size})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
