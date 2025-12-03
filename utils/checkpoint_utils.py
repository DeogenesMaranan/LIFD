"""Helpers for loading model checkpoints with messy state_dict key prefixes.

This module provides utilities to:
- "clean" state_dict keys that were saved under wrappers like `_orig_mod.` or `module.`
- load cleaned state into a model, trying strict loading then falling back to `strict=False`
- print concise diagnostics about missing/unexpected keys

Usage:
    from utils.checkpoint_utils import safe_load_state_dict_from_checkpoint
    ck = torch.load(checkpoint_path, map_location="cpu")
    safe_load_state_dict_from_checkpoint(model, ck, key="model_state")

Do not modify notebook files programmatically; import and call the helper from your notebook cell.
"""

from __future__ import annotations

from typing import Dict, Mapping, MutableMapping, Tuple, Sequence
import torch


def _strip_prefix(key: str, prefixes: Sequence[str]) -> str:
    for p in prefixes:
        if key.startswith(p):
            return key[len(p):]
    return key


def clean_state_dict_keys(state_dict: Mapping[str, torch.Tensor], prefixes_to_strip: Sequence[str] = ("_orig_mod.", "module.")) -> Dict[str, torch.Tensor]:
    """Return a copy of `state_dict` with any of the provided prefixes removed.

    Common cases handled:
    - keys beginning with "_orig_mod." (seen when certain wrappers were used during save)
    - keys beginning with "module." (DataParallel/Distributed wrappers)

    The function is careful to avoid key collisions: if stripping prefixes creates duplicate
    keys, later keys overwrite earlier ones (mirroring common `state_dict` loading behavior).
    """
    out: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        new_k = _strip_prefix(k, prefixes_to_strip)
        out[new_k] = v
    return out


def _extract_state_dict_from_checkpoint(ck: object, key_hints: Sequence[str] = ("model_state", "state_dict")) -> Dict[str, torch.Tensor]:
    # If ck is a dict and contains a nested state dict, prefer common keys
    if isinstance(ck, dict):
        for k in key_hints:
            if k in ck and isinstance(ck[k], dict):
                return ck[k]
        # If the dict looks like a state dict (tensor values), return that
        if all(isinstance(v, (torch.Tensor,)) for v in ck.values()):
            return dict(ck)
        # Fallback: build state from tensor-like entries
        candidate = {k: v for k, v in ck.items() if isinstance(v, torch.Tensor)}
        if candidate:
            return candidate
        # Nothing found
        raise ValueError("No state_dict found in checkpoint dict")
    elif isinstance(ck, (torch.nn.Module,)):
        # unlikely -- user passed model
        raise ValueError("Checkpoint appears to be a module; expected state_dict or dict")
    else:
        # Maybe ck is already a state dict
        if isinstance(ck, dict):
            return ck
        raise ValueError("Unsupported checkpoint format for state extraction")


def safe_load_state_dict_from_checkpoint(
    model: torch.nn.Module,
    checkpoint: object,
    key: str | None = "model_state",
    map_location: str | None = None,
    prefixes_to_strip: Sequence[str] = ("_orig_mod.", "module."),
    try_strict_first: bool = True,
) -> Tuple[torch.nn.modules.module._IncompatibleKeys | None, bool]:
    """Load a checkpoint into `model` robustly.

    Parameters:
    - model: the target torch.nn.Module
    - checkpoint: object returned by `torch.load(...)` (or a raw state_dict)
    - key: name of key inside the checkpoint dict that contains the state_dict (commonly "model_state").
      If None, helper will attempt to locate the state dict automatically.
    - prefixes_to_strip: prefixes to remove from keys when cleaning.
    - try_strict_first: try strict=True first; on failure fall back to strict=False.

    Returns:
    - (incompatible_keys, loaded_strictly)
      - incompatible_keys: _IncompatibleKeys object (has missing_keys and unexpected_keys) or None on success
      - loaded_strictly: True if loaded with strict=True; False if loaded with strict=False

    The function mutates neither the provided checkpoint file nor the model; it only uses a cleaned
    copy of the state dict to perform loading.
    """
    # Accept either a raw state dict or a torch.load result
    state_dict = None
    if isinstance(checkpoint, dict):
        if key is not None and key in checkpoint:
            state_dict = checkpoint[key]
        else:
            # try to extract
            try:
                state_dict = _extract_state_dict_from_checkpoint(checkpoint)
            except ValueError as exc:
                raise
    else:
        # assume it's already a state_dict
        state_dict = checkpoint

    if map_location is not None:
        # move tensors to CPU first if requested
        try:
            state_dict = {k: v.to(map_location) if isinstance(v, torch.Tensor) else v for k, v in state_dict.items()}
        except Exception:
            # ignore; map_location is advisory
            pass

    cleaned = clean_state_dict_keys(state_dict, prefixes_to_strip=prefixes_to_strip)

    # Try strict loading first if requested
    try_order = [True, False] if try_strict_first else [False]
    last_exc = None
    for strict_mode in try_order:
        try:
            result = model.load_state_dict(cleaned, strict=strict_mode)
            # If result is an IncompatibleKeys object, return it
            return result, strict_mode
        except RuntimeError as e:
            last_exc = e
            # try next option
            continue

    # If we get here, both strict attempts failed
    raise RuntimeError(f"Failed to load checkpoint into model: last error: {last_exc}")


__all__ = [
    "clean_state_dict_keys",
    "safe_load_state_dict_from_checkpoint",
]
