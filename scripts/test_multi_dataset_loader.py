"""Quick test script for multi-dataset prepared manifest loading and batching.

Creates two tiny prepared directories with a few .npz artifacts and manifest.parquet
then verifies that:
 - `PreparedForgeryDataset` can load the combined roots and resolve records
 - `Trainer` builds a DataLoader that uses `EvenMultiSourceBatchSampler` when
   multiple prepared roots contribute to the requested split

Run:
    python scripts/test_multi_dataset_loader.py

This script is lightweight and uses CPU only.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Adjust path to import project modules
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.data_preparation import PreparedForgeryDataset, EvenMultiSourceBatchSampler
from train import TrainConfig, Trainer
from model.hybrid_forgery_detector import HybridForgeryConfig


def make_simple_npz(path: Path, label: str, target_size: int = 384):
    path.parent.mkdir(parents=True, exist_ok=True)
    H = W = target_size
    img = (np.random.rand(H, W, 3).astype(np.float32)).astype(np.float32)
    mask = (np.random.rand(H, W, 1) > 0.9).astype(np.float32)
    meta = {
        "label": label,
        "split": "train",
        "source_image": str(path.name),
        "mask_path": None,
        "target_size": target_size,
        "variant_idx": 0,
        "variant_tag": "orig",
    }
    np.savez_compressed(path, image=img, mask=mask, meta=np.array(json.dumps(meta), dtype=object))


def write_manifest(root: Path, entries: list):
    df = pd.DataFrame(entries)
    manifest_path = root / "manifest.parquet"
    df.to_parquet(manifest_path, index=False)
    return manifest_path


def prepare_test_roots(base: Path):
    # create two prepared roots with small number of samples under train/real and train/fake
    roots = []
    for name in ("casia", "fantastic_reality"):
        root = base / name
        # clean if exists
        if root.exists():
            shutil.rmtree(root)
        # create 2 samples per root
        records = []
        for i in range(2):
            label = "fake" if i % 2 == 0 else "real"
            rel_dir = Path("train") / label
            filename = f"sample_{i}_{384}px_orig.npz"
            rel_path = str(rel_dir / filename)
            full_path = root / rel_path
            make_simple_npz(full_path, label, target_size=384)
            rec = {
                "split": "train",
                "target_size": 384,
                "relative_path": rel_path.replace("\\", "/"),
                "tar_path": None,
                "tar_member": filename,
                "label": label,
                "source_image": filename,
                "mask_path": None,
                "variant_idx": 0,
                "variant_tag": "orig",
            }
            records.append(rec)
        write_manifest(root, records)
        roots.append(root)
    return roots


def run_checks():
    base = Path("tmp_test_prepared")
    if base.exists():
        shutil.rmtree(base)
    base.mkdir(parents=True, exist_ok=True)

    roots = prepare_test_roots(base)
    print("Prepared test roots:")
    for r in roots:
        print(" -", r)

    # Test PreparedForgeryDataset with multiple roots
    ds = PreparedForgeryDataset(prepared_root=[str(roots[0]), str(roots[1])], split="train", target_size=384)
    assert len(ds) >= 4, "Expected at least 4 records across both roots"
    print("PreparedForgeryDataset loaded records:", len(ds))

    # Make sure every record has prepared_root field
    prepared_roots_in_records = {rec.get("prepared_root") for rec in ds.records}
    assert len(prepared_roots_in_records) == 2, f"Expected records from 2 prepared roots, got: {prepared_roots_in_records}"
    print("Record prepared_root values:", prepared_roots_in_records)

    # Build a Trainer that uses these prepared roots and check the train_loader
    cfg = TrainConfig()
    cfg.prepared_root = [str(roots[0]), str(roots[1])]
    cfg.target_size = 384
    cfg.batch_size = 4
    cfg.num_workers = 0
    cfg.device = "cpu"
    # Ensure no pretrained downloads
    cfg.model_config = HybridForgeryConfig(pretrained_backbones=False)

    trainer = Trainer(cfg)
    loader = trainer.train_loader

    # If multi-source sampler used, DataLoader should have batch_sampler set
    bs = getattr(loader, "batch_sampler", None)
    if bs is None:
        # In some PyTorch versions DataLoader may expose sampler instead
        bs = getattr(loader, "sampler", None)

    assert isinstance(bs, EvenMultiSourceBatchSampler), f"Expected EvenMultiSourceBatchSampler, got {type(bs)}"
    print("Trainer is using EvenMultiSourceBatchSampler for multi-source batching.")

    # Clean up
    shutil.rmtree(base)
    print("All checks passed.")


if __name__ == "__main__":
    run_checks()
