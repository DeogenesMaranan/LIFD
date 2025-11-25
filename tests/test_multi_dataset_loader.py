import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd

from data.data_preparation import PreparedForgeryDataset, EvenMultiSourceBatchSampler, EvenMultiSourceBalancedSampler
from train import TrainConfig, Trainer
from model.hybrid_forgery_detector import HybridForgeryConfig


def _make_simple_npz(path: Path, label: str, target_size: int = 384):
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


def _write_manifest(root: Path, entries: list):
    df = pd.DataFrame(entries)
    manifest_path = root / "manifest.parquet"
    df.to_parquet(manifest_path, index=False)
    return manifest_path


def _prepare_test_roots(base: Path):
    roots = []
    for name in ("casia", "fantastic_reality"):
        root = base / name
        if root.exists():
            shutil.rmtree(root)
        records = []
        for i in range(2):
            label = "fake" if i % 2 == 0 else "real"
            rel_dir = Path("train") / label
            filename = f"sample_{i}_{384}px_orig.npz"
            rel_path = str(rel_dir / filename)
            full_path = root / rel_path
            _make_simple_npz(full_path, label, target_size=384)
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
        _write_manifest(root, records)
        roots.append(root)
    return roots


def test_multi_dataset_loader_and_trainer(tmp_path):
    base = tmp_path / "tmp_test_prepared"
    roots = _prepare_test_roots(base)

    # PreparedForgeryDataset should load records from both roots
    ds = PreparedForgeryDataset(prepared_root=[str(roots[0]), str(roots[1])], split="train", target_size=384)
    assert len(ds) >= 4
    prepared_roots_in_records = {rec.get("prepared_root") for rec in ds.records}
    assert len(prepared_roots_in_records) == 2

    # Trainer should use EvenMultiSourceBatchSampler when multiple prepared roots present
    cfg = TrainConfig()
    cfg.prepared_root = [str(roots[0]), str(roots[1])]
    cfg.target_size = 384
    cfg.batch_size = 4
    cfg.num_workers = 0
    cfg.device = "cpu"
    cfg.model_config = HybridForgeryConfig(pretrained_backbones=False)

    trainer = Trainer(cfg)
    loader = trainer.train_loader

    # DataLoader should be using a batch_sampler of one of our multi-source classes
    batch_sampler = getattr(loader, "batch_sampler", None)
    assert isinstance(batch_sampler, (EvenMultiSourceBatchSampler, EvenMultiSourceBalancedSampler))

    # Iterate one epoch worth of batches to ensure sampler yields valid indices
    batches = list(iter(loader))
    assert len(batches) >= 1

    # cleanup
    shutil.rmtree(base)
