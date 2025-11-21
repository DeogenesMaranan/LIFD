from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from data.data_preparation import PreparedForgeryDataset
from model.hybrid_forgery_detector import HybridForgeryConfig, HybridForgeryDetector
from train import TrainConfig

try:  # tqdm is optional
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


@dataclass
class EvaluationSummary:
    metrics: Dict[str, float]
    confusion_matrix: np.ndarray
    samples: int


def load_model_from_checkpoint(checkpoint_path: str | Path, device: Optional[str] = None) -> Tuple[HybridForgeryDetector, TrainConfig]:
    """Load a trained model + configuration tuple from a checkpoint file."""

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint '{checkpoint_path}' does not exist")

    resolved_device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=resolved_device)
    config_dict = dict(checkpoint["config"])
    model_cfg_dict = config_dict.pop("model_config", {})
    model_config = HybridForgeryConfig(**model_cfg_dict)
    train_config = TrainConfig(**config_dict)
    train_config.model_config = model_config

    model = HybridForgeryDetector(model_config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(resolved_device)
    model.eval()
    return model, train_config


def evaluate_split(
    model: HybridForgeryDetector,
    train_config: TrainConfig,
    split: str = "test",
    batch_size: Optional[int] = None,
    device: Optional[torch.device | str] = None,
    max_batches: Optional[int] = None,
    threshold: float = 0.5,
    auto_threshold: bool = False,
    threshold_candidates: Optional[Sequence[float]] = None,
    threshold_metric: str = "f1",
) -> EvaluationSummary:
    """Compute pixel-wise metrics and confusion matrix on a given split."""

    resolved_device = _resolve_device(device or train_config.device)
    dataloader = _build_dataloader(train_config, split, batch_size, resolved_device)
    criterion = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_precision = 0.0
    total_recall = 0.0
    batches = 0

    if auto_threshold:
        candidate_thresholds = sorted({float(thr) for thr in (threshold_candidates or train_config.eval_thresholds or [])})
        if not candidate_thresholds:
            candidate_thresholds = [threshold]
    else:
        candidate_thresholds = [threshold]
    threshold_stats = {
        float(thr): {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0}
        for thr in candidate_thresholds
    }
    iterator = dataloader
    if tqdm:
        iterator = tqdm(iterator, desc=f"Evaluating {split}", leave=False)

    model.eval()
    with torch.inference_mode():
        for batch_idx, batch in enumerate(iterator):
            batch = _prepare_batch(batch, resolved_device)
            images, masks = batch["image"], batch["mask"]
            logits = model(images)
            if logits.shape[-2:] != masks.shape[-2:]:
                logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            loss = criterion(logits, masks)
            probs = torch.sigmoid(logits)
            for thr in candidate_thresholds:
                preds = (probs > thr).float()
                stats = threshold_stats[thr]
                tp = (preds * masks).sum().double().item()
                fp = (preds * (1 - masks)).sum().double().item()
                fn = ((1 - preds) * masks).sum().double().item()
                tn = ((1 - preds) * (1 - masks)).sum().double().item()
                stats["tp"] += tp
                stats["fp"] += fp
                stats["fn"] += fn
                stats["tn"] += tn
                if thr == candidate_thresholds[0]:  # accumulate loss-scale metrics once per loop
                    dice, iou = _segmentation_metrics(preds, masks)
                    precision, recall = _precision_recall(preds, masks)
                    total_dice += dice
                    total_iou += iou
                    total_precision += precision
                    total_recall += recall
            total_loss += loss.item()
            batches += 1

            if max_batches is not None and (batch_idx + 1) >= max_batches:
                break

    if batches == 0:
        raise RuntimeError(f"No batches evaluated for split '{split}'. Ensure the prepared data exists.")

    metrics_by_threshold, best_info = _finalize_thresholds(threshold_stats)
    if auto_threshold:
        selected_threshold = _select_best_threshold(metrics_by_threshold, threshold_metric)
    else:
        selected_threshold = float(threshold)
    selected_metrics = metrics_by_threshold.get(selected_threshold, {})

    metrics = {
        "loss": total_loss / batches,
        "dice": selected_metrics.get("dice", total_dice / batches),
        "iou": selected_metrics.get("iou", total_iou / batches),
        "precision": selected_metrics.get("precision", total_precision / batches),
        "recall": selected_metrics.get("recall", total_recall / batches),
        "f1": selected_metrics.get(
            "f1", _f1(total_precision / batches, total_recall / batches)
        ),
        "threshold": selected_threshold,
        "thresholds": metrics_by_threshold,
        "best_threshold": best_info,
    }

    confusion = _stats_to_confusion(metrics_by_threshold.get(selected_threshold, {}))
    return EvaluationSummary(metrics=metrics, confusion_matrix=confusion, samples=batches)


def collect_visual_samples(
    model: HybridForgeryDetector,
    train_config: TrainConfig,
    split: str = "test",
    num_samples: int = 10,
    device: Optional[torch.device | str] = None,
    threshold: float = 0.5,
) -> List[Dict[str, Image.Image]]:
    """Return a list of PIL images for qualitative inspection (image/gt/pred/overlay)."""

    resolved_device = _resolve_device(device or train_config.device)
    dataset = PreparedForgeryDataset(
        prepared_root=train_config.prepared_root,
        split=split,
        target_size=train_config.target_size,
        include_features=False,
        return_masks=True,
    )
    indices = list(range(len(dataset)))
    rng = np.random.default_rng(seed=0)
    rng.shuffle(indices)

    samples: List[Dict[str, Image.Image]] = []
    model.eval()
    with torch.inference_mode():
        for idx in indices:
            sample = dataset[idx]
            image = sample["image"].unsqueeze(0).to(resolved_device)
            mask_tensor = sample.get("mask")
            if mask_tensor is None:
                continue  # skip entries without masks so GT column is meaningful
            mask_tensor = mask_tensor.to(resolved_device)
            logits = model(image)
            if logits.shape[-2:] != mask_tensor.shape[-2:]:
                logits = F.interpolate(logits, size=mask_tensor.shape[-2:], mode="bilinear", align_corners=False)
            probs = torch.sigmoid(logits)
            pred = (probs > threshold).double()

            sample_entry = {
                "image": _tensor_to_pil(image[0]),
                "ground_truth": _mask_to_pil(mask_tensor[0]),
                "prediction": _mask_to_pil(pred[0]),
            }
            sample_entry["overlay"] = _overlay_prediction(sample_entry["image"], sample_entry["prediction"])
            samples.append(sample_entry)
            if len(samples) >= num_samples:
                break

    if len(samples) < num_samples:
        print(f"Warning: only collected {len(samples)} samples with masks in split='{split}'.")
    return samples


def _build_dataloader(
    train_config: TrainConfig,
    split: str,
    batch_size: Optional[int],
    device: torch.device,
) -> DataLoader:
    dataset = PreparedForgeryDataset(
        prepared_root=train_config.prepared_root,
        split=split,
        target_size=train_config.target_size,
        include_features=False,
        return_masks=True,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size or train_config.batch_size,
        shuffle=False,
        num_workers=train_config.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=_collate_batch,
    )


def _collate_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    images = torch.stack([item["image"].float() for item in batch])
    masks = []
    for item in batch:
        mask = item.get("mask")
        if mask is None:
            mask = torch.zeros(1, images.shape[-2], images.shape[-1], dtype=torch.float32)
        else:
            mask = mask.float()
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
        masks.append(mask)
    return {"image": images, "mask": torch.stack(masks)}


def _prepare_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def _segmentation_metrics(preds: torch.Tensor, targets: torch.Tensor) -> Tuple[float, float]:
    eps = 1e-6
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    preds_sum = preds.sum(dim=(1, 2, 3))
    targets_sum = targets.sum(dim=(1, 2, 3))
    dice = (2 * intersection + eps) / (preds_sum + targets_sum + eps)
    union = preds_sum + targets_sum - intersection
    iou = (intersection + eps) / (union + eps)
    return dice.mean().item(), iou.mean().item()


def _precision_recall(preds: torch.Tensor, targets: torch.Tensor) -> Tuple[float, float]:
    eps = 1e-6
    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()
    fn = ((1 - preds) * targets).sum()
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    return precision.item(), recall.item()


def _confusion_counts(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)
    tp = (preds_flat * targets_flat).sum()
    fp = (preds_flat * (1 - targets_flat)).sum()
    fn = ((1 - preds_flat) * targets_flat).sum()
    tn = ((1 - preds_flat) * (1 - targets_flat)).sum()
    return torch.tensor([[tn, fp], [fn, tp]], dtype=torch.double, device=preds.device)


def _finalize_thresholds(stats: Dict[float, Dict[str, float]]) -> tuple[Dict[float, Dict[str, float]], Dict[str, float]]:
    eps = 1e-8
    metrics: Dict[float, Dict[str, float]] = {}
    best_metric = -math.inf
    best_thr = None
    for threshold, counts in stats.items():
        tp = counts["tp"]
        fp = counts["fp"]
        fn = counts["fn"]
        tn = counts["tn"]
        precision = tp / (tp + fp + eps) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn + eps) if (tp + fn) > 0 else 0.0
        dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
        iou = (tp + eps) / (tp + fp + fn + eps)
        f1 = (2 * precision * recall / (precision + recall + eps)) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / (tp + fp + fn + tn + eps)
        metrics[threshold] = {
            "dice": dice,
            "iou": iou,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        }
        if f1 > best_metric:
            best_metric = f1
            best_thr = threshold
    best_summary = {"value": best_thr if best_thr is not None else float("nan")}
    if best_thr is not None:
        best_summary.update(metrics[best_thr])
    return metrics, best_summary


def _select_best_threshold(metrics: Dict[float, Dict[str, float]], metric: str) -> float:
    metric = metric.lower()
    best_value = -math.inf
    best_threshold = next(iter(metrics.keys()), 0.5)
    for threshold, vals in metrics.items():
        score = vals.get(metric, vals.get("f1", 0.0))
        if score > best_value:
            best_value = score
            best_threshold = threshold
    return best_threshold


def _stats_to_confusion(entry: Dict[str, float]) -> np.ndarray:
    confusion = entry.get("confusion") if isinstance(entry, dict) else None
    if not confusion:
        return np.zeros((2, 2), dtype=float)
    return np.array(
        [[confusion.get("tn", 0.0), confusion.get("fp", 0.0)], [confusion.get("fn", 0.0), confusion.get("tp", 0.0)]]
    )


def _f1(precision: float, recall: float) -> float:
    eps = 1e-6
    return (2 * precision * recall + eps) / (precision + recall + eps)


def _resolve_device(device_like: Optional[torch.device | str]) -> torch.device:
    if device_like is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_like)


def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    arr = tensor.detach().cpu().clamp(0, 1).numpy()
    arr = np.transpose(arr, (1, 2, 0))
    arr = (arr * 255.0).astype(np.uint8)
    return Image.fromarray(arr)


def _mask_to_pil(tensor: torch.Tensor) -> Image.Image:
    arr = tensor.detach().cpu().clamp(0, 1).squeeze().numpy()
    arr = (arr * 255.0).astype(np.uint8)
    return Image.fromarray(arr)


def _overlay_prediction(image: Image.Image, prediction_mask: Image.Image, alpha: float = 0.4) -> Image.Image:
    image_rgb = image.convert("RGB")
    image_arr = np.array(image_rgb).astype(np.float32)
    pred_arr = np.array(prediction_mask.convert("L")) / 255.0
    pred_arr = pred_arr[..., None]

    overlay_color = np.zeros_like(image_arr)
    overlay_color[..., 0] = 255  # red overlay for predicted tampered regions

    blended = (1 - alpha * pred_arr) * image_arr + alpha * overlay_color * pred_arr
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)
