from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

    conf_counts = torch.zeros(2, 2, dtype=torch.double, device=resolved_device)
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
            preds = (probs > threshold).float()

            dice, iou = _segmentation_metrics(preds, masks)
            precision, recall = _precision_recall(preds, masks)
            conf_counts += _confusion_counts(preds, masks)

            total_loss += loss.item()
            total_dice += dice
            total_iou += iou
            total_precision += precision
            total_recall += recall
            batches += 1

            if max_batches is not None and (batch_idx + 1) >= max_batches:
                break

    if batches == 0:
        raise RuntimeError(f"No batches evaluated for split '{split}'. Ensure the prepared data exists.")

    metrics = {
        "loss": total_loss / batches,
        "dice": total_dice / batches,
        "iou": total_iou / batches,
        "precision": total_precision / batches,
        "recall": total_recall / batches,
        "f1": _f1(total_precision / batches, total_recall / batches),
    }

    confusion = conf_counts.cpu().numpy()
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
