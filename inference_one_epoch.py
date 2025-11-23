"""Utility script to run inference from a saved checkpoint (dataset or single image)."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFilter, ImageOps
from torch.utils.data import DataLoader

from data.data_preparation import PreparedForgeryDataset
from evaluation.eval_utils import load_model_from_checkpoint

try:  # tqdm is optional
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

try:  # Pillow 9+ exposes enums via Image.Resampling
    RESAMPLE_BICUBIC = Image.Resampling.BICUBIC
    RESAMPLE_NEAREST = Image.Resampling.NEAREST
except AttributeError:  # pragma: no cover
    RESAMPLE_BICUBIC = Image.BICUBIC
    RESAMPLE_NEAREST = Image.NEAREST


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference over a dataset split or a single raw image.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt", help="Path to the checkpoint file")
    parser.add_argument("--split", type=str, default="val", help="Prepared split to evaluate (train/val/test)")
    parser.add_argument("--prepared-root", type=str, default=None, help="Override prepared data root if needed")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size for inference")
    parser.add_argument("--num-workers", type=int, default=None, help="Override dataloader worker count")
    parser.add_argument("--device", type=str, default=None, help="Torch device identifier, e.g. cuda:0 or cpu")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for predictions")
    parser.add_argument("--max-batches", type=int, default=None, help="Limit number of batches for quick smoke tests")
    parser.add_argument("--progress", action="store_true", help="Show tqdm progress bar if available")
    parser.add_argument("--image", type=str, default=None, help="Path to a single RGB image for inference")
    parser.add_argument("--mask", type=str, default=None, help="Optional ground-truth mask aligned with --image")
    parser.add_argument("--save-mask", type=str, default=None, help="If set, save the predicted mask to this path")
    parser.add_argument(
        "--gaussian-radius",
        type=float,
        default=1.0,
        help="Gaussian blur radius for noise features when --image is used",
    )
    return parser.parse_args()


def should_include_features(train_config) -> bool:
    explicit = getattr(train_config, "include_aux_features", None)
    if explicit is not None:
        return bool(explicit)
    model_cfg = getattr(train_config, "model_config", None)
    return bool(getattr(model_cfg, "use_noise_branch", False))


def build_dataloader(train_config, split: str, batch_size: int, num_workers: int, include_features: bool, device: torch.device) -> DataLoader:
    dataset = PreparedForgeryDataset(
        prepared_root=train_config.prepared_root,
        split=split,
        target_size=train_config.target_size,
        include_features=include_features,
        return_masks=True,
    )
    pin_memory = device.type == "cuda"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_collate_batch,
    )


def _collate_batch(batch: list[Dict[str, Any]]) -> Dict[str, Any]:
    images = torch.stack([item["image"].float() for item in batch])
    collated: Dict[str, Any] = {"image": images}

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
    collated["mask"] = torch.stack(masks)

    for key in ("high_pass", "residual"):
        if all(key in sample for sample in batch):
            collated[key] = torch.stack([sample[key].float() for sample in batch])

    if all("label" in sample for sample in batch):
        collated["label"] = torch.stack([sample["label"].long() for sample in batch])
    if all("meta" in sample for sample in batch):
        collated["meta"] = [sample["meta"] for sample in batch]
    return collated


def _prepare_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    prepared: Dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            prepared[key] = value.to(device, non_blocking=True)
        else:
            prepared[key] = value
    return prepared


def _extract_noise_inputs(batch: Dict[str, Any]) -> Optional[Dict[str, torch.Tensor]]:
    noise: Dict[str, torch.Tensor] = {}
    for key in ("residual", "high_pass"):
        tensor = batch.get(key)
        if torch.is_tensor(tensor):
            noise[key] = tensor
    return noise or None


def _segmentation_metrics(preds: torch.Tensor, targets: torch.Tensor) -> tuple[float, float]:
    eps = 1e-6
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    preds_sum = preds.sum(dim=(1, 2, 3))
    targets_sum = targets.sum(dim=(1, 2, 3))
    dice = (2 * intersection + eps) / (preds_sum + targets_sum + eps)
    union = preds_sum + targets_sum - intersection
    iou = (intersection + eps) / (union + eps)
    return dice.mean().item(), iou.mean().item()


def _precision_recall(preds: torch.Tensor, targets: torch.Tensor) -> tuple[float, float]:
    eps = 1e-6
    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()
    fn = ((1 - preds) * targets).sum()
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    return precision.item(), recall.item()


def _load_image_tensor(image_path: Path, target_size: int) -> tuple[torch.Tensor, Image.Image]:
    pil_image = Image.open(image_path).convert("RGB")
    resized = ImageOps.fit(pil_image, (target_size, target_size), method=RESAMPLE_BICUBIC)
    array = np.array(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array.transpose(2, 0, 1)).float()
    return tensor, resized


def _load_mask_tensor(mask_path: Path, target_size: int) -> torch.Tensor:
    pil_mask = Image.open(mask_path).convert("L")
    resized = ImageOps.fit(pil_mask, (target_size, target_size), method=RESAMPLE_NEAREST)
    array = np.array(resized, dtype=np.float32) / 255.0
    if array.ndim == 2:
        array = array[None, ...]
    return torch.from_numpy(array).float()


def _compute_noise_tensors(rgb_image: Image.Image, gaussian_radius: float) -> tuple[torch.Tensor, torch.Tensor]:
    image_uint8 = np.array(rgb_image, dtype=np.uint8)
    image_float = image_uint8.astype(np.float32)
    blur = rgb_image.filter(ImageFilter.GaussianBlur(radius=max(0.1, gaussian_radius)))
    blur_arr = np.array(blur, dtype=np.float32)
    residual = (image_float - blur_arr) / 255.0
    high_pass = np.abs(residual)
    high_pass = np.clip(high_pass, 0.0, 1.0)
    residual_tensor = torch.from_numpy(residual.transpose(2, 0, 1)).float()
    high_pass_tensor = torch.from_numpy(high_pass.transpose(2, 0, 1)).float()
    return high_pass_tensor, residual_tensor


def _save_mask_image(tensor: torch.Tensor, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    array = tensor.squeeze().detach().cpu().clamp(0, 1).numpy()
    image = Image.fromarray((array * 255.0).astype(np.uint8))
    image.save(target_path)


def run_single_image(
    model: torch.nn.Module,
    image_path: Path,
    mask_path: Optional[Path],
    device: torch.device,
    train_config,
    threshold: float,
    include_features: bool,
    gaussian_radius: float,
) -> Dict[str, float | torch.Tensor]:
    image_tensor, resized_image = _load_image_tensor(image_path, train_config.target_size)
    image_batch = image_tensor.unsqueeze(0).to(device)

    noise_inputs: Optional[Dict[str, torch.Tensor]] = None
    if include_features and getattr(train_config.model_config, "use_noise_branch", False):
        high_pass_tensor, residual_tensor = _compute_noise_tensors(resized_image, gaussian_radius)
        noise_inputs = {}
        if getattr(train_config.model_config, "noise_branch_use_high_pass", True):
            noise_inputs["high_pass"] = high_pass_tensor.unsqueeze(0).to(device)
        if getattr(train_config.model_config, "noise_branch_use_residual", True):
            noise_inputs["residual"] = residual_tensor.unsqueeze(0).to(device)
        if not noise_inputs:
            noise_inputs = None

    model.eval()
    with torch.inference_mode():
        logits = model(image_batch, noise_features=noise_inputs)
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()

    result: Dict[str, float | torch.Tensor] = {
        "prob_mean": probs.mean().item(),
        "prob_max": probs.max().item(),
        "prob_min": probs.min().item(),
        "prediction": preds.squeeze().detach().cpu(),
    }

    if mask_path is None:
        return result

    mask_tensor = _load_mask_tensor(mask_path, train_config.target_size)
    mask_batch = mask_tensor.unsqueeze(0).to(device)
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(logits, mask_batch)
    dice, iou = _segmentation_metrics(preds, mask_batch)
    precision, recall = _precision_recall(preds, mask_batch)
    f1 = (2 * precision * recall) / max(precision + recall, 1e-6)
    result.update(
        {
            "loss": loss.item(),
            "dice": dice,
            "iou": iou,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )
    return result


def run_single_epoch(model: torch.nn.Module, dataloader: DataLoader, device: torch.device, threshold: float, max_batches: Optional[int], show_progress: bool) -> Dict[str, float]:
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_precision = 0.0
    total_recall = 0.0
    batches = 0

    iterator = dataloader
    if show_progress and tqdm is not None:
        iterator = tqdm(iterator, desc="Running inference", leave=False)

    model.eval()
    with torch.inference_mode():
        for batch_idx, batch in enumerate(iterator, start=1):
            batch = _prepare_batch(batch, device)
            images = batch["image"]
            masks = batch["mask"]
            noise_inputs = _extract_noise_inputs(batch)
            logits = model(images, noise_features=noise_inputs)
            if logits.shape[-2:] != masks.shape[-2:]:
                logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            loss = criterion(logits, masks)
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()
            dice, iou = _segmentation_metrics(preds, masks)
            precision, recall = _precision_recall(preds, masks)

            total_loss += loss.item()
            total_dice += dice
            total_iou += iou
            total_precision += precision
            total_recall += recall
            batches += 1

            if max_batches is not None and batch_idx >= max_batches:
                break

    if batches == 0:
        raise RuntimeError("No batches were processed. Check that the prepared dataset exists.")

    avg_precision = total_precision / batches
    avg_recall = total_recall / batches
    f1 = (2 * avg_precision * avg_recall) / max(avg_precision + avg_recall, 1e-6)
    return {
        "loss": total_loss / batches,
        "dice": total_dice / batches,
        "iou": total_iou / batches,
        "precision": avg_precision,
        "recall": avg_recall,
        "f1": f1,
        "batches": batches,
    }


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint '{checkpoint_path}' does not exist")

    model, train_config = load_model_from_checkpoint(checkpoint_path, device=args.device)

    if args.prepared_root:
        train_config.prepared_root = args.prepared_root
    if args.batch_size is not None:
        train_config.batch_size = args.batch_size
    if args.num_workers is not None:
        train_config.num_workers = args.num_workers

    device = torch.device(args.device) if args.device else train_config.resolved_device()
    model.to(device)

    include_features = should_include_features(train_config)

    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            raise FileNotFoundError(f"Image '{image_path}' not found")
        mask_path = Path(args.mask) if args.mask else None
        if mask_path and not mask_path.exists():
            raise FileNotFoundError(f"Mask '{mask_path}' not found")
        result = run_single_image(
            model=model,
            image_path=image_path,
            mask_path=mask_path,
            device=device,
            train_config=train_config,
            threshold=args.threshold,
            include_features=include_features,
            gaussian_radius=args.gaussian_radius,
        )
        prediction = result.pop("prediction")
        print("Single-image inference finished.")
        for key, value in result.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
        if args.save_mask:
            _save_mask_image(prediction, Path(args.save_mask))
            print(f"Saved predicted mask to {args.save_mask}")
        return

    dataloader = build_dataloader(
        train_config=train_config,
        split=args.split,
        batch_size=train_config.batch_size,
        num_workers=train_config.num_workers,
        include_features=include_features,
        device=device,
    )

    metrics = run_single_epoch(
        model=model,
        dataloader=dataloader,
        device=device,
        threshold=args.threshold,
        max_batches=args.max_batches,
        show_progress=args.progress,
    )

    print("Inference finished.")
    print(f"Split: {args.split} | Batches: {metrics['batches']} | Threshold: {args.threshold:.2f}")
    print(
        f"loss={metrics['loss']:.4f} "
        f"dice={metrics['dice']:.4f} "
        f"iou={metrics['iou']:.4f} "
        f"precision={metrics['precision']:.4f} "
        f"recall={metrics['recall']:.4f} "
        f"f1={metrics['f1']:.4f}"
    )


if __name__ == "__main__":
    main()
