from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import amp
from torch.utils.data import DataLoader

from data.data_preparation import PreparedForgeryDataset
from model.hybrid_forgery_detector import HybridForgeryConfig, HybridForgeryDetector

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

if torch.cuda.is_available():  # Enable hardware-specific speed-ups when possible
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        probs = torch.sigmoid(logits)
        dims = (1, 2, 3)
        intersection = (probs * targets).sum(dim=dims)
        denom = probs.sum(dim=dims) + targets.sum(dim=dims)
        dice = (2 * intersection + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[float] = 0.25,
        pos_weight: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.register_buffer("_pos_weight", torch.tensor(pos_weight) if pos_weight is not None else None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        pos_weight = self._pos_weight
        if pos_weight is not None and pos_weight.device != logits.device:
            pos_weight = pos_weight.to(logits.device)
        bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight, reduction="none")
        probs = torch.sigmoid(logits)
        probs_t = targets * probs + (1 - targets) * (1 - probs)
        modulating = (1.0 - probs_t).clamp(min=0) ** self.gamma
        if self.alpha is not None:
            alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
            modulating = modulating * alpha_t
        return (modulating * bce).mean()


@dataclass
class TrainConfig:
    """Configuration bundle for launching training runs.

    Designed so notebooks can instantiate and edit this directly before calling
    :func:`run_training`.
    """

    prepared_root: str = "prepared/CASIA2"
    target_size: int = 128
    train_split: str = "train"
    val_split: str = "val"
    batch_size: int = 8
    num_epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    num_workers: int = 4
    prefetch_factor: Optional[int] = 4
    persistent_workers: bool = True
    pin_memory: Optional[bool] = None
    grad_accumulation_steps: int = 1
    device: Optional[str] = None
    grad_clip_norm: Optional[float] = 1.0
    log_interval: int = 10
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval: int = 1
    save_best_only: bool = True
    use_amp: bool = True
    max_train_batches: Optional[int] = None
    max_val_batches: Optional[int] = None
    resume_from: Optional[str] = None
    include_aux_features: Optional[bool] = None
    loss_type: str = "bce_dice"
    bce_weight: float = 0.7
    dice_weight: float = 0.3
    pos_weight: Optional[float] = None
    dice_smooth: float = 1.0
    focal_gamma: float = 2.0
    focal_alpha: Optional[float] = 0.25
    eval_thresholds: Optional[List[float]] = field(default_factory=lambda: [0.3, 0.5, 0.7])
    primary_eval_threshold: Optional[float] = None
    lr_scheduler_type: Optional[str] = None
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 2
    model_config: HybridForgeryConfig = field(default_factory=HybridForgeryConfig)

    def resolved_device(self) -> torch.device:
        if self.device:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer:
    def __init__(self, config: TrainConfig) -> None:
        self.config = config
        self.device = config.resolved_device()
        self.model = HybridForgeryDetector(config.model_config).to(self.device)
        self.criterion = self._build_loss_fn()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        self.scheduler = self._build_scheduler()

        raw_thresholds = config.eval_thresholds or [0.5]
        self.eval_thresholds = sorted({float(thr) for thr in raw_thresholds}) or [0.5]
        if config.primary_eval_threshold is not None:
            self.primary_threshold = float(config.primary_eval_threshold)
            if self.primary_threshold not in self.eval_thresholds:
                self.eval_thresholds.append(self.primary_threshold)
                self.eval_thresholds.sort()
        else:
            self.primary_threshold = float(self.eval_thresholds[0])

        self._autocast_factory = lambda enabled=True: nullcontext()
        amp_enabled = config.use_amp and self.device.type == "cuda"
        if amp_enabled:
            try:
                self.scaler = amp.GradScaler(device_type="cuda", enabled=True)
                self._autocast_factory = lambda enabled=True: amp.autocast(
                    device_type="cuda", enabled=enabled
                )
            except TypeError:
                from torch.cuda.amp import GradScaler as LegacyGradScaler, autocast as legacy_autocast

                self.scaler = LegacyGradScaler(enabled=True)
                self._autocast_factory = lambda enabled=True: legacy_autocast(enabled=enabled)
        else:
            self.scaler = amp.GradScaler(enabled=False)

        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.train_loader = self._build_dataloader(config.train_split, shuffle=True)
        try:
            self.val_loader = self._build_dataloader(config.val_split, shuffle=False)
        except FileNotFoundError:
            self.val_loader = None
        self.best_val_loss = math.inf
        self.start_epoch = 1
        if self.config.resume_from:
            self.start_epoch = self._load_checkpoint(self.config.resume_from)

    def _build_dataloader(self, split: str, shuffle: bool) -> DataLoader:
        dataset = PreparedForgeryDataset(
            prepared_root=self.config.prepared_root,
            split=split,
            target_size=self.config.target_size,
            include_features=self._should_include_aux_features(),
            return_masks=True,
        )
        pin_memory = self.config.pin_memory
        if pin_memory is None:
            pin_memory = self.device.type == "cuda"

        loader_kwargs = dict(
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=pin_memory,
            collate_fn=self._collate_batch,
        )
        if self.config.num_workers > 0:
            loader_kwargs["persistent_workers"] = self.config.persistent_workers
            if self.config.prefetch_factor is not None:
                loader_kwargs["prefetch_factor"] = self.config.prefetch_factor

        return DataLoader(dataset, **loader_kwargs)

    def _should_include_aux_features(self) -> bool:
        if self.config.include_aux_features is not None:
            return self.config.include_aux_features
        return bool(self.config.model_config.use_noise_branch)

    @staticmethod
    def _collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
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
            if all(key in item for item in batch):
                collated[key] = torch.stack([item[key].float() for item in batch])

        if all("label" in item for item in batch):
            collated["label"] = torch.stack([item["label"].long() for item in batch])

        if all("meta" in item for item in batch):
            collated["meta"] = [item["meta"] for item in batch]

        return collated

    def _prepare_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        prepared: Dict[str, Any] = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                prepared[key] = value.to(self.device, non_blocking=True)
            else:
                prepared[key] = value
        return prepared

    def _autocast(self):
        return self._autocast_factory(self.scaler.is_enabled())

    def _extract_noise_inputs(self, batch: Dict[str, Any]) -> Optional[Dict[str, torch.Tensor]]:
        if not self.config.model_config.use_noise_branch:
            return None
        noise: Dict[str, torch.Tensor] = {}
        for key in ("residual", "high_pass"):
            tensor = batch.get(key)
            if torch.is_tensor(tensor):
                noise[key] = tensor
        return noise or None

    def _build_loss_fn(self) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        loss_type = (self.config.loss_type or "bce").lower()
        dice_loss = SoftDiceLoss(smooth=self.config.dice_smooth)

        def bce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            pos_weight = self._get_pos_weight_tensor(logits)
            return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)

        if loss_type == "dice":
            return lambda logits, targets: dice_loss(logits, targets)
        if loss_type == "focal":
            focal = FocalLoss(
                gamma=self.config.focal_gamma,
                alpha=self.config.focal_alpha,
                pos_weight=self.config.pos_weight,
            )

            def focal_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
                return focal(logits, targets)

            return focal_loss
        if loss_type == "bce_dice":
            bce_weight = self.config.bce_weight
            dice_weight = self.config.dice_weight
            denom = bce_weight + dice_weight
            bce_scale = bce_weight / denom if denom > 0 else 0.5
            dice_scale = dice_weight / denom if denom > 0 else 0.5

            def combined_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
                return bce_scale * bce_loss(logits, targets) + dice_scale * dice_loss(logits, targets)

            return combined_loss
        return bce_loss

    def _get_pos_weight_tensor(self, reference: torch.Tensor) -> Optional[torch.Tensor]:
        if self.config.pos_weight is None:
            return None
        return torch.tensor(
            self.config.pos_weight,
            device=reference.device,
            dtype=reference.dtype,
        )

    def _build_scheduler(self):
        sched_type = (self.config.lr_scheduler_type or "").lower()
        if sched_type == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                factor=self.config.lr_scheduler_factor,
                patience=self.config.lr_scheduler_patience,
            )
        if sched_type == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs,
            )
        return None

    def _step_scheduler(self, metric: float) -> None:
        if not self.scheduler:
            return
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(metric)
        else:
            self.scheduler.step()

    def train(self) -> Dict[str, List[float]]:
        history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_dice": [],
            "val_iou": [],
            "val_precision": [],
            "val_recall": [],
            "val_f1": [],
        }
        if self.start_epoch > self.config.num_epochs:
            print(
                f"Start epoch ({self.start_epoch}) exceeds configured num_epochs ({self.config.num_epochs}); "
                "skipping training."
            )
            return history

        for epoch in range(self.start_epoch, self.config.num_epochs + 1):
            train_loss = self._run_epoch(epoch)
            history["train_loss"].append(train_loss)

            val_metrics = {
                "loss": float("nan"),
                "dice": float("nan"),
                "iou": float("nan"),
                "precision": float("nan"),
                "recall": float("nan"),
                "f1": float("nan"),
            }
            if self.val_loader is not None:
                val_metrics = self._run_validation()
                history["val_loss"].append(val_metrics["loss"])
                history["val_dice"].append(val_metrics["dice"])
                history["val_iou"].append(val_metrics["iou"])
                history["val_precision"].append(val_metrics["precision"])
                history["val_recall"].append(val_metrics["recall"])
                history["val_f1"].append(val_metrics["f1"])

                primary_metrics = val_metrics.get("thresholds", {}).get(self.primary_threshold, {})
                best_summary = val_metrics.get("best_threshold", {})
                if primary_metrics:
                    best_thr_val = best_summary.get("value")
                    best_thr_fmt = f"{best_thr_val:.2f}" if isinstance(best_thr_val, (float, int)) else "n/a"
                    best_f1_val = best_summary.get("f1")
                    best_f1_fmt = f"{best_f1_val:.4f}" if isinstance(best_f1_val, (float, int)) else "n/a"
                    print(
                        f"Epoch {epoch} [val] loss={val_metrics['loss']:.4f} "
                        f"dice@{self.primary_threshold:.2f}={primary_metrics.get('dice', float('nan')):.4f} "
                        f"precision={primary_metrics.get('precision', float('nan')):.4f} "
                        f"recall={primary_metrics.get('recall', float('nan')):.4f} "
                        f"best_thr={best_thr_fmt} best_f1={best_f1_fmt}"
                    )

            if epoch % self.config.checkpoint_interval == 0:
                self._save_checkpoint(epoch, val_metrics)

            scheduler_metric = val_metrics.get("loss", train_loss)
            if math.isnan(scheduler_metric):
                scheduler_metric = train_loss
            self._step_scheduler(scheduler_metric)
        return history

    def _run_epoch(self, epoch: int) -> float:
        self.model.train()
        running_loss = 0.0
        total_steps = len(self.train_loader)
        iterator = self.train_loader
        if tqdm:
            iterator = tqdm(iterator, desc=f"Epoch {epoch} [train]", leave=False)
        steps_completed = 0
        accum_target = max(1, self.config.grad_accumulation_steps)
        accum_counter = 0
        self.optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(iterator, start=1):
            batch = self._prepare_batch(batch)
            images, masks = batch["image"], batch["mask"]
            noise_inputs = self._extract_noise_inputs(batch)
            with self._autocast():
                logits = self.model(images, noise_features=noise_inputs)
                if logits.shape[-2:] != masks.shape[-2:]:
                    logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
                loss = self.criterion(logits, masks)
            running_loss += loss.item()

            scaled_loss = loss / accum_target
            self.scaler.scale(scaled_loss).backward()
            accum_counter += 1

            reached_cap = self.config.max_train_batches is not None and step >= self.config.max_train_batches
            should_step = accum_counter == accum_target or step == total_steps or reached_cap
            if should_step:
                if self.config.grad_clip_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                accum_counter = 0

            steps_completed = step
            if not tqdm and step % self.config.log_interval == 0:
                print(
                    f"Epoch {epoch} | Step {step}/{total_steps} "
                    f"| Loss: {loss.item():.4f} | Avg: {running_loss / step:.4f}"
                )
            if reached_cap:
                break
        steps_denominator = steps_completed if steps_completed > 0 else 1
        return running_loss / steps_denominator

    @torch.inference_mode()
    def _run_validation(self) -> Dict[str, float]:
        if self.val_loader is None:
            return {
                "loss": float("nan"),
                "dice": float("nan"),
                "iou": float("nan"),
                "precision": float("nan"),
                "recall": float("nan"),
                "f1": float("nan"),
                "thresholds": {},
                "best_threshold": {},
            }
        self.model.eval()
        total_loss = 0.0
        steps = 0
        max_val_batches = self.config.max_val_batches
        threshold_stats = self._init_threshold_accumulators()
        for batch_idx, batch in enumerate(self.val_loader, start=1):
            batch = self._prepare_batch(batch)
            images, masks = batch["image"], batch["mask"]
            noise_inputs = self._extract_noise_inputs(batch)
            with self._autocast():
                logits = self.model(images, noise_features=noise_inputs)
            if logits.shape[-2:] != masks.shape[-2:]:
                logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            loss = self.criterion(logits, masks)
            probs = torch.sigmoid(logits)
            self._update_threshold_stats(probs, masks, threshold_stats)

            total_loss += loss.item()
            steps += 1
            if max_val_batches is not None and batch_idx >= max_val_batches:
                break
        avg_loss = total_loss / steps if steps else float("nan")
        metrics_by_threshold, best_summary = self._finalize_threshold_metrics(threshold_stats)
        primary_metrics = metrics_by_threshold.get(self.primary_threshold, {})
        result = {
            "loss": avg_loss,
            "thresholds": metrics_by_threshold,
            "best_threshold": best_summary,
        }
        if primary_metrics:
            result.update(
                {
                    "dice": primary_metrics.get("dice", float("nan")),
                    "iou": primary_metrics.get("iou", float("nan")),
                    "precision": primary_metrics.get("precision", float("nan")),
                    "recall": primary_metrics.get("recall", float("nan")),
                    "f1": primary_metrics.get("f1", float("nan")),
                }
            )
        return result

    def _save_checkpoint(self, epoch: int, val_metrics: Dict[str, float]) -> None:
        checkpoint = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scaler_state": self.scaler.state_dict(),
            "config": {
                **asdict(self.config),
                "model_config": asdict(self.config.model_config),
            },
            "val_metrics": val_metrics,
        }
        target = self.checkpoint_dir / f"epoch_{epoch:03d}.pt"
        torch.save(checkpoint, target)

        current_loss = val_metrics.get("loss", math.inf)
        if self.config.save_best_only and current_loss < self.best_val_loss:
            self.best_val_loss = current_loss
            best_path = self.checkpoint_dir / "best.pt"
            torch.save(checkpoint, best_path)

    def _load_checkpoint(self, checkpoint_path: str) -> int:
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint '{checkpoint_path}' does not exist")
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self._move_optimizer_state_to_device()
        scaler_state = checkpoint.get("scaler_state")
        if scaler_state is not None:
            self.scaler.load_state_dict(scaler_state)
        val_loss = checkpoint.get("val_metrics", {}).get("loss", math.inf)
        self.best_val_loss = val_loss
        loaded_epoch = checkpoint.get("epoch", 0)
        print(f"Loaded checkpoint from {path} (epoch {loaded_epoch})")
        return loaded_epoch + 1

    def _move_optimizer_state_to_device(self) -> None:
        for state in self.optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(self.device)

    def _init_threshold_accumulators(self) -> Dict[float, Dict[str, float]]:
        return {
            float(thr): {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0}
            for thr in self.eval_thresholds
        }

    def _update_threshold_stats(
        self,
        probs: torch.Tensor,
        targets: torch.Tensor,
        accumulators: Dict[float, Dict[str, float]],
    ) -> None:
        targets_bin = (targets > 0.5).float()
        inv_targets = 1.0 - targets_bin
        for threshold, stats in accumulators.items():
            preds = (probs > threshold).float()
            inv_preds = 1.0 - preds
            tp = (preds * targets_bin).sum().double().item()
            fp = (preds * inv_targets).sum().double().item()
            fn = (inv_preds * targets_bin).sum().double().item()
            tn = (inv_preds * inv_targets).sum().double().item()
            stats["tp"] += tp
            stats["fp"] += fp
            stats["fn"] += fn
            stats["tn"] += tn

    def _finalize_threshold_metrics(
        self, accumulators: Dict[float, Dict[str, float]]
    ) -> tuple[Dict[float, Dict[str, float | Dict[str, float]]], Dict[str, float | Dict[str, float]]]:
        eps = 1e-8
        metrics_by_threshold: Dict[float, Dict[str, float]] = {}
        best_threshold = None
        best_f1 = -math.inf
        for threshold, stats in accumulators.items():
            tp = stats["tp"]
            fp = stats["fp"]
            fn = stats["fn"]
            tn = stats["tn"]
            support = tp + fp + fn + tn
            precision = tp / (tp + fp + eps) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn + eps) if (tp + fn) > 0 else 0.0
            dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
            iou = (tp + eps) / (tp + fp + fn + eps)
            f1 = (2 * precision * recall / (precision + recall + eps)) if (precision + recall) > 0 else 0.0
            accuracy = (tp + tn) / support if support > 0 else 0.0
            metrics_by_threshold[threshold] = {
                "dice": dice,
                "iou": iou,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "accuracy": accuracy,
                "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
            }
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        best_summary: Dict[str, float] = {"value": best_threshold if best_threshold is not None else float("nan")}
        if best_threshold is not None:
            best_summary.update(metrics_by_threshold[best_threshold])
        return metrics_by_threshold, best_summary


def run_training(config: TrainConfig) -> Dict[str, List[float]]:
    """High-level helper so notebooks can simply do ``run_training(TrainConfig(...))``."""

    trainer = Trainer(config)
    train_batches = len(trainer.train_loader)
    val_batches = 0 if trainer.val_loader is None else len(trainer.val_loader)
    print(
        f"Starting training for {config.num_epochs} epochs on {trainer.device} | "
        f"train batches: {train_batches} | val batches: {val_batches}"
    )
    if config.resume_from:
        print(
            f"Resuming from checkpoint '{config.resume_from}' starting at epoch {trainer.start_epoch}"
        )
    history = trainer.train()
    print("Training finished. Checkpoints saved to", trainer.checkpoint_dir)
    return history


if __name__ == "__main__":
    default_config = TrainConfig()
    run_training(default_config)
