from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

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
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )

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
            include_features=False,
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

    @staticmethod
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

    def _prepare_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

    def _autocast(self):
        return self._autocast_factory(self.scaler.is_enabled())

    def train(self) -> Dict[str, List[float]]:
        history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "val_dice": [], "val_iou": []}
        if self.start_epoch > self.config.num_epochs:
            print(
                f"Start epoch ({self.start_epoch}) exceeds configured num_epochs ({self.config.num_epochs}); "
                "skipping training."
            )
            return history

        for epoch in range(self.start_epoch, self.config.num_epochs + 1):
            train_loss = self._run_epoch(epoch)
            history["train_loss"].append(train_loss)

            val_metrics = {"loss": float("nan"), "dice": float("nan"), "iou": float("nan")}
            if self.val_loader is not None:
                val_metrics = self._run_validation()
                history["val_loss"].append(val_metrics["loss"])
                history["val_dice"].append(val_metrics["dice"])
                history["val_iou"].append(val_metrics["iou"])

            if epoch % self.config.checkpoint_interval == 0:
                self._save_checkpoint(epoch, val_metrics)
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
            with self._autocast():
                logits = self.model(images)
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
            return {"loss": float("nan"), "dice": float("nan"), "iou": float("nan")}
        self.model.eval()
        total_loss = 0.0
        total_dice = 0.0
        total_iou = 0.0
        steps = 0
        max_val_batches = self.config.max_val_batches
        for batch_idx, batch in enumerate(self.val_loader, start=1):
            batch = self._prepare_batch(batch)
            images, masks = batch["image"], batch["mask"]
            with self._autocast():
                logits = self.model(images)
            if logits.shape[-2:] != masks.shape[-2:]:
                logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            loss = self.criterion(logits, masks)
            dice, iou = self._compute_segmentation_metrics(logits, masks)

            total_loss += loss.item()
            total_dice += dice
            total_iou += iou
            steps += 1
            if max_val_batches is not None and batch_idx >= max_val_batches:
                break
        return {"loss": total_loss / steps, "dice": total_dice / steps, "iou": total_iou / steps}

    @staticmethod
    def _compute_segmentation_metrics(logits: torch.Tensor, targets: torch.Tensor) -> tuple[float, float]:
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        eps = 1e-6
        intersection = (preds * targets).sum(dim=(1, 2, 3))
        union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - intersection
        dice = (2 * intersection + eps) / (preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + eps)
        iou = (intersection + eps) / (union + eps)
        return dice.mean().item(), iou.mean().item()

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
