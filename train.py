from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from datetime import datetime
from torch import amp
from torch.utils.data import DataLoader, WeightedRandomSampler

from data.data_preparation import (
    PreparedForgeryDataset,
    EvenMultiSourceBatchSampler,
    EvenMultiSourceBalancedSampler,
)
from model.hybrid_forgery_detector import HybridForgeryConfig, HybridForgeryDetector
from losses.segmentation_losses import CombinedSegmentationLoss, LossConfig

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

    prepared_root: str | Sequence[str] = "prepared/CASIA2"
    target_size: int = 320
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
    use_channels_last: bool = True
    use_torch_compile: bool = True
    torch_compile_backend: Optional[str] = None
    # Prefetch batches and move them to device asynchronously
    use_data_prefetch: bool = True
    max_train_batches: Optional[int] = None
    max_val_batches: Optional[int] = None
    resume_from: Optional[str] = None
    auto_continue_latest: bool = True
    include_aux_features: Optional[bool] = None
    loss_config: LossConfig = field(default_factory=LossConfig)
    balance_real_fake: bool = True
    balanced_positive_ratio: float = 0.5
    eval_thresholds: Optional[List[float]] = field(default_factory=lambda: [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    primary_eval_threshold: Optional[float] = None
    lr_scheduler_type: Optional[str] = None
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 2
    early_stopping_patience: Optional[int] = None
    early_stopping_min_delta: float = 1e-4
    model_config: HybridForgeryConfig = field(default_factory=HybridForgeryConfig)

    def resolved_device(self) -> torch.device:
        if self.device:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer:
    def __init__(self, config: TrainConfig) -> None:
        self.config = config
        self.device = config.resolved_device()
        model = HybridForgeryDetector(config.model_config)
        try:
            if config.use_channels_last and self.device.type == "cuda":
                self.model = model.to(self.device, memory_format=torch.channels_last)
                self._use_channels_last = True
            else:
                self.model = model.to(self.device)
                self._use_channels_last = False
        except Exception:
            self.model = model.to(self.device)
            self._use_channels_last = False
        self.loss_fn = CombinedSegmentationLoss(config.loss_config).to(self.device)
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
                self.scaler = amp.GradScaler("cuda", enabled=True)
                self._autocast_factory = lambda enabled=True: amp.autocast("cuda", enabled=enabled)
            except TypeError:
                
                try:
                    self.scaler = amp.GradScaler(device_type="cuda", enabled=True)
                    self._autocast_factory = lambda enabled=True: amp.autocast(device_type="cuda", enabled=enabled)
                except Exception:
                    from torch.cuda.amp import GradScaler as LegacyGradScaler, autocast as legacy_autocast

                    self.scaler = LegacyGradScaler(enabled=True)
                    self._autocast_factory = lambda enabled=True: legacy_autocast(enabled=enabled)
        else:
            self.scaler = amp.GradScaler(enabled=False)

        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        try:
            if self.device.type == "cuda":
                try:
                    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
                        if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
                            torch.backends.cuda.matmul.fp32_precision = "tf32"
                            print("Enabled torch.backends.cuda.matmul.fp32_precision='tf32'")
                        elif hasattr(torch, "set_float32_matmul_precision"):
                            torch.set_float32_matmul_precision("high")
                            print("Called torch.set_float32_matmul_precision('high') (fallback)")
                except Exception:
                    pass

                try:
                    if hasattr(torch.backends, "cudnn") and hasattr(torch.backends.cudnn, "conv"):
                        if hasattr(torch.backends.cudnn.conv, "fp32_precision"):
                            torch.backends.cudnn.conv.fp32_precision = "tf32"
                            print("Enabled torch.backends.cudnn.conv.fp32_precision='tf32'")
                except Exception:
                    pass
        except Exception:
            pass

        if getattr(config, "use_torch_compile", False):
            try:
                compile_kwargs = {}
                backend = getattr(config, "torch_compile_backend", None)
                if backend:
                    compile_kwargs["backend"] = backend
                if hasattr(torch, "compile"):
                    self.model = torch.compile(self.model, **compile_kwargs)
            except Exception as exc:
                print(f"Warning: torch.compile failed or unavailable: {exc}")

        self.train_loader = self._build_dataloader(config.train_split, shuffle=True)
        try:
            self.val_loader = self._build_dataloader(config.val_split, shuffle=False)
        except FileNotFoundError:
            self.val_loader = None
        self.best_val_loss = math.inf
        self._epochs_without_improvement = 0
        self.start_epoch = 1
        if self.config.resume_from:
            self.start_epoch = self._load_checkpoint(self.config.resume_from)
        elif getattr(self.config, "auto_continue_latest", False):
            latest = self._find_latest_epoch_checkpoint()
            if latest is not None:
                try:
                    self.start_epoch = self._load_checkpoint(str(latest))
                except Exception as exc:
                    print(f"Failed to auto-load latest checkpoint '{latest}': {exc}")

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

        sampler = None
        multi_source = False
        try:
            prepared_roots = {rec.get("prepared_root") for rec in dataset.records}
            multi_source = len(prepared_roots) > 1
        except Exception:
            multi_source = False

        # If multi-source, use a sampler that keeps per-batch dataset evenness.
        # If class balancing is requested, use the combined balanced sampler
        # that performs per-source class-aware sampling; otherwise use the
        # simpler even-per-source sampler.
        if shuffle and multi_source:
            if self.config.balance_real_fake:
                pos_ratio = float(self.config.balanced_positive_ratio)
                batch_sampler = EvenMultiSourceBalancedSampler(
                    dataset, batch_size=self.config.batch_size, pos_ratio=pos_ratio, shuffle=True, seed=self.config.loss_config.__dict__.get('seed', 42) if hasattr(self.config, 'loss_config') else 42
                )
            else:
                batch_sampler = EvenMultiSourceBatchSampler(dataset, batch_size=self.config.batch_size, shuffle=True, drop_last=False, seed=self.config.loss_config.__dict__.get('seed', 42) if hasattr(self.config, 'loss_config') else 42)
            loader_kwargs = dict(
                batch_sampler=batch_sampler,
                num_workers=self.config.num_workers,
                pin_memory=pin_memory,
                collate_fn=self._collate_batch,
            )
            if self.config.num_workers > 0:
                loader_kwargs["persistent_workers"] = self.config.persistent_workers
                if self.config.prefetch_factor is not None:
                    loader_kwargs["prefetch_factor"] = self.config.prefetch_factor
            return DataLoader(dataset, **loader_kwargs)

        # Fallback: previous behavior (including optional balanced sampler)
        if shuffle and self.config.balance_real_fake:
            sampler = self._build_balanced_sampler(dataset)
            if sampler is not None:
                shuffle = False

        loader_kwargs = dict(
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=pin_memory,
            collate_fn=self._collate_batch,
        )
        if sampler is not None:
            loader_kwargs["sampler"] = sampler
        if self.config.num_workers > 0:
            loader_kwargs["persistent_workers"] = self.config.persistent_workers
            if self.config.prefetch_factor is not None:
                loader_kwargs["prefetch_factor"] = self.config.prefetch_factor

        return DataLoader(dataset, **loader_kwargs)

    def _build_scheduler(self):
        """Build learning-rate scheduler based on config.

        This method lives on the Trainer so `self._build_scheduler()` called
        from `__init__` resolves correctly even if there's an accidental
        duplicate definition elsewhere in the file.
        """
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


class DataPrefetcher:
    """Simple prefetcher that moves batches to `device` on a separate CUDA stream.

    It expects the underlying DataLoader to yield CPU tensors. The prefetcher
    will move tensors (and optionally convert them to channels-last) so the
    training loop can consume already-device-resident batches.
    """

    def __init__(self, loader, device: torch.device, use_channels_last: bool = False):
        self.loader = iter(loader)
        self.device = device
        self.use_channels_last = use_channels_last
        self.stream = torch.cuda.Stream() if self.device.type == "cuda" else None
        self.next_batch = None
        self._preload()

    def _to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        for k, v in list(batch.items()):
            if torch.is_tensor(v):
                if self.use_channels_last and k in ("image", "high_pass", "residual") and v.ndim == 4:
                    batch[k] = v.to(self.device, non_blocking=True).contiguous(memory_format=torch.channels_last)
                else:
                    batch[k] = v.to(self.device, non_blocking=True).contiguous()
        return batch

    def _preload(self):
        try:
            batch = next(self.loader)
        except StopIteration:
            self.next_batch = None
            return
        if self.stream is None:
            self.next_batch = self._to_device(batch)
            return
        # Move to device on the prefetch stream
        with torch.cuda.stream(self.stream):
            self.next_batch = self._to_device(batch)

    def next(self) -> Optional[Dict[str, Any]]:
        if self.stream is not None:
            # Ensure current stream waits for preload stream copy
            torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.next_batch
        if batch is None:
            return None
        # Kick off loading of the next batch
        self._preload()
        return batch

    def __iter__(self):
        while True:
            batch = self.next()
            if batch is None:
                break
            yield batch

    def _build_balanced_sampler(self, dataset: PreparedForgeryDataset) -> Optional[WeightedRandomSampler]:
        labels = getattr(dataset, "sample_labels", None)
        if not labels:
            return None
        label_tensor = torch.tensor(labels, dtype=torch.long)
        class_counts = torch.bincount(label_tensor, minlength=2).float()
        if class_counts.sum() == 0:
            return None
        pos_ratio = float(self.config.balanced_positive_ratio)
        pos_ratio = min(max(pos_ratio, 0.05), 0.95)
        neg_ratio = 1.0 - pos_ratio
        class_weights = torch.zeros_like(class_counts)
        class_weights[0] = neg_ratio / class_counts[0].clamp_min(1.0)
        class_weights[1] = pos_ratio / class_counts[1].clamp_min(1.0)
        sample_weights = class_weights[label_tensor]
        return WeightedRandomSampler(sample_weights.double(), num_samples=len(sample_weights), replacement=True)

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
                if self._use_channels_last and key in ("image", "high_pass", "residual"):
                    prepared[key] = (
                        value.to(self.device, non_blocking=True)
                        .contiguous(memory_format=torch.channels_last)
                    )
                else:
                    prepared[key] = value.to(self.device, non_blocking=True).contiguous()
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
                val_metrics = self._run_validation(epoch)
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

            improved = False
            if self.val_loader is not None:
                val_loss_value = val_metrics.get("loss", math.inf)
                if math.isfinite(val_loss_value):
                    improved = self._register_val_result(val_loss_value)

            if epoch % self.config.checkpoint_interval == 0:
                self._save_checkpoint(epoch, val_metrics, is_best=improved)

            scheduler_metric = val_metrics.get("loss", train_loss)
            if math.isnan(scheduler_metric):
                scheduler_metric = train_loss
            self._step_scheduler(scheduler_metric)

            if (
                self.val_loader is not None
                and self.config.early_stopping_patience is not None
                and self._epochs_without_improvement >= self.config.early_stopping_patience
            ):
                print(
                    f"Early stopping triggered after {self._epochs_without_improvement} epochs "
                    "without validation improvement."
                )
                break
        return history

    def _run_epoch(self, epoch: int) -> float:
        self.model.train()
        running_loss = 0.0
        total_steps = len(self.train_loader)
        iterator = self.train_loader
        use_prefetch = getattr(self.config, "use_data_prefetch", False) and self.device.type == "cuda"
        prefetcher = None
        if use_prefetch:
            # Use DataPrefetcher (already moves tensors to device); skip _prepare_batch
            prefetcher = DataPrefetcher(self.train_loader, self.device, use_channels_last=self._use_channels_last)
            iterator = prefetcher
        else:
            iterator = self.train_loader
            if tqdm:
                iterator = tqdm(iterator, desc=f"Epoch {epoch} [train]", leave=False)
        steps_completed = 0
        accum_target = max(1, self.config.grad_accumulation_steps)
        accum_counter = 0
        self.optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(iterator, start=1):
            # If prefetcher moved tensors to device already, avoid double-moving
            if prefetcher is None:
                batch = self._prepare_batch(batch)
            images, masks = batch["image"], batch["mask"]
            noise_inputs = self._extract_noise_inputs(batch)
            with self._autocast():
                logits = self.model(images, noise_features=noise_inputs)
                if logits.shape[-2:] != masks.shape[-2:]:
                    logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
                loss, _ = self.loss_fn(logits, masks)
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
    def _run_validation(self, epoch: Optional[int] = None) -> Dict[str, float]:
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

        iterator = self.val_loader
        if tqdm:
            total_batches = len(self.val_loader)
            if max_val_batches is not None:
                total_batches = min(total_batches, max_val_batches)
            desc = f"Epoch {epoch} [val]" if epoch is not None else "Validation"
            iterator = tqdm(self.val_loader, desc=desc, total=total_batches, leave=False)

        for batch_idx, batch in enumerate(iterator, start=1):
            batch = self._prepare_batch(batch)
            images, masks = batch["image"], batch["mask"]
            noise_inputs = self._extract_noise_inputs(batch)
            with self._autocast():
                logits = self.model(images, noise_features=noise_inputs)
            if logits.shape[-2:] != masks.shape[-2:]:
                logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            loss, _ = self.loss_fn(logits, masks)
            probs = torch.sigmoid(logits)
            self._update_threshold_stats(probs, masks, threshold_stats)

            total_loss += loss.item()
            steps += 1

            if tqdm:
                avg_so_far = total_loss / steps if steps else float("nan")
                try:
                    iterator.set_postfix({"loss": f"{avg_so_far:.4f}"})
                except Exception:
                    pass

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

    def _register_val_result(self, current_loss: float) -> bool:
        delta = self.config.early_stopping_min_delta
        improved = current_loss + delta < self.best_val_loss
        if improved:
            self.best_val_loss = current_loss
            self._epochs_without_improvement = 0
        else:
            self._epochs_without_improvement += 1
        return improved

    def _save_checkpoint(self, epoch: int, val_metrics: Dict[str, float], is_best: bool = False) -> None:
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
        if is_best or not self.config.save_best_only:
            best_path = self.checkpoint_dir / "best.pt"
            torch.save(checkpoint, best_path)
        # Append this epoch's metrics into a single `metrics.json` file that
        # contains an array of epoch records. This keeps a compact history in
        # one file rather than many per-epoch files.
        try:
            metrics_record = {
                "epoch": epoch,
                "val_metrics": val_metrics,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }
            history_path = self.checkpoint_dir / "metrics.json"
            history: List[Dict[str, Any]] = []
            if history_path.exists():
                try:
                    with history_path.open("r", encoding="utf-8") as f:
                        existing = json.load(f)
                    # If existing is a dict (old format), coerce to list
                    if isinstance(existing, list):
                        history = existing
                    elif isinstance(existing, dict):
                        history = [existing]
                    else:
                        history = []
                except Exception:
                    # If file is unreadable, start fresh but don't crash
                    history = []
            history.append(metrics_record)
            # Write atomically: write to a temp file then replace
            tmp_path = self.checkpoint_dir / "metrics.json.tmp"
            with tmp_path.open("w", encoding="utf-8") as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
            try:
                tmp_path.replace(history_path)
            except Exception:
                # Fallback to non-atomic write on failure
                with history_path.open("w", encoding="utf-8") as f:
                    json.dump(history, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            print(f"Warning: failed to append metrics JSON for epoch {epoch}: {exc}")

    def _find_latest_epoch_checkpoint(self) -> Optional[Path]:
        # Look for files named like `epoch_###.pt` and return the path to the
        # checkpoint with the highest epoch number, or None if none found.
        candidates = []
        try:
            for p in self.checkpoint_dir.iterdir():
                if p.is_file() and p.name.startswith("epoch_") and p.suffix == ".pt":
                    # filename like epoch_001.pt
                    try:
                        num = int(p.stem.split("_")[-1])
                        candidates.append((num, p))
                    except Exception:
                        continue
        except Exception:
            return None
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0])
        return candidates[-1][1]

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
        self._epochs_without_improvement = 0
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
