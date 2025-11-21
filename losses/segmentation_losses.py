from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LossConfig:
    use_bce: bool = True
    bce_weight: float = 1.0
    bce_pos_weight: Optional[float] = None
    use_dice: bool = True
    dice_weight: float = 1.0
    dice_smooth: float = 1e-5
    use_focal_tversky: bool = False
    focal_tversky_weight: float = 0.5
    focal_alpha: float = 0.7
    focal_beta: float = 0.3
    focal_gamma: float = 1.33
    use_boundary: bool = False
    boundary_weight: float = 0.05
    boundary_kernel_size: int = 3
    apply_lovasz: bool = False
    lovasz_weight: float = 0.5


def _sigmoid_probs(logits: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(logits)


def _flatten(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.view(tensor.size(0), -1)


class CombinedSegmentationLoss(nn.Module):
    def __init__(self, config: LossConfig) -> None:
        super().__init__()
        self.config = config
        pos_weight = None
        if config.bce_pos_weight is not None:
            pos_weight = torch.tensor(config.bce_pos_weight, dtype=torch.float32)
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight) if config.use_bce else None
        self.register_buffer(
            "boundary_kernel_x",
            self._make_sobel_kernel(axis="x", ksize=config.boundary_kernel_size)
            if config.use_boundary
            else torch.empty(0),
            persistent=False,
        )
        self.register_buffer(
            "boundary_kernel_y",
            self._make_sobel_kernel(axis="y", ksize=config.boundary_kernel_size)
            if config.use_boundary
            else torch.empty(0),
            persistent=False,
        )

    @staticmethod
    def _make_sobel_kernel(axis: str, ksize: int) -> torch.Tensor:
        if ksize not in {3, 5}:
            ksize = 3
        if ksize == 3:
            if axis == "x":
                kernel = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32)
            else:
                kernel = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32)
        else:
            if axis == "x":
                kernel = torch.tensor(
                    [[2, 1, 0, -1, -2], [3, 2, 0, -2, -3], [4, 3, 0, -3, -4], [3, 2, 0, -2, -3], [2, 1, 0, -1, -2]],
                    dtype=torch.float32,
                )
            else:
                kernel = torch.tensor(
                    [[2, 3, 4, 3, 2], [1, 2, 3, 2, 1], [0, 0, 0, 0, 0], [-1, -2, -3, -2, -1], [-2, -3, -4, -3, -2]],
                    dtype=torch.float32,
                )
        return kernel.view(1, 1, ksize, ksize)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, Dict[str, float]]:
        total_loss = torch.zeros(1, device=logits.device, dtype=logits.dtype)
        terms: Dict[str, float] = {}
        probs = _sigmoid_probs(logits)

        if self.config.use_bce and self.bce is not None:
            bce_loss = self.bce(logits, targets)
            total_loss = total_loss + self.config.bce_weight * bce_loss
            terms["bce"] = bce_loss.detach().item()

        if self.config.use_dice:
            dice_loss = self._dice_loss(probs, targets)
            total_loss = total_loss + self.config.dice_weight * dice_loss
            terms["dice"] = dice_loss.detach().item()

        if self.config.use_focal_tversky:
            ft_loss = self._focal_tversky_loss(probs, targets)
            total_loss = total_loss + self.config.focal_tversky_weight * ft_loss
            terms["focal_tversky"] = ft_loss.detach().item()

        if self.config.use_boundary:
            boundary_loss = self._boundary_loss(probs, targets)
            total_loss = total_loss + self.config.boundary_weight * boundary_loss
            terms["boundary"] = boundary_loss.detach().item()

        if self.config.apply_lovasz:
            lovasz_loss = self._lovasz_hinge(logits, targets)
            total_loss = total_loss + self.config.lovasz_weight * lovasz_loss
            terms["lovasz"] = lovasz_loss.detach().item()

        return total_loss.squeeze(), terms

    def _dice_loss(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        eps = self.config.dice_smooth
        p = _flatten(probs)
        t = _flatten(targets)
        intersection = (p * t).sum(dim=-1)
        denominator = p.sum(dim=-1) + t.sum(dim=-1)
        dice = (2 * intersection + eps) / (denominator + eps)
        return 1 - dice.mean()

    def _focal_tversky_loss(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        alpha = self.config.focal_alpha
        beta = self.config.focal_beta
        gamma = self.config.focal_gamma
        p = _flatten(probs)
        t = _flatten(targets)
        tp = (p * t).sum(dim=-1)
        fp = (p * (1 - t)).sum(dim=-1)
        fn = ((1 - p) * t).sum(dim=-1)
        tversky = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
        return torch.pow(1 - tversky, gamma).mean()

    def _boundary_loss(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.boundary_kernel_x.numel() == 0:
            return torch.tensor(0.0, device=probs.device, dtype=probs.dtype)
        grad_pred = self._spatial_gradient(probs)
        grad_target = self._spatial_gradient(targets)
        return F.l1_loss(grad_pred, grad_target)

    def _spatial_gradient(self, tensor: torch.Tensor) -> torch.Tensor:
        padding = self.boundary_kernel_x.shape[-1] // 2
        grad_x = F.conv2d(tensor, self.boundary_kernel_x, padding=padding)
        grad_y = F.conv2d(tensor, self.boundary_kernel_y, padding=padding)
        return torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-6)

    def _lovasz_hinge(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        signs = targets * 2 - 1
        errors = 1 - logits * signs
        errors_sorted, perm = torch.sort(errors.view(errors.size(0), -1), dim=1, descending=True)
        perm = perm.detach()
        targets_sorted = targets.view(targets.size(0), -1).gather(1, perm)
        grad = self._lovasz_grad(targets_sorted)
        losses = F.relu(errors_sorted) * grad
        return losses.mean()

    @staticmethod
    def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
        gts = gt_sorted.sum(dim=1)
        intersection = gts - gt_sorted.cumsum(dim=1)
        union = gts + (1 - gt_sorted).cumsum(dim=1)
        jaccard = 1 - intersection / union.clamp_min(1)
        jaccard = torch.cat([jaccard[:, :1], jaccard[:, 1:] - jaccard[:, :-1]], dim=1)
        return jaccard
