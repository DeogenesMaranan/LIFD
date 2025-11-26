from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LossConfig:
    # BCE (recommended always ON)
    use_bce: bool = True
    bce_weight: float = 1.0
    bce_pos_weight: Optional[float] = None

    # Dice loss
    use_dice: bool = True
    dice_weight: float = 1.0
    dice_smooth: float = 1e-5

    # Focal Tversky
    use_focal_tversky: bool = False
    focal_tversky_weight: float = 0.5
    focal_alpha: float = 0.7
    focal_beta: float = 0.3
    focal_gamma: float = 1.33

    # Boundary loss
    use_boundary: bool = False
    boundary_weight: float = 0.05
    boundary_kernel_size: int = 3

    # Lovasz hinge
    apply_lovasz: bool = False
    lovasz_weight: float = 0.5


def _flatten(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.view(tensor.size(0), -1)


class CombinedSegmentationLoss(nn.Module):
    """
    Final high-quality segmentation loss:
    BCEWithLogitsLoss + Dice + optional extras
    """

    def __init__(self, config: LossConfig) -> None:
        super().__init__()
        self.config = config

        # BCEWithLogitsLoss ALWAYS (best practice)
        pos_weight = None
        if config.bce_pos_weight is not None:
            pos_weight = torch.tensor(config.bce_pos_weight, dtype=torch.float32)

        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight) if config.use_bce else None

        # Boundary kernels
        self.register_buffer(
            "boundary_kernel_x",
            self._make_sobel_kernel("x", config.boundary_kernel_size)
            if config.use_boundary else torch.empty(0),
            persistent=False,
        )
        self.register_buffer(
            "boundary_kernel_y",
            self._make_sobel_kernel("y", config.boundary_kernel_size)
            if config.use_boundary else torch.empty(0),
            persistent=False,
        )

    @staticmethod
    def _make_sobel_kernel(axis: str, ksize: int) -> torch.Tensor:
        if ksize not in {3, 5}:
            ksize = 3

        if ksize == 3:
            kernel = torch.tensor(
                [[1, 0, -1], [2, 0, -2], [1, 0, -1]] if axis == "x" else
                [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                dtype=torch.float32
            )
        else:  # 5x5
            if axis == "x":
                kernel = torch.tensor(
                    [[2, 1, 0, -1, -2],
                     [3, 2, 0, -2, -3],
                     [4, 3, 0, -3, -4],
                     [3, 2, 0, -2, -3],
                     [2, 1, 0, -1, -2]], dtype=torch.float32
                )
            else:
                kernel = torch.tensor(
                    [[2, 3, 4, 3, 2],
                     [1, 2, 3, 2, 1],
                     [0, 0, 0, 0, 0],
                     [-1, -2, -3, -2, -1],
                     [-2, -3, -4, -3, -2]], dtype=torch.float32
                )

        return kernel.view(1, 1, ksize, ksize)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        total = torch.zeros(1, device=logits.device)
        terms: Dict[str, float] = {}

        # BCE (logits → BCEWithLogits handles sigmoid internally)
        if self.config.use_bce:
            bce_loss = self.bce(logits, targets)
            total += self.config.bce_weight * bce_loss
            terms["bce"] = bce_loss.item()

        # Dice (probabilities)
        if self.config.use_dice:
            probs = torch.sigmoid(logits)
            dice_loss = self._dice_loss(probs, targets)
            total += self.config.dice_weight * dice_loss
            terms["dice"] = dice_loss.item()

        # Optional losses
        if self.config.use_focal_tversky:
            probs = torch.sigmoid(logits)
            ft = self._focal_tversky_loss(probs, targets)
            total += self.config.focal_tversky_weight * ft
            terms["focal_tversky"] = ft.item()

        if self.config.use_boundary:
            probs = torch.sigmoid(logits)
            boundary = self._boundary_loss(probs, targets)
            total += self.config.boundary_weight * boundary
            terms["boundary"] = boundary.item()

        if self.config.apply_lovasz:
            lovasz = self._lovasz_hinge(logits, targets)
            total += self.config.lovasz_weight * lovasz
            terms["lovasz"] = lovasz.item()

        return total.squeeze(), terms

    # === Loss components ===

    def _dice_loss(self, probs, targets):
        eps = self.config.dice_smooth
        p = _flatten(probs)
        t = _flatten(targets)
        intersection = (p * t).sum(dim=-1)
        denom = p.sum(dim=-1) + t.sum(dim=-1)
        dice = (2 * intersection + eps) / (denom + eps)
        return 1 - dice.mean()

    def _focal_tversky_loss(self, probs, targets):
        eps = 1e-6
        p = _flatten(probs)
        t = _flatten(targets)
        tp = (p * t).sum(dim=-1)
        fp = (p * (1 - t)).sum(dim=-1)
        fn = ((1 - p) * t).sum(dim=-1)
        alpha = self.config.focal_alpha
        beta = self.config.focal_beta
        gamma = self.config.focal_gamma
        tversky = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
        return torch.pow(1 - tversky, gamma).mean()

    def _boundary_loss(self, probs, targets):
        if self.boundary_kernel_x.numel() == 0:
            return torch.tensor(0.0, device=probs.device)
        g1 = self._spatial_gradient(probs)
        g2 = self._spatial_gradient(targets)
        return F.l1_loss(g1, g2)

    def _spatial_gradient(self, x):
        pad = self.boundary_kernel_x.shape[-1] // 2
        kx = self.boundary_kernel_x
        ky = self.boundary_kernel_y
        gx = F.conv2d(x, kx, padding=pad)
        gy = F.conv2d(x, ky, padding=pad)
        return torch.sqrt(gx * gx + gy * gy + 1e-6)

    def _lovasz_hinge(self, logits, targets):
        signs = targets * 2 - 1
        errors = 1 - logits * signs
        errors_sorted, perm = torch.sort(errors.view(errors.size(0), -1), dim=1, descending=True)
        perm = perm.detach()
        targets_sorted = targets.view(targets.size(0), -1).gather(1, perm)
        grad = self._lovasz_grad(targets_sorted)
        return (F.relu(errors_sorted) * grad).mean()

    @staticmethod
    def _lovasz_grad(gt_sorted):
        gts = gt_sorted.sum(dim=1, keepdim=True)
        intersection = gts - gt_sorted.cumsum(dim=1)
        union = gts + (1 - gt_sorted).cumsum(dim=1)
        jacc = 1 - intersection / union.clamp_min(1.0)
        return torch.cat([jacc[:, :1], jacc[:, 1:] - jacc[:, :-1]], dim=1)
