from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import torch
from torch import nn
import torch.nn.functional as F


class ChannelGate(nn.Module):
    """Channel attention used to re-weight noise-aware features."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return x * self.net(x)


class DepthwisePointwiseRefiner(nn.Module):
    """Depthwise 3x3 + pointwise 1x1 block for cheap refinement."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)


class FeatureFusionModule(nn.Module):
    """Fuse multi-scale feature pyramids originating from different backbones."""

    def __init__(
        self,
        backbone_channels: Dict[str, List[int]],
        fused_channels: int | List[int] = 256,
        fusion_mode: str = "sum",
        noise_backbone_names: Optional[Iterable[str]] = None,
        gate_reduction: int = 4,
    ) -> None:
        super().__init__()
        if not backbone_channels:
            raise ValueError("At least one backbone must be provided for feature fusion.")

        self.num_stages = len(next(iter(backbone_channels.values())))
        self.num_backbones = len(backbone_channels)
        for channels in backbone_channels.values():
            if len(channels) != self.num_stages:
                raise ValueError("All backbones must expose the same number of feature stages.")

        if isinstance(fused_channels, int):
            fused_channels = [fused_channels] * self.num_stages
        if len(fused_channels) != self.num_stages:
            raise ValueError("fused_channels must match the number of feature stages.")
        fusion_mode = fusion_mode.lower()
        if fusion_mode not in {"sum", "concat"}:
            raise ValueError("fusion_mode must be either 'sum' or 'concat'.")
        self.fusion_mode = fusion_mode

        self.fused_channels = fused_channels
        self.backbone_projections = nn.ModuleDict()
        for name, channels in backbone_channels.items():
            stages = nn.ModuleList(
                [
                    nn.Conv2d(ch, fused_channels[idx], kernel_size=1)
                    for idx, ch in enumerate(channels)
                ]
            )
            self.backbone_projections[name] = stages

        self.noise_backbone_names = set(noise_backbone_names or [])
        invalid = self.noise_backbone_names.difference(backbone_channels.keys())
        if invalid:
            raise ValueError(f"Unknown noise backbone names supplied: {sorted(invalid)}")
        self.noise_gates = nn.ModuleDict(
            {
                name: nn.ModuleList(
                    [ChannelGate(fused_channels[idx], gate_reduction) for idx in range(self.num_stages)]
                )
                for name in self.noise_backbone_names
            }
        )

        if self.fusion_mode == "concat":
            self.concat_projections = nn.ModuleList(
                [
                    nn.Conv2d(fused_channels[idx] * self.num_backbones, fused_channels[idx], kernel_size=1)
                    for idx in range(self.num_stages)
                ]
            )
        else:
            self.concat_projections = None

        self.stage_refiners = nn.ModuleList(
            [DepthwisePointwiseRefiner(fused_channels[idx]) for idx in range(self.num_stages)]
        )

    def forward(self, features: Dict[str, List[torch.Tensor]]) -> List[torch.Tensor]:
        """Fuse features per-scale by projecting and summing across backbones."""

        expected_sources = set(self.backbone_projections.keys())
        provided_sources = set(features.keys())
        if provided_sources != expected_sources:
            missing = expected_sources - provided_sources
            extra = provided_sources - expected_sources
            raise ValueError(
                "Feature dict keys do not match configured backbones; "
                f"missing={sorted(missing)} extra={sorted(extra)}"
            )

        fused_pyramid: List[torch.Tensor] = []
        for stage_idx in range(self.num_stages):
            projected = []
            for name, feat_list in features.items():
                stage_feat = feat_list[stage_idx]
                stage_proj_module = self.backbone_projections[name][stage_idx]
                expected_channels = stage_proj_module.in_channels
                if (
                    stage_feat.dim() == 4
                    and stage_feat.shape[1] != expected_channels
                    and stage_feat.shape[-1] == expected_channels
                ):
                    stage_feat = stage_feat.permute(0, 3, 1, 2).contiguous()
                stage_proj = stage_proj_module(stage_feat)
                if name in self.noise_gates:
                    stage_proj = self.noise_gates[name][stage_idx](stage_proj)
                projected.append(stage_proj)
            aligned = self._align_spatial(projected)
            if len(aligned) == 1:
                fused = aligned[0]
            elif self.fusion_mode == "sum":
                fused = torch.stack(aligned, dim=0).sum(dim=0)
            else:
                fused = torch.cat(aligned, dim=1)
                fused = self.concat_projections[stage_idx](fused)
            fused = self.stage_refiners[stage_idx](fused)
            fused_pyramid.append(fused)
        return fused_pyramid

    @staticmethod
    def _align_spatial(tensors: List[torch.Tensor]) -> List[torch.Tensor]:
        if len(tensors) <= 1:
            return tensors
        target_h = max(t.shape[-2] for t in tensors)
        target_w = max(t.shape[-1] for t in tensors)
        aligned = []
        for tensor in tensors:
            if tensor.shape[-2:] != (target_h, target_w):
                tensor = F.interpolate(
                    tensor,
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                )
            aligned.append(tensor)
        return aligned
