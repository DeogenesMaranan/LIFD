from __future__ import annotations

from typing import Dict, List

import torch
from torch import nn
import torch.nn.functional as F


class FeatureFusionModule(nn.Module):
    """Fuse multi-scale feature pyramids originating from different backbones."""

    def __init__(
        self,
        backbone_channels: Dict[str, List[int]],
        fused_channels: int | List[int] = 256,
    ) -> None:
        super().__init__()
        if not backbone_channels:
            raise ValueError("At least one backbone must be provided for feature fusion.")

        self.num_stages = len(next(iter(backbone_channels.values())))
        for channels in backbone_channels.values():
            if len(channels) != self.num_stages:
                raise ValueError("All backbones must expose the same number of feature stages.")

        if isinstance(fused_channels, int):
            fused_channels = [fused_channels] * self.num_stages
        if len(fused_channels) != self.num_stages:
            raise ValueError("fused_channels must match the number of feature stages.")

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

        self.stage_fusers = nn.ModuleList(
            [
                nn.Conv2d(fused_channels[idx], fused_channels[idx], kernel_size=3, padding=1)
                for idx in range(self.num_stages)
            ]
        )

    def forward(self, features: Dict[str, List[torch.Tensor]]) -> List[torch.Tensor]:
        """Fuse features per-scale by projecting and summing across backbones."""

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
                stage_proj = self.backbone_projections[name][stage_idx](stage_feat)
                projected.append(stage_proj)
            if len(projected) == 1:
                stacked = projected[0][None, ...]
            else:
                # Align spatial dimensions before stacking, target = max height/width for this scale
                target_h = max(t.shape[-2] for t in projected)
                target_w = max(t.shape[-1] for t in projected)
                aligned = []
                for tensor in projected:
                    if tensor.shape[-2:] != (target_h, target_w):
                        tensor = F.interpolate(
                            tensor,
                            size=(target_h, target_w),
                            mode="bilinear",
                            align_corners=False,
                        )
                    aligned.append(tensor)
                stacked = torch.stack(aligned)
            fused = torch.sum(stacked, dim=0)
            fused = self.stage_fusers[stage_idx](fused)
            fused_pyramid.append(fused)
        return fused_pyramid
