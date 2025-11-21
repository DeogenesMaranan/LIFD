from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F
from torch import nn


class FPNDecoder(nn.Module):
    """Feature Pyramid Network decoder for segmentation ablations."""

    def __init__(
        self,
        in_channels_list: List[int],
        fpn_channels: int = 256,
        out_channels: int = 1,
    ) -> None:
        super().__init__()
        if len(in_channels_list) < 2:
            raise ValueError("FPNDecoder expects at least two feature stages.")

        self.lateral_convs = nn.ModuleList(
            [nn.Conv2d(ch, fpn_channels, kernel_size=1) for ch in in_channels_list]
        )
        self.output_convs = nn.ModuleList(
            [
                nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1)
                for _ in in_channels_list
            ]
        )
        self.mask_head = nn.Sequential(
            nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(fpn_channels, out_channels, kernel_size=1),
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        # Expect features ordered high → low resolution.
        laterals = [lateral(feat) for lateral, feat in zip(self.lateral_convs, features)]
        pyramid = laterals.copy()

        for idx in range(len(pyramid) - 1, 0, -1):
            upsampled = F.interpolate(pyramid[idx], size=pyramid[idx - 1].shape[-2:], mode="bilinear", align_corners=False)
            pyramid[idx - 1] = pyramid[idx - 1] + upsampled

        pyramid = [conv(feat) for conv, feat in zip(self.output_convs, pyramid)]
        out = pyramid[0]
        for feat in pyramid[1:]:
            upsampled = F.interpolate(feat, size=out.shape[-2:], mode="bilinear", align_corners=False)
            out = out + upsampled
        mask = self.mask_head(out)
        return mask
