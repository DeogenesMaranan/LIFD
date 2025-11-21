from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import nn


class DecoderBlock(nn.Module):
    """Single UNet-style upsampling block."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        use_skip: bool = True,
    ) -> None:
        super().__init__()
        self.use_skip = use_skip
        mid_channels = max(out_channels, in_channels)
        self.conv1 = nn.Conv2d(in_channels + (skip_channels if use_skip else 0), mid_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, 3, padding=1)
        self.norm1 = nn.BatchNorm2d(mid_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if self.use_skip and skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        x = self.act(self.norm1(self.conv1(x)))
        x = self.act(self.norm2(self.conv2(x)))
        return x


class UNetDecoder(nn.Module):
    """UNet decoder consuming fused pyramid features to output a mask."""

    def __init__(
        self,
        in_channels_list: List[int],
        use_skip_connections: bool = True,
        out_channels: int = 1,
    ) -> None:
        super().__init__()
        if len(in_channels_list) < 2:
            raise ValueError("UNetDecoder expects at least two feature scales.")

        self.use_skip_connections = use_skip_connections
        decoder_blocks = []
        for idx in range(len(in_channels_list) - 1, 0, -1):
            in_ch = in_channels_list[idx]
            skip_ch = in_channels_list[idx - 1]
            out_ch = skip_ch
            block = DecoderBlock(
                in_channels=in_ch,
                skip_channels=skip_ch,
                out_channels=out_ch,
                use_skip=use_skip_connections,
            )
            decoder_blocks.append(block)
        self.decoder_blocks = nn.ModuleList(decoder_blocks)
        self.mask_head = nn.Conv2d(in_channels_list[0], out_channels, kernel_size=1)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        x = features[-1]
        skip_features = features[:-1]
        for block, skip in zip(self.decoder_blocks, reversed(skip_features)):
            skip_tensor = skip if self.use_skip_connections else None
            x = block(x, skip_tensor)
        mask = self.mask_head(x)
        return mask
