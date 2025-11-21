from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)


class ResidualNoiseBackbone(nn.Module):
    """Lightweight CNN that turns residual/high-pass cues into pyramid features."""

    def __init__(
        self,
        in_channels: int,
        base_channels: int = 32,
        num_stages: int = 4,
    ) -> None:
        super().__init__()
        if num_stages < 1:
            raise ValueError("ResidualNoiseBackbone requires at least one stage.")
        self.num_stages = num_stages
        self.base_channels = base_channels
        self.input_channels = in_channels

        stage_channels = [base_channels * (2 ** idx) for idx in range(num_stages)]
        self.feature_dims = stage_channels

        blocks = []
        downsamplers = []
        current_in = in_channels
        for idx, out_channels in enumerate(stage_channels):
            blocks.append(ConvBlock(current_in, out_channels))
            current_in = out_channels
            if idx < num_stages - 1:
                downsamplers.append(
                    nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True),
                    )
                )
        self.blocks = nn.ModuleList(blocks)
        self.downsamplers = nn.ModuleList(downsamplers)

    def forward(
        self,
        residual: Optional[torch.Tensor] = None,
        high_pass: Optional[torch.Tensor] = None,
    ) -> list[torch.Tensor]:
        x = self._build_input(residual, high_pass)
        features: list[torch.Tensor] = []
        for idx, block in enumerate(self.blocks):
            x = block(x)
            features.append(x)
            if idx < len(self.downsamplers):
                x = self.downsamplers[idx](x)
        return features

    def _build_input(
        self,
        residual: Optional[torch.Tensor],
        high_pass: Optional[torch.Tensor],
    ) -> torch.Tensor:
        tensors = []
        if residual is not None:
            tensors.append(residual)
        if high_pass is not None:
            tensors.append(high_pass)
        if not tensors:
            raise ValueError("ResidualNoiseBackbone requires at least one auxiliary tensor (residual/high_pass).")
        if len(tensors) == 1:
            x = tensors[0]
        else:
            x = torch.cat(tensors, dim=1)
        if x.shape[1] != self.input_channels:
            if x.shape[1] < self.input_channels:
                pad = self.input_channels - x.shape[1]
                padding = torch.zeros(x.shape[0], pad, x.shape[2], x.shape[3], device=x.device, dtype=x.dtype)
                x = torch.cat([x, padding], dim=1)
            else:
                x = x[:, : self.input_channels]
        return x
