from __future__ import annotations

from typing import Iterable, Optional

import torch
from torch import nn


class LightweightConvBlock(nn.Module):
    """Depthwise-separable block with optional residual to minimize FLOPs."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion: float = 0.5,
        min_hidden_channels: int = 8,
    ) -> None:
        super().__init__()
        hidden_channels = max(int(out_channels * expansion), min_hidden_channels)
        self.use_residual = in_channels == out_channels

        reduce_layers: list[nn.Module] = []
        if in_channels != hidden_channels:
            reduce_layers.extend(
                [
                    nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(hidden_channels),
                    nn.GELU(),
                ]
            )
        else:
            reduce_layers.append(nn.Identity())

        self.reduce = nn.Sequential(*reduce_layers)
        self.depthwise = nn.Sequential(
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                groups=hidden_channels,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
        )
        self.project = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        identity = x
        x = self.reduce(x)
        x = self.depthwise(x)
        x = self.project(x)
        if self.use_residual:
            x = x + identity
        return self.activation(x)


class LightweightDownsampler(nn.Module):
    """Depthwise-strided reduction that preserves high-frequency cues."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)


class ChannelGate(nn.Module):
    """Lightweight squeeze-excitation gate applied per stage."""

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


class ResidualNoiseBackbone(nn.Module):
    """Lightweight CNN that turns residual/high-pass cues into pyramid features."""

    def __init__(
        self,
        in_channels: int,
        base_channels: int = 32,
        num_stages: int = 4,
        expansion: float = 0.5,
        min_hidden_channels: int = 8,
        use_stage_gating: bool = True,
        gate_reduction: int = 4,
    ) -> None:
        super().__init__()
        if num_stages < 1:
            raise ValueError("ResidualNoiseBackbone requires at least one stage.")
        if num_stages != 4:
            # Matching the UNet decoder layout typically requires four resolution levels
            raise ValueError("ResidualNoiseBackbone expects exactly four stages to match decoder skips.")
        self.num_stages = num_stages
        self.base_channels = base_channels
        self.input_channels = in_channels

        stage_channels = [base_channels * (2 ** idx) for idx in range(num_stages)]
        self.feature_dims = stage_channels

        self.input_projection = nn.Sequential(
            nn.Conv2d(in_channels, stage_channels[0], kernel_size=1, bias=False),
            nn.BatchNorm2d(stage_channels[0]),
            nn.GELU(),
        )

        blocks = []
        downsamplers = []
        stage_gates = []
        current_in = stage_channels[0]
        for idx, out_channels in enumerate(stage_channels):
            blocks.append(
                LightweightConvBlock(
                    current_in,
                    out_channels,
                    expansion=expansion,
                    min_hidden_channels=min_hidden_channels,
                )
            )
            gate = ChannelGate(out_channels, gate_reduction) if use_stage_gating else nn.Identity()
            stage_gates.append(gate)
            current_in = out_channels
            if idx < num_stages - 1:
                downsamplers.append(LightweightDownsampler(out_channels))
        self.blocks = nn.ModuleList(blocks)
        self.downsamplers = nn.ModuleList(downsamplers)
        self.stage_gates = nn.ModuleList(stage_gates)

    def forward(
        self,
        residual: Optional[torch.Tensor] = None,
        high_pass: Optional[torch.Tensor] = None,
        srm: Optional[torch.Tensor] = None,
        extra_tensors: Optional[Iterable[torch.Tensor]] = None,
    ) -> list[torch.Tensor]:
        x = self._build_input(residual, high_pass, srm, extra_tensors)
        x = self.input_projection(x)
        features: list[torch.Tensor] = []
        for idx, (block, gate) in enumerate(zip(self.blocks, self.stage_gates)):
            x = block(x)
            x = gate(x)
            features.append(x)
            if idx < len(self.downsamplers):
                x = self.downsamplers[idx](x)
        return features

    def _build_input(
        self,
        residual: Optional[torch.Tensor],
        high_pass: Optional[torch.Tensor],
        srm: Optional[torch.Tensor],
        extra_tensors: Optional[Iterable[torch.Tensor]],
    ) -> torch.Tensor:
        tensors = [t for t in (residual, high_pass, srm) if t is not None]
        if extra_tensors is not None:
            tensors.extend(list(extra_tensors))
        if not tensors:
            raise ValueError(
                "ResidualNoiseBackbone requires at least one auxiliary tensor (residual/high_pass/srm)."
            )
        spatial_shapes = {tensor.shape[-2:] for tensor in tensors}
        if len(spatial_shapes) != 1:
            raise ValueError("All noise tensors must share the same spatial resolution before concatenation.")
        x = tensors[0] if len(tensors) == 1 else torch.cat(tensors, dim=1)
        if x.shape[1] != self.input_channels:
            raise ValueError(
                f"Noise backbone expected {self.input_channels} channels after concatenation, got {x.shape[1]}."
            )
        return x
