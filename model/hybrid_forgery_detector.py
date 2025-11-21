from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import nn

from backbones.efficientnet_b0 import EfficientNetB0Backbone
from backbones.segformer_mit import SegFormerMiTBackbone
from backbones.swin_transformer import SwinTinyBackbone
from decoder.unet_decoder import UNetDecoder
from fusion.feature_fusion import FeatureFusionModule


@dataclass
class HybridForgeryConfig:
    """Configuration flags describing which components to enable for ablations."""

    use_efficientnet: bool = True
    use_swin: bool = True
    use_segformer: bool = False
    use_unet_decoder: bool = True
    use_skip_connections: bool = True
    pretrained_backbones: bool = True
    fused_channels: int = 256


class HybridForgeryDetector(nn.Module):
    """Modular architecture for image forgery segmentation with flexible ablations.

    Ablation hints:
    - EfficientNet only: enable only use_efficientnet.
    - Swin only: enable only use_swin.
    - SegFormer only: enable only use_segformer.
    - EfficientNet + Swin: enable both use_efficientnet and use_swin.
    - EfficientNet + SegFormer: enable use_efficientnet and use_segformer.
    - No UNet decoder: set use_unet_decoder=False to fall back to bilinear upsampling.
    """

    def __init__(self, config: HybridForgeryConfig | None = None) -> None:
        super().__init__()
        self.config = config or HybridForgeryConfig()

        self.backbones = nn.ModuleDict()
        backbone_channels: Dict[str, List[int]] = {}

        if self.config.use_efficientnet:
            eff = EfficientNetB0Backbone(pretrained=self.config.pretrained_backbones)
            self.backbones["efficientnet"] = eff
            backbone_channels["efficientnet"] = eff.feature_dims

        if self.config.use_swin:
            swin = SwinTinyBackbone(pretrained=self.config.pretrained_backbones)
            self.backbones["swin"] = swin
            backbone_channels["swin"] = swin.feature_dims

        if self.config.use_segformer:
            segformer = SegFormerMiTBackbone(pretrained=self.config.pretrained_backbones)
            self.backbones["segformer"] = segformer
            backbone_channels["segformer"] = segformer.feature_dims

        if not backbone_channels:
            raise ValueError("At least one backbone must be enabled in the config.")

        self.fusion = FeatureFusionModule(backbone_channels, fused_channels=self.config.fused_channels)
        self.fused_channels = self.fusion.fused_channels

        if self.config.use_unet_decoder:
            self.decoder = UNetDecoder(
                in_channels_list=self.fused_channels,
                use_skip_connections=self.config.use_skip_connections,
                out_channels=1,
            )
            self.simple_head = None
        else:
            self.decoder = None
            self.simple_head = nn.Sequential(
                nn.Conv2d(self.fused_channels[0], self.fused_channels[0], kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.fused_channels[0], 1, kernel_size=1),
            )

    def _extract_features(self, x: torch.Tensor) -> Dict[str, List[torch.Tensor]]:
        feature_dict: Dict[str, List[torch.Tensor]] = {}
        for name, backbone in self.backbones.items():
            feature_dict[name] = backbone(x)
        return feature_dict

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pyramid_features = self._extract_features(x)
        fused = self.fusion(pyramid_features)
        if self.decoder is not None:
            mask_logits = self.decoder(fused)
        else:
            high_res = fused[0]
            if len(fused) > 1:
                for feat in fused[1:]:
                    high_res = high_res + F.interpolate(feat, size=high_res.shape[-2:], mode="bilinear", align_corners=False)
            mask_logits = self.simple_head(high_res)
        return mask_logits

    @torch.inference_mode()
    def predict_mask(self, x: torch.Tensor, threshold: float | None = None) -> torch.Tensor:
        logits = self.forward(x)
        probs = torch.sigmoid(logits)
        if threshold is None:
            return probs
        return (probs > threshold).float()
