from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from backbones.efficientnet_b0 import EfficientNetB0Backbone
from backbones.residual_noise_branch import ResidualNoiseBackbone
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
    use_noise_branch: bool = True
    noise_branch_base_channels: int = 32
    noise_branch_num_stages: int = 4
    noise_branch_use_residual: bool = True
    noise_branch_use_high_pass: bool = True
    use_boundary_refiner: bool = True
    boundary_refiner_channels: int = 64
    backbone_input_size: int | tuple[int, int] = 320
    gradient_checkpointing: bool = False


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
            eff = EfficientNetB0Backbone(
                pretrained=self.config.pretrained_backbones,
                input_size=self.config.backbone_input_size,
            )
            self.backbones["efficientnet"] = eff
            backbone_channels["efficientnet"] = eff.feature_dims

        if self.config.use_swin:
            swin = SwinTinyBackbone(
                pretrained=self.config.pretrained_backbones,
                input_size=self.config.backbone_input_size,
            )
            self.backbones["swin"] = swin
            backbone_channels["swin"] = swin.feature_dims

        if self.config.use_segformer:
            segformer = SegFormerMiTBackbone(
                pretrained=self.config.pretrained_backbones,
                input_size=self.config.backbone_input_size,
            )
            self.backbones["segformer"] = segformer
            backbone_channels["segformer"] = segformer.feature_dims

        self.noise_branch: ResidualNoiseBackbone | None = None
        self._noise_residual_channels = 3 if self.config.noise_branch_use_residual else 0
        self._noise_high_pass_channels = 3 if self.config.noise_branch_use_high_pass else 0
        noise_in_channels = self._noise_residual_channels + self._noise_high_pass_channels
        if self.config.use_noise_branch:
            if noise_in_channels == 0:
                raise ValueError("Noise branch enabled but no residual/high-pass inputs selected in config.")
            if backbone_channels:
                num_stages = len(next(iter(backbone_channels.values())))
            else:
                num_stages = self.config.noise_branch_num_stages
            self.noise_branch = ResidualNoiseBackbone(
                in_channels=noise_in_channels,
                base_channels=self.config.noise_branch_base_channels,
                num_stages=num_stages,
            )
            backbone_channels["noise"] = self.noise_branch.feature_dims

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

        self.boundary_refiner: nn.Module | None = None
        if self.config.use_boundary_refiner:
            in_channels = 2  # mask logits + gradient magnitude
            self.boundary_refiner = nn.Sequential(
                nn.Conv2d(in_channels, self.config.boundary_refiner_channels, kernel_size=3, padding=1),
                nn.GroupNorm(4, self.config.boundary_refiner_channels),
                nn.GELU(),
                nn.Conv2d(self.config.boundary_refiner_channels, 1, kernel_size=3, padding=1),
            )
            sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32)
            sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32)
            self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3), persistent=False)
            self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3), persistent=False)

    def _extract_features(self, x: torch.Tensor) -> Dict[str, List[torch.Tensor]]:
        feature_dict: Dict[str, List[torch.Tensor]] = {}
        use_checkpoint = self.config.gradient_checkpointing and torch.is_grad_enabled()
        for name, backbone in self.backbones.items():
            if use_checkpoint:
                # Checkpointing trades a little compute for a sizable activation memory drop.
                def _forward_module(inp: torch.Tensor, module=backbone):
                    outputs = module(inp)
                    return tuple(outputs)

                try:
                    features = checkpoint(_forward_module, x, use_reentrant=False)
                except TypeError:
                    features = checkpoint(_forward_module, x)
                feature_dict[name] = [feat for feat in features]
            else:
                feature_dict[name] = backbone(x)
        return feature_dict

    def forward(
        self,
        x: torch.Tensor,
        noise_features: Dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        pyramid_features = self._extract_features(x)
        if self.noise_branch is not None:
            residual, high_pass = self._resolve_noise_inputs(x, noise_features)
            noise_pyramid = self.noise_branch(residual=residual, high_pass=high_pass)
            pyramid_features["noise"] = noise_pyramid
        fused = self.fusion(pyramid_features)
        if self.decoder is not None:
            mask_logits = self.decoder(fused)
        else:
            high_res = fused[0]
            if len(fused) > 1:
                for feat in fused[1:]:
                    high_res = high_res + F.interpolate(feat, size=high_res.shape[-2:], mode="bilinear", align_corners=False)
            mask_logits = self.simple_head(high_res)
        if self.boundary_refiner is not None:
            grad_mag = self._image_gradient_magnitude(x)
            ref_input = torch.cat([mask_logits, grad_mag], dim=1)
            mask_logits = mask_logits + self.boundary_refiner(ref_input)
        return mask_logits

    @torch.inference_mode()
    def predict_mask(
        self,
        x: torch.Tensor,
        threshold: float | None = None,
        noise_features: Dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Return raw logits by default. If `threshold` is provided, apply
        sigmoid to convert logits to probabilities and return a thresholded
        binary mask.

        This keeps the model output as logits (no sigmoid) for training and
        downstream losses that expect logits, while still supporting
        convenience thresholding for evaluation or inference code.
        """
        logits = self.forward(x, noise_features=noise_features)
        if threshold is None:
            return logits
        probs = torch.sigmoid(logits)
        return (probs > threshold).float()

    def _resolve_noise_inputs(
        self,
        image: torch.Tensor,
        noise_features: Dict[str, torch.Tensor] | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        residual = None
        high_pass = None
        if self._noise_residual_channels > 0:
            residual = (noise_features or {}).get("residual")
            if residual is None:
                residual = torch.zeros(
                    image.shape[0],
                    self._noise_residual_channels,
                    image.shape[-2],
                    image.shape[-1],
                    device=image.device,
                    dtype=image.dtype,
                )
        if self._noise_high_pass_channels > 0:
            high_pass = (noise_features or {}).get("high_pass")
            if high_pass is None:
                high_pass = torch.zeros(
                    image.shape[0],
                    self._noise_high_pass_channels,
                    image.shape[-2],
                    image.shape[-1],
                    device=image.device,
                    dtype=image.dtype,
                )
        return residual, high_pass

    def _image_gradient_magnitude(self, image: torch.Tensor) -> torch.Tensor:
        if not hasattr(self, "sobel_x") or self.boundary_refiner is None:
            return torch.zeros_like(image[:, :1])
        kernel_x = self.sobel_x.to(image.device, image.dtype).repeat(image.shape[1], 1, 1, 1)
        kernel_y = self.sobel_y.to(image.device, image.dtype).repeat(image.shape[1], 1, 1, 1)
        grad_x = F.conv2d(image, kernel_x, padding=1, groups=image.shape[1])
        grad_y = F.conv2d(image, kernel_y, padding=1, groups=image.shape[1])
        grad_mag = torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-6)
        return grad_mag.sum(dim=1, keepdim=True)
