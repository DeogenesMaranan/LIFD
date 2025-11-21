import torch
import torch.nn.functional as F
from torch import nn
import timm


class SegFormerMiTBackbone(nn.Module):
    """SegFormer MiT-B0 backbone used as an alternate transformer encoder."""

    def __init__(
        self,
        pretrained: bool = True,
        out_indices=(0, 1, 2, 3),
        enforce_input_size: bool = False,
    ) -> None:
        super().__init__()
        self.out_indices = out_indices
        self.enforce_input_size = enforce_input_size
        self.model = timm.create_model(
            "segformer_b0",
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices,
        )
        self.feature_dims = self.model.feature_info.channels()
        input_size = self.model.default_cfg.get("input_size", (3, 224, 224))
        self.expected_hw = input_size[1:]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return SegFormer multi-scale features (high → low resolution)."""

        if self.enforce_input_size and self.expected_hw and x.shape[-2:] != self.expected_hw:
            x = F.interpolate(x, size=self.expected_hw, mode="bilinear", align_corners=False)
        features = self.model(x)
        return list(features)
