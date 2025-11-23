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
    ) -> None:
        super().__init__()
        self.out_indices = out_indices
        self.model = timm.create_model(
            "segformer_b0",
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices,
        )
        self.feature_dims = self.model.feature_info.channels()

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return SegFormer multi-scale features (high → low resolution)."""
        features = self.model(x)
        return list(features)
