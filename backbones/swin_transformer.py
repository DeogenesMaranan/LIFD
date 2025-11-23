import torch
import torch.nn.functional as F
from torch import nn
import timm


class SwinTinyBackbone(nn.Module):
    """Swin Transformer Tiny backbone for capturing global context."""

    def __init__(
        self,
        pretrained: bool = True,
        out_indices=(0, 1, 2, 3),
    ) -> None:
        super().__init__()
        self.out_indices = out_indices
        self.model = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices,
        )
        self.feature_dims = self.model.feature_info.channels()

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return hierarchical Swin features (high → low resolution)."""
        features = self.model(x)
        return list(features)
