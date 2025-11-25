import torch
import torch.nn.functional as F
from torch import nn
import timm


class EfficientNetB0Backbone(nn.Module):
    """EfficientNet-B0 feature extractor returning multi-scale features."""

    def __init__(
        self,
        pretrained: bool = True,
        out_indices=(0, 1, 2, 3),
        enforce_input_size: bool = False,
        input_size: int | tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        self.out_indices = out_indices
        self.enforce_input_size = enforce_input_size
        if input_size is not None:
            if isinstance(input_size, int):
                input_hw = (input_size, input_size)
            else:
                input_hw = tuple(input_size)
        else:
            input_hw = None

        # EfficientNet timm implementation does not accept `img_size` in all variants,
        # so avoid passing it to `create_model`. We still honour the provided
        # `input_size` by setting `expected_hw` so `enforce_input_size` can work.
        self.model = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices,
        )
        self.feature_dims = self.model.feature_info.channels()
        cfg_size = self.model.default_cfg.get("input_size", (3, 224, 224))
        self.expected_hw = input_hw or cfg_size[1:]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return feature pyramid ordered from high to low resolution."""

        if self.enforce_input_size and self.expected_hw and x.shape[-2:] != self.expected_hw:
            x = F.interpolate(x, size=self.expected_hw, mode="bilinear", align_corners=False)
        features = self.model(x)
        return list(features)
