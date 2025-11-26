import torch
import torch.nn.functional as F
from torch import nn
import timm
import logging

# Reduce noisy timm warnings when adapting pretrained weights to a modified
# model (e.g. different classifier / batchnorm buffers). The timm builder
# logs unexpected keys at WARNING level; raise to ERROR to keep console
# output cleaner during normal training runs.
try:
    logging.getLogger("timm.models._builder").setLevel(logging.ERROR)
except Exception:
    pass


class SwinTinyBackbone(nn.Module):
    """Swin Transformer Tiny backbone for capturing global context."""

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

        extra_args = {}
        if input_size is not None:
            if isinstance(input_size, int):
                input_hw = (input_size, input_size)
            else:
                input_hw = tuple(input_size)
            extra_args["img_size"] = input_hw
        else:
            input_hw = None

        self.model = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices,
            **extra_args,
        )
        self.feature_dims = self.model.feature_info.channels()
        cfg_size = self.model.default_cfg.get("input_size", (3, 224, 224))
        self.expected_hw = input_hw or cfg_size[1:]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return hierarchical Swin features (high → low resolution)."""

        if self.enforce_input_size and self.expected_hw and x.shape[-2:] != self.expected_hw:
            x = F.interpolate(x, size=self.expected_hw, mode="bilinear", align_corners=False)
        features = self.model(x)
        return list(features)
