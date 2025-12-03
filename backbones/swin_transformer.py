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
        else:
            self._maybe_update_patch_embed_shape(x)
        features = self.model(x)
        return list(features)

    def _maybe_update_patch_embed_shape(self, x: torch.Tensor) -> None:
        """Update the internal patch embedding shape so Swin accepts arbitrary inputs."""

        patch_embed = getattr(self.model, "patch_embed", None)
        if patch_embed is None:
            return
        h, w = x.shape[-2:]
        current_size = getattr(patch_embed, "img_size", None)
        new_size = (h, w)
        if current_size == new_size:
            return
        patch_embed.img_size = new_size
        patch_size = getattr(patch_embed, "patch_size", (4, 4))
        if hasattr(patch_embed, "grid_size") and patch_size is not None:
            patch_h = patch_size[0]
            patch_w = patch_size[1]
            if h % patch_h != 0 or w % patch_w != 0:
                raise ValueError(
                    "Input spatial dims must be multiples of the Swin patch size (4) for variable resizing."
                )
            grid_size = (h // patch_h, w // patch_w)
            patch_embed.grid_size = grid_size
            if hasattr(patch_embed, "num_patches"):
                patch_embed.num_patches = grid_size[0] * grid_size[1]

        self._update_stage_resolutions(new_size)

    def _update_stage_resolutions(self, spatial_size: tuple[int, int]) -> None:
        patch_embed = getattr(self.model, "patch_embed", None)
        if patch_embed is None:
            return
        patch_size = getattr(patch_embed, "patch_size", (4, 4))
        feat_size = (spatial_size[0] // patch_size[0], spatial_size[1] // patch_size[1])
        idx = 0
        while True:
            stage = getattr(self.model, f"layers_{idx}", None)
            if stage is None:
                break
            window = stage.blocks[0].window_size[0] if stage.blocks else 7
            stage.set_input_size(feat_size, window_size=window)
            feat_size = stage.output_resolution
            idx += 1
