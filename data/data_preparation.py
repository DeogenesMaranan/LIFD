"""Data preparation pipeline for hybrid forgery detection datasets (PyTorch-only).

This module converts raw datasets organised as
<dataset_root>/<dataset_name>/<real|fake|mask>/... into a preprocessed
structure that is cheap to load during training. All CPU-heavy image work
is performed once up front so the training loop can stream tensors quickly.
"""

from __future__ import annotations
import json
import math
import random
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
try:
    import imageio.v3 as iio
except ImportError:
    iio = None

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

import torch
from torch.utils.data import Dataset as TorchDataset

IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

try:
    RESAMPLE_BICUBIC = Image.Resampling.BICUBIC
    RESAMPLE_NEAREST = Image.Resampling.NEAREST
except AttributeError:
    RESAMPLE_BICUBIC = Image.BICUBIC
    RESAMPLE_NEAREST = Image.NEAREST


@dataclass
class DatasetStructureConfig:
    dataset_root: Path | str
    dataset_name: str
    real_subdir: str = "real"
    fake_subdir: str = "fake"
    mask_subdir: str = "mask"
    mask_suffix: str = ""
    prepared_root: Path | str = "prepared"

    def as_path(self) -> Path:
        return Path(self.dataset_root) / self.dataset_name if self.dataset_name else Path(self.dataset_root)

    def prepared_path(self) -> Path:
        return Path(self.prepared_root) / self.dataset_name


@dataclass
class SplitConfig:
    train: float = 0.7
    val: float = 0.15
    test: float = 0.15
    seed: int = 13

    def validate(self) -> None:
        total = self.train + self.val + self.test
        if not math.isclose(total, 1.0, rel_tol=1e-3):
            raise ValueError(f"Split percentages must add up to 1.0, got {total:.4f}")


@dataclass
class AugmentationConfig:
    enable: bool = True
    copies_per_sample: int = 1
    enable_flips: bool = True
    enable_rotations: bool = True
    enable_random_crop: bool = True
    enable_color_jitter: bool = True
    enable_noise: bool = True
    max_rotation_degrees: float = 10.0
    crop_scale_range: Tuple[float, float] = (0.9, 1.0)
    color_jitter_factors: Tuple[float, float] = (0.9, 1.1)
    noise_std_range: Tuple[float, float] = (0.0, 0.02)


@dataclass
class PreparationConfig:
    target_sizes: Tuple[int, ...] = (256,)
    normalization_mode: str = "zero_one"
    compute_high_pass: bool = True
    gaussian_radius: float = 1.0
    store_uint8_copy: bool = False

    def normalize(self, array_uint8: np.ndarray) -> np.ndarray:
        array = array_uint8.astype(np.float32)
        if self.normalization_mode == "zero_one":
            return array / 255.0
        if self.normalization_mode == "minus_one_one":
            return (array / 127.5) - 1.0
        raise ValueError("Unsupported normalization_mode. Use 'zero_one' or 'minus_one_one'.")


@dataclass
class SampleRecord:
    image_path: Path
    mask_path: Optional[Path]
    label: str


class DataPreparationPipeline:
    def __init__(self, structure: DatasetStructureConfig,
                 split: SplitConfig | None = None,
                 prep: PreparationConfig | None = None,
                 augment: AugmentationConfig | None = None) -> None:
        self.structure = structure
        self.split = split or SplitConfig()
        self.prep = prep or PreparationConfig()
        self.augment = augment or AugmentationConfig()
        self.split.validate()
        self._rng = random.Random(self.split.seed)
        self._aug_rng = random.Random(self.split.seed + 1234)
        self._prepared_root = self.structure.prepared_path()
        self._prepared_root.mkdir(parents=True, exist_ok=True)
        self._sample_counter = 0
        self._last_sample_log = self._prepared_root / "last_sample.log"

    def prepare(self) -> Dict[str, List[str]]:
        samples_by_label = self._collect_samples()
        split_map = self._split_samples(samples_by_label)
        manifest: Dict[str, List[str]] = {"train": [], "val": [], "test": []}

        print("Samples per split:", {k: len(v) for k, v in split_map.items()})

        total_samples = sum(len(samples) for samples in split_map.values())
        progress = tqdm(total=total_samples, desc="Preparing samples", unit="sample") if tqdm else None

        try:
            for split_name, samples in split_map.items():
                for sample in samples:
                    self._sample_counter += 1
                    self._log_sample_progress(split_name, sample)
                    try:
                        generated_files = self._process_sample(sample, split_name)
                    except Exception as exc:  # pragma: no cover - runtime diagnostic aid
                        if progress:
                            progress.close()
                        context = (
                            f"Failed while processing '{sample.image_path}'"
                            f" (label={sample.label}, split={split_name}).")
                        traceback.print_exc()
                        raise RuntimeError(context) from exc
                    manifest[split_name].extend(generated_files)
                    if progress:
                        progress.update(1)
        finally:
            if progress:
                progress.close()

        manifest_path = self._prepared_root / "manifest.json"
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump({k: [str(p) for p in v] for k, v in manifest.items()}, f, indent=2)
        return manifest

    # ----------------- Internal helper methods (unchanged) ----------------- #
    def _log_sample_progress(self, split_name: str, sample: SampleRecord) -> None:
        log_entry = {
            "sample_index": self._sample_counter,
            "split": split_name,
            "label": sample.label,
            "image": str(sample.image_path),
            "mask": str(sample.mask_path) if sample.mask_path else None,
        }
        self._last_sample_log.parent.mkdir(parents=True, exist_ok=True)
        self._last_sample_log.write_text(json.dumps(log_entry, indent=2))

    def _load_image(self, path: Optional[Path], mode: str) -> Optional[Image.Image]:
        if path is None:
            return None
        if not path.exists():
            return None
        suffix = path.suffix.lower()
        use_safe_loader = suffix in {".tif", ".tiff"}
        if use_safe_loader and iio is None:
            raise RuntimeError(
                f"Cannot safely load TIFF '{path}'. Install imageio to enable TIFF fallback.")
        if use_safe_loader and iio is not None:
            array = iio.imread(path)
            if array.ndim == 2 and mode.upper() not in {"L", "LA"}:
                array = np.stack([array] * 3, axis=-1)
            return Image.fromarray(array).convert(mode)
        with Image.open(path) as image:
            return image.convert(mode)
    def _collect_samples(self) -> Dict[str, List[SampleRecord]]:
        dataset_path = self.structure.as_path()
        real_dir = dataset_path / self.structure.real_subdir
        fake_dir = dataset_path / self.structure.fake_subdir
        mask_dir = dataset_path / self.structure.mask_subdir

        samples: Dict[str, List[SampleRecord]] = {"real": [], "fake": []}

        for image_path in self._iter_images(real_dir):
            samples["real"].append(SampleRecord(image_path=image_path, mask_path=None, label="real"))

        for image_path in self._iter_images(fake_dir):
            mask_path = self._resolve_mask_path(mask_dir, image_path)
            samples["fake"].append(SampleRecord(image_path=image_path, mask_path=mask_path, label="fake"))

        if not samples["real"] and not samples["fake"]:
            raise FileNotFoundError(
                f"No samples found beneath {dataset_path}. Check DatasetStructureConfig paths.")
        return samples

    def _iter_images(self, directory: Path) -> Iterator[Path]:
        if not directory.exists():
            return
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMG_EXTENSIONS:
                yield path

    def _resolve_mask_path(self, mask_dir: Path, image_path: Path) -> Optional[Path]:
        if not mask_dir.exists():
            return None
        candidates: List[Path] = []
        stem_with_suffix = image_path.stem + self.structure.mask_suffix
        candidates.append(mask_dir / f"{stem_with_suffix}{image_path.suffix}")
        for ext in IMG_EXTENSIONS:
            candidates.append(mask_dir / f"{stem_with_suffix}{ext}")
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _split_samples(self, samples_by_label: Dict[str, List[SampleRecord]]) -> Dict[str, List[SampleRecord]]:
        split_map: Dict[str, List[SampleRecord]] = {"train": [], "val": [], "test": []}
        for label, samples in samples_by_label.items():
            shuffled = samples.copy()
            self._rng.shuffle(shuffled)
            total = len(shuffled)
            if total == 0:
                continue
            train_count = int(total * self.split.train)
            val_count = int(total * self.split.val)
            test_count = max(0, total - train_count - val_count)
            train_split = shuffled[:train_count]
            val_split = shuffled[train_count:train_count + val_count]
            test_split = shuffled[train_count + val_count:train_count + val_count + test_count]
            split_map["train"].extend(train_split)
            split_map["val"].extend(val_split)
            split_map["test"].extend(test_split)
        return split_map

    def _process_sample(self, sample: SampleRecord, split_name: str) -> List[str]:
        generated: List[str] = []
        base_image = self._load_image(sample.image_path, mode="RGB")
        mask_image = self._load_image(sample.mask_path, mode="L") if sample.mask_path else None

        variants = self._apply_augmentations(base_image, mask_image)
        for variant_idx, (variant_image, variant_mask, variant_tag) in enumerate(variants):
            generated.extend(self._save_all_resolutions(
                variant_image, variant_mask, sample, split_name, variant_idx, variant_tag))
        return generated

    def _save_all_resolutions(self, image: Image.Image, mask: Optional[Image.Image],
                              sample: SampleRecord, split_name: str,
                              variant_idx: int, variant_tag: str) -> List[str]:
        paths: List[str] = []
        for target_size in self.prep.target_sizes:
            resized_image = ImageOps.fit(image, (target_size, target_size), method=RESAMPLE_BICUBIC)
            resized_mask = None
            if mask is not None:
                resized_mask = ImageOps.fit(mask, (target_size, target_size), method=RESAMPLE_NEAREST)
            saved_path = self._process_resolution(
                resized_image, resized_mask, sample, split_name, target_size, variant_idx, variant_tag)
            paths.append(str(saved_path))
        return paths

    def _process_resolution(self, image: Image.Image, mask: Optional[Image.Image],
                             sample: SampleRecord, split_name: str, target_size: int,
                             variant_idx: int, variant_tag: str) -> Path:
        image_uint8 = np.array(image, dtype=np.uint8)
        image_float = self.prep.normalize(image_uint8).astype(np.float32)

        arrays: Dict[str, Any] = {"image": image_float}
        if self.prep.store_uint8_copy:
            arrays["image_uint8"] = image_uint8

        if mask is not None:
            mask_arr = np.array(mask, dtype=np.uint8)
            mask_arr = mask_arr[:, :, None] if mask_arr.ndim == 2 else mask_arr
            arrays["mask"] = (mask_arr.astype(np.float32) / 255.0)

        if self.prep.compute_high_pass:
            high_pass, residual = self._compute_high_pass_features(image_uint8)
            arrays["high_pass"] = high_pass
            arrays["residual"] = residual

        meta = {
            "label": sample.label,
            "split": split_name,
            "source_image": str(sample.image_path),
            "mask_path": str(sample.mask_path) if sample.mask_path else None,
            "target_size": target_size,
            "variant_idx": variant_idx,
            "variant_tag": variant_tag,
        }
        arrays["meta"] = np.array(json.dumps(meta), dtype=object)

        output_path = self._build_output_path(split_name, sample, target_size, variant_tag)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, **arrays)
        return output_path

    def _build_output_path(self, split_name: str, sample: SampleRecord,
                           target_size: int, variant_tag: str) -> Path:
        safe_variant = variant_tag.replace(" ", "_")
        split_dir = self._prepared_root / split_name / sample.label
        filename = f"{sample.image_path.stem}_{target_size}px_{safe_variant}.npz"
        return split_dir / filename

    def _compute_high_pass_features(self, image_uint8: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        image_float = image_uint8.astype(np.float32)
        pil_image = Image.fromarray(image_uint8)
        blurred = pil_image.filter(ImageFilter.GaussianBlur(radius=max(0.1, self.prep.gaussian_radius)))
        blurred_arr = np.array(blurred, dtype=np.float32)
        residual = image_float - blurred_arr
        high_pass = np.abs(residual)
        if np.max(high_pass) > 0:
            high_pass = high_pass / 255.0
        residual = residual / 255.0
        return high_pass.astype(np.float32), residual.astype(np.float32)

    def _apply_augmentations(self, image: Image.Image, mask: Optional[Image.Image]) -> List[Tuple[Image.Image, Optional[Image.Image], str]]:
        variants: List[Tuple[Image.Image, Optional[Image.Image], str]] = [(image.copy(), mask.copy() if mask else None, "orig")]
        if not self.augment.enable:
            return variants

        for idx in range(self.augment.copies_per_sample):
            aug_img = image.copy()
            aug_mask = mask.copy() if mask else None
            tag_parts: List[str] = []

            if self.augment.enable_flips and self._aug_rng.random() < 0.5:
                aug_img = ImageOps.mirror(aug_img)
                if aug_mask:
                    aug_mask = ImageOps.mirror(aug_mask)
                tag_parts.append("hflip")

            if self.augment.enable_flips and self._aug_rng.random() < 0.2:
                aug_img = ImageOps.flip(aug_img)
                if aug_mask:
                    aug_mask = ImageOps.flip(aug_mask)
                tag_parts.append("vflip")

            if self.augment.enable_rotations and self.augment.max_rotation_degrees > 0:
                angle = self._aug_rng.uniform(-self.augment.max_rotation_degrees, self.augment.max_rotation_degrees)
                if abs(angle) > 1e-3:
                    aug_img = aug_img.rotate(angle, resample=RESAMPLE_BICUBIC, expand=False, fillcolor=0)
                    if aug_mask:
                        aug_mask = aug_mask.rotate(angle, resample=RESAMPLE_NEAREST, expand=False, fillcolor=0)
                    tag_parts.append(f"rot{int(angle)}")

            if self.augment.enable_random_crop:
                aug_img, aug_mask, cropped = self._random_crop(aug_img, aug_mask, self.augment.crop_scale_range)
                if cropped:
                    tag_parts.append("crop")

            if self.augment.enable_color_jitter:
                aug_img = self._color_jitter(aug_img)
                tag_parts.append("jitter")

            if self.augment.enable_noise:
                aug_img = self._add_noise(aug_img)
                tag_parts.append("noise")

            variant_tag = "aug" + str(idx + 1)
            if tag_parts:
                variant_tag += "_" + "-".join(tag_parts)
            variants.append((aug_img, aug_mask, variant_tag))
        return variants

    def _random_crop(self, image: Image.Image, mask: Optional[Image.Image],
                     scale_range: Tuple[float, float]) -> Tuple[Image.Image, Optional[Image.Image], bool]:
        min_scale, max_scale = scale_range
        min_scale = max(0.1, min_scale)
        max_scale = min(1.0, max_scale)
        if min_scale >= max_scale:
            return image, mask, False
        scale = self._aug_rng.uniform(min_scale, max_scale)
        if math.isclose(scale, 1.0, rel_tol=1e-3):
            return image, mask, False
        w, h = image.size
        crop_w = max(1, int(w * scale))
        crop_h = max(1, int(h * scale))
        if crop_w == w and crop_h == h:
            return image, mask, False
        max_left = w - crop_w
        max_top = h - crop_h
        left = self._aug_rng.randint(0, max(0, max_left))
        top = self._aug_rng.randint(0, max(0, max_top))
        box = (left, top, left + crop_w, top + crop_h)
        cropped_image = image.crop(box)
        cropped_mask = mask.crop(box) if mask else None
        return cropped_image, cropped_mask, True

    def _color_jitter(self, image: Image.Image) -> Image.Image:
        factor_range = self.augment.color_jitter_factors
        brightness = ImageEnhance.Brightness(image).enhance(
            self._aug_rng.uniform(*factor_range))
        contrast = ImageEnhance.Contrast(brightness).enhance(
            self._aug_rng.uniform(*factor_range))
        colorized = ImageEnhance.Color(contrast).enhance(
            self._aug_rng.uniform(*factor_range))
        return colorized

    def _add_noise(self, image: Image.Image) -> Image.Image:
        std_min, std_max = self.augment.noise_std_range
        std = self._aug_rng.uniform(std_min, std_max)
        if std <= 0:
            return image
        arr = np.array(image, dtype=np.float32)
        rng = np.random.default_rng(self._aug_rng.randrange(0, 2**32 - 1))
        noise = rng.normal(0.0, std * 255.0, size=arr.shape)
        noised = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noised)

# ----------------- PyTorch Dataset Loader ----------------- #
class PreparedForgeryDataset(TorchDataset):
    def __init__(self, prepared_root: Path | str, split: str, target_size: int,
                 include_features: bool = True, return_masks: bool = True) -> None:
        self.prepared_root = Path(prepared_root)
        self.split = split
        self.target_size = target_size
        pattern = f"*_{target_size}px_*.npz"
        self.files = sorted((self.prepared_root / split).rglob(pattern))
        if not self.files:
            raise FileNotFoundError(f"No preprocessed samples found for split='{split}', size={target_size}.")
        self.include_features = include_features
        self.return_masks = return_masks

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        path = self.files[index]
        with np.load(path, allow_pickle=True) as data:
            image = torch.from_numpy(data["image"].transpose(2, 0, 1))
            sample: Dict[str, Any] = {"image": image}
            if self.return_masks and "mask" in data.files:
                mask = torch.from_numpy(data["mask"].transpose(2, 0, 1))
                sample["mask"] = mask
            if self.include_features:
                if "high_pass" in data.files and data["high_pass"].size:
                    sample["high_pass"] = torch.from_numpy(data["high_pass"].transpose(2, 0, 1))
                if "residual" in data.files and data["residual"].size:
                    sample["residual"] = torch.from_numpy(data["residual"].transpose(2, 0, 1))
            meta = json.loads(str(data["meta"].item())) if "meta" in data.files else {}
            sample["label"] = torch.tensor(1 if meta.get("label") == "fake" else 0)
            sample["meta"] = meta
            return sample

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    structure = DatasetStructureConfig(
        dataset_root="./datasets",
        dataset_name="CASIA2",
        real_subdir="real",
        fake_subdir="fake",
        mask_subdir="mask",
        mask_suffix="_gt",
        prepared_root="prepared",
    )

    split = SplitConfig(
        train=0.7, 
        val=0.15, 
        test=0.15, 
        seed=42)
    
    prep = PreparationConfig(
        target_sizes=(128,), 
        normalization_mode="zero_one", 
        compute_high_pass=True)
    
    augment = AugmentationConfig(
        enable=True, 
        copies_per_sample=2, 
        max_rotation_degrees=15,
        crop_scale_range=(0.8, 1.0), 
        noise_std_range=(0.0, 0.03))

    pipeline = DataPreparationPipeline(structure, split, prep, augment)
    manifest = pipeline.prepare()
    print("Preparation finished.")
    for split_name, files in manifest.items():
        print(f"{split_name}: {len(files)} artifacts")