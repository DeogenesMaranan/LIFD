"""Data preparation pipeline for hybrid forgery detection datasets (PyTorch-only).

This module converts raw datasets organised as
<dataset_root>/<dataset_name>/<real|fake|mask>/... into a preprocessed
structure that is cheap to load during training. All CPU-heavy image work
is performed once up front so the training loop can stream tensors quickly.
"""

from __future__ import annotations
import io
import json
import math
import os
import random
import traceback
import tarfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
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
MANIFEST_FILENAME = "manifest.parquet"

try:
    RESAMPLE_BICUBIC = Image.Resampling.BICUBIC
    RESAMPLE_NEAREST = Image.Resampling.NEAREST
except AttributeError:
    RESAMPLE_BICUBIC = Image.BICUBIC
    RESAMPLE_NEAREST = Image.NEAREST


@dataclass
class DatasetStructureConfig:
    dataset_root: Path | str
    dataset_names: List[str] | str
    real_subdir: str = "real"
    fake_subdir: str = "fake"
    mask_subdir: str = "mask"
    mask_suffix: str = ""
    prepared_root: Path | str = "prepared"

    def get_dataset_names(self) -> List[str]:
        if isinstance(self.dataset_names, str):
            return [self.dataset_names]
        return list(self.dataset_names)

    def as_path(self, dataset_name: str) -> Path:
        return Path(self.dataset_root) / dataset_name if dataset_name else Path(self.dataset_root)

    def prepared_path(self, dataset_name: str) -> Path:
        return Path(self.prepared_root) / dataset_name


@dataclass
class SplitConfig:
    train: float = 0.7
    val: float = 0.15
    test: float = 0.15
    seed: int = 13
    enable_test_split: bool = True

    def validate(self) -> None:
        total = self.train + self.val + (self.test if self.enable_test_split else 0.0)
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
    target_sizes: Tuple[int, ...] = (384,)
    normalization_mode: str = "zero_one"
    compute_high_pass: bool = True
    gaussian_radius: float = 1.0
    store_uint8_copy: bool = False
    storage_dtype: np.dtype | str = np.float16

    def __post_init__(self) -> None:
        self.storage_dtype = np.dtype(self.storage_dtype)

    def normalize(self, array_uint8: np.ndarray) -> np.ndarray:
        array = array_uint8.astype(np.float32)
        if self.normalization_mode == "zero_one":
            return array / 255.0
        if self.normalization_mode == "minus_one_one":
            return (array / 127.5) - 1.0
        raise ValueError("Unsupported normalization_mode. Use 'zero_one' or 'minus_one_one'.")

    def cast_for_storage(self, array: np.ndarray) -> np.ndarray:
        if array.dtype == self.storage_dtype:
            return array
        return array.astype(self.storage_dtype)


@dataclass
class SampleRecord:
    image_path: Path
    mask_path: Optional[Path]
    label: str


class DataPreparationPipeline:
    def __init__(self, structure: DatasetStructureConfig,
                 split: SplitConfig | None = None,
                 prep: PreparationConfig | None = None,
                 augment: AugmentationConfig | None = None,
                 max_workers: Optional[int] = None) -> None:
        self.structure = structure
        self.split = split or SplitConfig()
        self.prep = prep or PreparationConfig()
        self.augment = augment or AugmentationConfig()
        self.split.validate()
        cpu_count = os.cpu_count() or 1
        if max_workers is None:
            self.max_workers = cpu_count
        else:
            self.max_workers = max(1, min(max_workers, cpu_count))
        self._rng = random.Random(self.split.seed)
        self._aug_seed_base = self.split.seed + 1234
        self._sample_counter = 0
        self._last_sample_log = None

    def prepare(self, test_only: bool = False) -> Dict[str, pd.DataFrame]:
        """
        If test_only is True, only process the test split for all datasets.
        Returns a dict mapping dataset_name to manifest DataFrame.
        """
        results = {}
        for dataset_name in self.structure.get_dataset_names():
            prepared_root = self.structure.prepared_path(dataset_name)
            prepared_root.mkdir(parents=True, exist_ok=True)
            self._prepared_root = prepared_root
            self._last_sample_log = prepared_root / "last_sample.log"

            samples_by_label = self._collect_samples(dataset_name)
            split_map = self._split_samples(samples_by_label)
            if not self.split.enable_test_split:
                split_map.pop("test", None)
            if test_only:
                split_map = {"test": split_map.get("test", [])}

            print(f"[{dataset_name}] Samples per split:", {k: len(v) for k, v in split_map.items()})

            total_samples = sum(len(samples) for samples in split_map.values())
            progress = tqdm(total=total_samples, desc=f"Preparing samples [{dataset_name}]", unit="sample") if tqdm else None

            executor = ThreadPoolExecutor(max_workers=self.max_workers) if self.max_workers > 1 else None
            manifest_entries: List[Dict[str, Any]] = []
            try:
                for split_name, samples in split_map.items():
                    if not samples:
                        continue
                    if executor is None:
                        for sample in samples:
                            self._sample_counter += 1
                            seed = self._aug_seed_base + self._sample_counter
                            self._log_sample_progress(split_name, sample)
                            try:
                                generated_entries = self._process_sample(sample, split_name, seed)
                            except Exception as exc:
                                if progress:
                                    progress.close()
                                context = (
                                    f"Failed while processing '{sample.image_path}'"
                                    f" (label={sample.label}, split={split_name}).")
                                traceback.print_exc()
                                raise RuntimeError(context) from exc
                            manifest_entries.extend(generated_entries)
                            if progress:
                                progress.update(1)
                    else:
                        future_map = {}
                        for sample in samples:
                            self._sample_counter += 1
                            seed = self._aug_seed_base + self._sample_counter
                            self._log_sample_progress(split_name, sample)
                            future = executor.submit(self._process_sample, sample, split_name, seed)
                            future_map[future] = sample
                        for future in as_completed(future_map):
                            sample = future_map[future]
                            try:
                                generated_entries = future.result()
                            except Exception as exc:
                                if progress:
                                    progress.close()
                                context = (
                                    f"Failed while processing '{sample.image_path}'"
                                    f" (label={sample.label}, split={split_name}).")
                                traceback.print_exc()
                                raise RuntimeError(context) from exc
                            manifest_entries.extend(generated_entries)
                            if progress:
                                progress.update(1)
            finally:
                if progress:
                    progress.close()
                if executor:
                    executor.shutdown(wait=True)

            self._create_tar_shards(manifest_entries)

            manifest_df = pd.DataFrame(manifest_entries)
            manifest_path = prepared_root / MANIFEST_FILENAME
            manifest_df.to_parquet(manifest_path, index=False)

            split_counts = manifest_df.groupby("split").size().to_dict() if not manifest_df.empty else {}
            print(f"[{dataset_name}] Artifacts per split:", split_counts)

            results[dataset_name] = manifest_df
        return results

    # ----------------- TAR Sharding ----------------- #
    def _create_tar_shards(self, manifest_entries: List[Dict[str, Any]], shard_size: int = 500) -> None:
        if not manifest_entries:
            print("No artifacts to shard.")
            return

        entry_by_rel_path = {
            entry.get("relative_path"): entry
            for entry in manifest_entries
            if entry.get("relative_path")
        }

        print("Creating TAR shards...")
        for split_name in ["train", "val", "test"]:
            split_dir = self._prepared_root / split_name
            if not split_dir.exists():
                continue

            for stale_tar in split_dir.glob("shard_*.tar"):
                stale_tar.unlink()

            npz_files = sorted(split_dir.rglob("*.npz"))
            if not npz_files:
                continue

            shard_idx = 0
            for i in range(0, len(npz_files), shard_size):
                shard_files = npz_files[i:i + shard_size]
                tar_path = split_dir / f"shard_{shard_idx:04d}.tar"
                tar_rel_path = str(tar_path.relative_to(self._prepared_root))
                with tarfile.open(tar_path, "w") as tar:
                    for npz_file in shard_files:
                        rel_path = str(npz_file.relative_to(self._prepared_root))
                        tar.add(npz_file, arcname=npz_file.name)
                        entry = entry_by_rel_path.get(rel_path)
                        if entry is not None:
                            entry["tar_path"] = tar_rel_path
                            entry["tar_member"] = npz_file.name
                        npz_file.unlink()
                shard_idx += 1
                print(f"{split_name}: created {tar_path} with {len(shard_files)} samples")
        print("All TAR shards created.")

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
        if path is None or not path.exists():
            return None
        suffix = path.suffix.lower()
        use_safe_loader = suffix in {".tif", ".tiff"}
        if use_safe_loader and iio is None:
            raise RuntimeError(f"Cannot safely load TIFF '{path}'. Install imageio to enable TIFF fallback.")
        if use_safe_loader and iio is not None:
            array = iio.imread(path)
            if array.ndim == 2 and mode.upper() not in {"L", "LA"}:
                array = np.stack([array] * 3, axis=-1)
            return Image.fromarray(array).convert(mode)
        with Image.open(path) as image:
            return image.convert(mode)

    def _collect_samples(self, dataset_name: str) -> Dict[str, List[SampleRecord]]:
        dataset_path = self.structure.as_path(dataset_name)
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
            raise FileNotFoundError(f"No samples found beneath {dataset_path}.")
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
        stem_with_suffix = image_path.stem + self.structure.mask_suffix
        for ext in [image_path.suffix] + list(IMG_EXTENSIONS):
            candidate = mask_dir / f"{stem_with_suffix}{ext}"
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
            test_count = max(0, total - train_count - val_count) if self.split.enable_test_split else 0
            split_map["train"].extend(shuffled[:train_count])
            split_map["val"].extend(shuffled[train_count:train_count + val_count])
            if self.split.enable_test_split:
                split_map["test"].extend(shuffled[train_count + val_count:train_count + val_count + test_count])
        return split_map

    def _process_sample(self, sample: SampleRecord, split_name: str,
                        rng_seed: Optional[int] = None) -> List[Dict[str, Any]]:
        generated: List[Dict[str, Any]] = []
        base_image = self._load_image(sample.image_path, mode="RGB")
        mask_image = self._load_image(sample.mask_path, mode="L") if sample.mask_path else None

        variants = self._apply_augmentations(base_image, mask_image, rng_seed)
        for variant_idx, (variant_image, variant_mask, variant_tag) in enumerate(variants):
            generated.extend(self._save_all_resolutions(
                variant_image, variant_mask, sample, split_name, variant_idx, variant_tag))
        return generated

    def _save_all_resolutions(self, image: Image.Image, mask: Optional[Image.Image],
                              sample: SampleRecord, split_name: str,
                              variant_idx: int, variant_tag: str) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        for target_size in self.prep.target_sizes:
            resized_image = ImageOps.fit(image, (target_size, target_size), method=RESAMPLE_BICUBIC)
            resized_mask = None
            if mask is not None:
                resized_mask = ImageOps.fit(mask, (target_size, target_size), method=RESAMPLE_NEAREST)
            entry = self._process_resolution(
                resized_image, resized_mask, sample, split_name, target_size, variant_idx, variant_tag)
            entries.append(entry)
        return entries

    def _process_resolution(self, image: Image.Image, mask: Optional[Image.Image],
                             sample: SampleRecord, split_name: str, target_size: int,
                             variant_idx: int, variant_tag: str) -> Dict[str, Any]:
        image_uint8 = np.array(image, dtype=np.uint8)
        image_float = self.prep.normalize(image_uint8).astype(np.float32)

        arrays: Dict[str, Any] = {"image": self.prep.cast_for_storage(image_float)}
        if self.prep.store_uint8_copy:
            arrays["image_uint8"] = image_uint8

        if mask is not None:
            mask_arr = np.array(mask, dtype=np.uint8)
            mask_arr = mask_arr[:, :, None] if mask_arr.ndim == 2 else mask_arr
            mask_float = (mask_arr.astype(np.float32) / 255.0)
            arrays["mask"] = self.prep.cast_for_storage(mask_float)

        if self.prep.compute_high_pass:
            high_pass, residual = self._compute_high_pass_features(image_uint8)
            arrays["high_pass"] = self.prep.cast_for_storage(high_pass)
            arrays["residual"] = self.prep.cast_for_storage(residual)

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

        entry = {
            **meta,
            "relative_path": str(output_path.relative_to(self._prepared_root)),
            "tar_path": None,
            "tar_member": output_path.name,
            "storage_dtype": str(self.prep.storage_dtype),
        }
        return entry

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

    def _apply_augmentations(self, image: Image.Image, mask: Optional[Image.Image],
                              rng_seed: Optional[int]) -> List[Tuple[Image.Image, Optional[Image.Image], str]]:
        variants: List[Tuple[Image.Image, Optional[Image.Image], str]] = [(image.copy(), mask.copy() if mask else None, "orig")]
        if not self.augment.enable:
            return variants

        rng = random.Random(rng_seed if rng_seed is not None else self._aug_seed_base)

        for idx in range(self.augment.copies_per_sample):
            aug_img = image.copy()
            aug_mask = mask.copy() if mask else None
            tag_parts: List[str] = []

            if self.augment.enable_flips and rng.random() < 0.5:
                aug_img = ImageOps.mirror(aug_img)
                if aug_mask:
                    aug_mask = ImageOps.mirror(aug_mask)
                tag_parts.append("hflip")

            if self.augment.enable_flips and rng.random() < 0.2:
                aug_img = ImageOps.flip(aug_img)
                if aug_mask:
                    aug_mask = ImageOps.flip(aug_mask)
                tag_parts.append("vflip")

            if self.augment.enable_rotations and self.augment.max_rotation_degrees > 0:
                angle = rng.uniform(-self.augment.max_rotation_degrees, self.augment.max_rotation_degrees)
                if abs(angle) > 1e-3:
                    aug_img = aug_img.rotate(angle, resample=RESAMPLE_BICUBIC, expand=False, fillcolor=0)
                    if aug_mask:
                        aug_mask = aug_mask.rotate(angle, resample=RESAMPLE_NEAREST, expand=False, fillcolor=0)
                    tag_parts.append(f"rot{int(angle)}")

            if self.augment.enable_random_crop:
                aug_img, aug_mask, cropped = self._random_crop(aug_img, aug_mask, self.augment.crop_scale_range, rng)
                if cropped:
                    tag_parts.append("crop")

            if self.augment.enable_color_jitter:
                aug_img = self._color_jitter(aug_img, rng)
                tag_parts.append("jitter")

            if self.augment.enable_noise:
                aug_img = self._add_noise(aug_img, rng)
                tag_parts.append("noise")

            variant_tag = "aug" + str(idx + 1)
            if tag_parts:
                variant_tag += "_" + "-".join(tag_parts)
            variants.append((aug_img, aug_mask, variant_tag))
        return variants

    def _random_crop(self, image: Image.Image, mask: Optional[Image.Image],
                     scale_range: Tuple[float, float], rng: random.Random) -> Tuple[Image.Image, Optional[Image.Image], bool]:
        min_scale, max_scale = scale_range
        min_scale = max(0.1, min_scale)
        max_scale = min(1.0, max_scale)
        if min_scale >= max_scale:
            return image, mask, False
        scale = rng.uniform(min_scale, max_scale)
        if math.isclose(scale, 1.0, rel_tol=1e-3):
            return image, mask, False
        w, h = image.size
        crop_w = max(1, int(w * scale))
        crop_h = max(1, int(h * scale))
        if crop_w == w and crop_h == h:
            return image, mask, False
        max_left = w - crop_w
        max_top = h - crop_h
        left = rng.randint(0, max(0, max_left))
        top = rng.randint(0, max(0, max_top))
        box = (left, top, left + crop_w, top + crop_h)
        cropped_image = image.crop(box)
        cropped_mask = mask.crop(box) if mask else None
        return cropped_image, cropped_mask, True

    def _color_jitter(self, image: Image.Image, rng: random.Random) -> Image.Image:
        factor_range = self.augment.color_jitter_factors
        brightness = ImageEnhance.Brightness(image).enhance(
            rng.uniform(*factor_range))
        contrast = ImageEnhance.Contrast(brightness).enhance(
            rng.uniform(*factor_range))
        colorized = ImageEnhance.Color(contrast).enhance(
            rng.uniform(*factor_range))
        return colorized

    def _add_noise(self, image: Image.Image, rng: random.Random) -> Image.Image:
        std_min, std_max = self.augment.noise_std_range
        std = rng.uniform(std_min, std_max)
        if std <= 0:
            return image
        arr = np.array(image, dtype=np.float32)
        np_rng = np.random.default_rng(rng.randrange(0, 2**32 - 1))
        noise = np_rng.normal(0.0, std * 255.0, size=arr.shape)
        noised = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noised)

# ----------------- PyTorch Dataset Loader ----------------- #
class PreparedForgeryDataset(TorchDataset):
    def __init__(self, prepared_root: Path | str, split: str, target_size: int,
                 include_features: bool = True, return_masks: bool = True) -> None:
        self.prepared_root = Path(prepared_root)
        self.split = split
        self.target_size = target_size
        manifest_path = self.prepared_root / MANIFEST_FILENAME
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest file '{manifest_path}' not found.")

        manifest_df = pd.read_parquet(manifest_path)
        split_df = manifest_df[(manifest_df["split"] == split) &
                               (manifest_df["target_size"] == target_size)].reset_index(drop=True)
        if split_df.empty:
            raise FileNotFoundError(f"No preprocessed samples found for split='{split}', size={target_size}.")

        self.records = split_df.to_dict("records")
        for record in self.records:
            for key in ("tar_path", "relative_path"):
                value = record.get(key)
                if value:
                    record[key] = value.replace("\\", "/")
        self.include_features = include_features
        self.return_masks = return_masks
        self._tar_cache: Dict[str, tarfile.TarFile] = {}
        self._manifest_df = split_df
        self.sample_labels = self._load_or_build_labels()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]

        if record.get("tar_path"):
            tar_path = self.prepared_root / record["tar_path"]
            tar = self._get_tar_handle(tar_path)
            member_name = record["tar_member"]
            member = tar.extractfile(member_name)
            if member is None:
                raise FileNotFoundError(f"Member '{member_name}' not found inside '{tar_path}'.")
            try:
                npz_bytes = member.read()
            finally:
                member.close()
            npz_stream = io.BytesIO(npz_bytes)
            npz_source = np.load(npz_stream, allow_pickle=True)
        else:
            rel_path = record.get("relative_path")
            if not rel_path:
                raise FileNotFoundError("Record is missing both tar_path and relative_path.")
            npz_path = self.prepared_root / rel_path
            npz_source = np.load(npz_path, allow_pickle=True)

        with npz_source as data:
            image = torch.from_numpy(data["image"].transpose(2, 0, 1)).float()
            sample: Dict[str, Any] = {"image": image}
            if self.return_masks and "mask" in data.files:
                mask = torch.from_numpy(data["mask"].transpose(2, 0, 1)).float()
                sample["mask"] = mask
            if self.include_features:
                if "high_pass" in data.files and data["high_pass"].size:
                    sample["high_pass"] = torch.from_numpy(data["high_pass"].transpose(2, 0, 1)).float()
                if "residual" in data.files and data["residual"].size:
                    sample["residual"] = torch.from_numpy(data["residual"].transpose(2, 0, 1)).float()
            meta = json.loads(str(data["meta"].item())) if "meta" in data.files else {}
            sample["label"] = torch.tensor(1 if meta.get("label") == "fake" else 0)
            sample["meta"] = meta
            return sample

    def _load_or_build_labels(self) -> List[int]:
        cache_name = f".labels_{self.split}_{self.target_size}.npy"
        cache_path = self.prepared_root / cache_name
        if cache_path.exists():
            return np.load(cache_path).astype(np.int8).tolist()

        labels = np.asarray([1 if record.get("label") == "fake" else 0 for record in self.records], dtype=np.int8)
        np.save(cache_path, labels)
        return labels.tolist()

    def _get_tar_handle(self, tar_path: Path) -> tarfile.TarFile:
        key = str(tar_path.resolve())
        handle = self._tar_cache.get(key)
        if handle is None:
            handle = tarfile.open(tar_path, "r")
            self._tar_cache[key] = handle
        return handle

    def close(self) -> None:
        for handle in self._tar_cache.values():
            try:
                handle.close()
            except Exception:
                pass
        self._tar_cache.clear()

    def __del__(self) -> None:
        self.close()

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Example: process multiple datasets
    structure = DatasetStructureConfig(
        dataset_root="./datasets",
        dataset_names=["CASIA2", "AnotherDataset"],
        real_subdir="real",
        fake_subdir="fake",
        mask_subdir="mask",
        mask_suffix="_gt",
        prepared_root="prepared",
    )

    # Option 1: normal split (train/val/test)
    split = SplitConfig(train=0.7, val=0.15, test=0.15, seed=42, enable_test_split=True)
    prep = PreparationConfig(target_sizes=(384,), normalization_mode="zero_one", compute_high_pass=True)
    augment = AugmentationConfig(enable=True, copies_per_sample=2, max_rotation_degrees=15,
                                 crop_scale_range=(0.8, 1.0), noise_std_range=(0.0, 0.03))
    pipeline = DataPreparationPipeline(structure, split, prep, augment)
    results = pipeline.prepare()
    print("Preparation finished.")
    for ds_name, manifest_df in results.items():
        if not manifest_df.empty:
            for split_name, count in manifest_df.groupby("split").size().items():
                print(f"{ds_name} - {split_name}: {count} artifacts")

    # Option 2: skip test split
    split_no_test = SplitConfig(train=0.8, val=0.2, test=0.0, seed=42, enable_test_split=False)
    pipeline_no_test = DataPreparationPipeline(structure, split_no_test, prep, augment)
    results_no_test = pipeline_no_test.prepare()
    print("Preparation (no test split) finished.")
    for ds_name, manifest_df in results_no_test.items():
        if not manifest_df.empty:
            for split_name, count in manifest_df.groupby("split").size().items():
                print(f"{ds_name} - {split_name}: {count} artifacts")

    # Option 3: test split only
    split_test_only = SplitConfig(train=0.0, val=0.0, test=1.0, seed=42, enable_test_split=True)
    pipeline_test_only = DataPreparationPipeline(structure, split_test_only, prep, augment)
    results_test_only = pipeline_test_only.prepare(test_only=True)
    print("Preparation (test split only) finished.")
    for ds_name, manifest_df in results_test_only.items():
        if not manifest_df.empty:
            for split_name, count in manifest_df.groupby("split").size().items():
                print(f"{ds_name} - {split_name}: {count} artifacts")