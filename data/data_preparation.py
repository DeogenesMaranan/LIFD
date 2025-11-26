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
from typing import Any, Dict, Iterator, List, Optional, Tuple, Sequence

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
    # Keep augmentation minimal by default to limit storage usage.
    # `copies_per_sample` controls how many augmented variants are produced
    # in addition to the original. Set to 0 to produce only the original.
    copies_per_sample: int = 0
    enable_flips: bool = False
    enable_rotations: bool = False
    enable_random_crop: bool = False
    enable_color_jitter: bool = False
    enable_noise: bool = False
    max_rotation_degrees: float = 10.0
    crop_scale_range: Tuple[float, float] = (0.9, 1.0)
    color_jitter_factors: Tuple[float, float] = (0.9, 1.1)
    noise_std_range: Tuple[float, float] = (0.0, 0.02)


@dataclass
class PreparationConfig:
    target_sizes: Tuple[int, ...] = (320,)
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
        self._prepared_root = self.structure.prepared_path()
        self._prepared_root.mkdir(parents=True, exist_ok=True)
        self._sample_counter = 0
        self._last_sample_log = self._prepared_root / "last_sample.log"
        # Track how many fake samples were skipped because they lacked masks
        self._skipped_no_mask = 0

    def prepare(self) -> pd.DataFrame:
        samples_by_label = self._collect_samples()
        if getattr(self, "_skipped_no_mask", 0) > 0:
            print(f"Skipped {self._skipped_no_mask} fake samples without masks in {self.structure.as_path()}")
        split_map = self._split_samples(samples_by_label)
        manifest_entries: List[Dict[str, Any]] = []

        print("Samples per split:", {k: len(v) for k, v in split_map.items()})

        total_samples = sum(len(samples) for samples in split_map.values())
        progress = tqdm(total=total_samples, desc="Preparing samples", unit="sample") if tqdm else None

        executor = ThreadPoolExecutor(max_workers=self.max_workers) if self.max_workers > 1 else None

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
        manifest_path = self._prepared_root / MANIFEST_FILENAME
        manifest_df.to_parquet(manifest_path, index=False)

        split_counts = manifest_df.groupby("split").size().to_dict() if not manifest_df.empty else {}
        print("Artifacts per split:", split_counts)

        # Write a small JSON summary with counts to help estimate storage
        # and for bookkeeping. This includes the number of input samples
        # scanned, number of generated artifacts, number skipped due to
        # missing masks, and per-split artifact counts.
        try:
            summary = {
                "prepared_root": str(self._prepared_root),
                "total_input_samples": total_samples,
                "generated_artifacts": len(manifest_entries),
                "skipped_no_mask": int(getattr(self, "_skipped_no_mask", 0)),
                "artifacts_per_split": {k: int(v) for k, v in split_counts.items()},
            }
            summary_path = self._prepared_root / "prepare_summary.json"
            with open(summary_path, "w", encoding="utf-8") as fh:
                import json as _json

                _json.dump(summary, fh, indent=2)
            print(f"Wrote preparation summary to {summary_path}")
        except Exception:
            # Non-fatal: don't stop the pipeline for summary write errors
            pass

        return manifest_df

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

    def _collect_samples(self) -> Dict[str, List[SampleRecord]]:
        dataset_path = self.structure.as_path()
        real_dir = dataset_path / self.structure.real_subdir
        fake_dir = dataset_path / self.structure.fake_subdir
        mask_dir = dataset_path / self.structure.mask_subdir

        samples: Dict[str, List[SampleRecord]] = {"real": [], "fake": []}
        # Reset skipped counter once per dataset
        self._skipped_no_mask = 0

        # Collect real images
        for image_path in self._iter_images(real_dir):
            samples["real"].append(SampleRecord(image_path=image_path, mask_path=None, label="real"))

        # Collect fake images, skipping entries without masks
        for image_path in self._iter_images(fake_dir):
            mask_path = self._resolve_mask_path(mask_dir, image_path)
            # Skip tampered samples that do not have an associated mask
            # (user requested to exclude these from the prepared dataset).
            if mask_path is None:
                self._skipped_no_mask += 1
                continue
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
            # Compute per-split counts using floor (int) then distribute any
            # leftover samples (due to rounding) only to splits that have a
            # non-zero proportion. This prevents a configured 0.0 split
            # from receiving remainder samples.
            train_prop = float(self.split.train)
            val_prop = float(self.split.val)
            test_prop = float(self.split.test)

            counts = [int(total * train_prop), int(total * val_prop), int(total * test_prop)]
            rem = total - sum(counts)

            props = [train_prop, val_prop, test_prop]
            # If all proportions are zero (shouldn't happen), put everything in train
            if all(p == 0.0 for p in props):
                counts = [total, 0, 0]
                rem = 0

            idx = 0
            # Distribute remainder round-robin but only to splits with prop>0
            while rem > 0:
                target = idx % 3
                if props[target] > 0.0:
                    counts[target] += 1
                    rem -= 1
                idx += 1

            train_count, val_count, test_count = counts
            split_map["train"].extend(shuffled[:train_count])
            split_map["val"].extend(shuffled[train_count:train_count + val_count])
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
    def __init__(self, prepared_root: Path | str | Sequence[Path | str], split: str, target_size: int,
                 include_features: bool = True, return_masks: bool = True) -> None:
        """
        `prepared_root` may be:
        - a single path to a prepared dataset directory containing `manifest.parquet`,
        - a list/sequence of such prepared dataset directories, or
        - the path to a combined manifest parquet (created by the preprocessing pipeline)

        The loader will merge manifests and normalize `relative_path`/`tar_path`
        entries so they are resolvable against the correct prepared root.
        """
        self.split = split
        self.target_size = target_size

        # Normalize input into a list of prepared roots or a combined manifest file
        roots: List[Path] = []
        combined_manifest_path: Optional[Path] = None

        # Accept strings, Paths or sequences
        if isinstance(prepared_root, (list, tuple)):
            for p in prepared_root:
                roots.append(Path(p))
        else:
            p = Path(prepared_root)
            if p.is_file() and p.suffix.lower() in {".parquet", ".pq"}:
                combined_manifest_path = p
            else:
                roots.append(p)

        # Load manifests
        dfs: List[pd.DataFrame] = []
        if combined_manifest_path is not None:
            if not combined_manifest_path.exists():
                raise FileNotFoundError(f"Combined manifest '{combined_manifest_path}' not found.")
            combined_df = pd.read_parquet(combined_manifest_path)
            # combined manifest is expected to include a `prepared_root` column
            if "prepared_root" not in combined_df.columns:
                # If not present, assume relative paths are relative to the manifest parent
                combined_df["prepared_root"] = str(combined_manifest_path.parent)
            else:
                # Normalize / rebase prepared_root entries so they resolve inside
                # the environment where the manifest was loaded. Manifests created
                # on a different machine may contain relative paths like
                # 'prepared/CASIA2' which won't exist on the current host. For
                # each entry, prefer an existing absolute path; otherwise try to
                # rebase the (possibly relative) path under the manifest's
                # parent directory. If that still doesn't exist, fall back to
                # the manifest parent directory itself.
                def _rebase_pr(pr_value):
                    try:
                        p = Path(pr_value)
                    except Exception:
                        return str(combined_manifest_path.parent)
                    manifest_parent = combined_manifest_path.parent
                    # Absolute path: keep only if it exists; otherwise fall back
                    if p.is_absolute():
                        return str(p) if p.exists() else str(manifest_parent)

                    # If manifest contains a common "prepared/<dataset>" layout,
                    # try to map that to the snapshot-local dataset directory.
                    # e.g. 'prepared/CASIA2' -> '<snapshot_parent>/CASIA2'
                    parts = [part for part in p.parts if part]
                    if parts and parts[0].lower() == "prepared" and len(parts) >= 2:
                        dataset_name = parts[1]
                        candidate_alt = manifest_parent / dataset_name
                        if candidate_alt.exists():
                            return str(candidate_alt)

                    # General relative -> try manifest_parent / relative
                    candidate = manifest_parent / p
                    if candidate.exists():
                        return str(candidate)

                    # As a last attempt, try manifest_parent / last_name
                    last_name = p.name
                    if last_name:
                        candidate_last = manifest_parent / last_name
                        if candidate_last.exists():
                            return str(candidate_last)

                    # Last resort: manifest parent
                    return str(manifest_parent)

                combined_df["prepared_root"] = combined_df["prepared_root"].apply(_rebase_pr)
            dfs.append(combined_df)

        for root in roots:
            manifest_path = root / MANIFEST_FILENAME
            if not manifest_path.exists():
                raise FileNotFoundError(f"Manifest file '{manifest_path}' not found.")
            df = pd.read_parquet(manifest_path)
            # Record where these artifacts live so relative paths can be resolved
            df = df.copy()
            df["prepared_root"] = str(root)
            dfs.append(df)

        if not dfs:
            raise FileNotFoundError("No manifest data found for provided prepared_root(s).")

        manifest_df = pd.concat(dfs, ignore_index=True)
        # Filter by split and target_size
        split_df = manifest_df[(manifest_df["split"] == split) & (manifest_df["target_size"] == target_size)].reset_index(drop=True)
        if split_df.empty:
            raise FileNotFoundError(f"No preprocessed samples found for split='{split}', size={target_size}.")

        # Normalize path separators and keep the prepared_root column
        records = split_df.to_dict("records")
        for record in records:
            for key in ("tar_path", "relative_path"):
                value = record.get(key)
                if value and isinstance(value, str):
                    record[key] = value.replace("\\", "/")

        self.records = records
        # Expose a canonical prepared_root attribute for single-root code paths.
        # If multiple prepared roots are used, this will be the first one.
        self.prepared_root = Path(records[0].get("prepared_root", "."))
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
            tar_path = Path(record.get("prepared_root", ".")) / record["tar_path"]
            # If the tar path doesn't exist, try inserting the dataset name
            # between the prepared_root and tar_path (common combined-manifest layout).
            if not tar_path.exists():
                dataset_name = record.get("dataset")
                if dataset_name:
                    alt = Path(record.get("prepared_root", ".")) / dataset_name / record["tar_path"]
                    if alt.exists():
                        tar_path = alt
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
            npz_path = Path(record.get("prepared_root", ".")) / rel_path
            # If not found, try inserting the dataset name as an intermediate
            if not npz_path.exists():
                dataset_name = record.get("dataset")
                if dataset_name:
                    alt_npz = Path(record.get("prepared_root", ".")) / dataset_name / rel_path
                    if alt_npz.exists():
                        npz_path = alt_npz
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
        cache_path = Path(self.records[0].get("prepared_root", ".")) / cache_name
        if cache_path.exists():
            return np.load(cache_path).astype(np.int8).tolist()

        labels = np.asarray([1 if record.get("label") == "fake" else 0 for record in self.records], dtype=np.int8)
        np.save(cache_path, labels)
        return labels.tolist()

    def _get_tar_handle(self, tar_path: Path) -> tarfile.TarFile:
        key = str(tar_path.resolve())
        handle = self._tar_cache.get(key)
        if handle is None:
            if not tar_path.exists():
                raise FileNotFoundError(f"Tar shard not found: '{tar_path}'. Checked prepared_root and dataset-name fallback.")
            handle = tarfile.open(tar_path, "r")
            self._tar_cache[key] = handle
        return handle

    def close(self) -> None:
        if not hasattr(self, "_tar_cache") or self._tar_cache is None:
            return
        for handle in list(self._tar_cache.values()):
            try:
                handle.close()
            except Exception:
                pass
        self._tar_cache.clear()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class EvenMultiSourceBatchSampler:
    """Batch sampler that attempts to produce batches containing an even
    number of samples from each provided prepared_root (dataset source).

    Behavior:
    - Groups dataset indices by the `prepared_root` value present on each
      record (the loader annotates this column during init).
    - For each batch, requests `batch_size // num_sources` samples from each
      source and distributes any remainder to the first sources.
    - If a source runs out of samples during an epoch, it is reshuffled and
      reused (this produces oversampling but keeps per-batch balance).
    """

    def __init__(self, dataset: PreparedForgeryDataset, batch_size: int, shuffle: bool = True, drop_last: bool = False, seed: int = 42):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)

    def __iter__(self):
        # Group indices by prepared_root
        groups = {}
        for idx, rec in enumerate(self.dataset.records):
            root = rec.get("prepared_root", "__none__")
            groups.setdefault(root, []).append(idx)

        roots = list(groups.keys())
        if not roots:
            return

        num_roots = len(roots)
        base_quota = self.batch_size // num_roots
        remainder = self.batch_size % num_roots

        # Prepare per-root pools (mutable queues)
        rng = random.Random(self.seed)
        pools = {}
        for root in roots:
            items = groups[root].copy()
            if self.shuffle:
                rng.shuffle(items)
            pools[root] = items

        # Compute number of batches: use the maximum number of batches required
        # to exhaust any source (so all sources contribute roughly equally).
        batches_est = 0
        for i, root in enumerate(roots):
            per_batch = base_quota + (1 if i < remainder else 0)
            if per_batch <= 0:
                continue
            batches_for_root = math.ceil(len(groups[root]) / per_batch)
            batches_est = max(batches_est, batches_for_root)

        if batches_est == 0:
            return

        for _ in range(batches_est):
            batch = []
            for i, root in enumerate(roots):
                per = base_quota + (1 if i < remainder else 0)
                pool = pools[root]
                # Replenish pool by reshuffling the original list if needed
                while len(pool) < per:
                    refill = groups[root].copy()
                    if self.shuffle:
                        rng.shuffle(refill)
                    pool.extend(refill)
                for _ in range(per):
                    batch.append(pool.pop(0))

            if len(batch) < self.batch_size:
                if self.drop_last:
                    continue
                # Fill missing spots by sampling from available pools
                i = 0
                while len(batch) < self.batch_size:
                    root = roots[i % num_roots]
                    if not pools[root]:
                        refill = groups[root].copy()
                        if self.shuffle:
                            rng.shuffle(refill)
                        pools[root].extend(refill)
                    batch.append(pools[root].pop(0))
                    i += 1

            yield batch

    def __len__(self):
        # Conservative length: computed similarly to iteration
        groups = {}
        for idx, rec in enumerate(self.dataset.records):
            root = rec.get("prepared_root", "__none__")
            groups.setdefault(root, []).append(idx)
        roots = list(groups.keys())
        if not roots:
            return 0
        num_roots = len(roots)
        base_quota = self.batch_size // num_roots
        remainder = self.batch_size % num_roots
        batches_est = 0
        for i, root in enumerate(roots):
            per_batch = base_quota + (1 if i < remainder else 0)
            if per_batch <= 0:
                continue
            batches_for_root = math.ceil(len(groups[root]) / per_batch)
            batches_est = max(batches_est, batches_for_root)
        return batches_est
    

class EvenMultiSourceBalancedSampler:
    """Sampler that builds batches balanced across prepared_root sources
    while also performing per-source class-balanced sampling.

    For each source, we compute per-sample probabilities so that the expected
    fraction of positive (fake) samples matches `pos_ratio`. During batch
    construction we draw samples from each source according to these
    probabilities (with replacement) and assemble them into a single batch
    preserving even per-source counts.
    """

    def __init__(self, dataset: PreparedForgeryDataset, batch_size: int, pos_ratio: float = 0.5, shuffle: bool = True, seed: int = 42):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.pos_ratio = float(pos_ratio)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)

    def __iter__(self):
        # Group indices by prepared_root
        groups = {}
        for idx, rec in enumerate(self.dataset.records):
            root = rec.get("prepared_root", "__none__")
            groups.setdefault(root, []).append(idx)

        roots = list(groups.keys())
        if not roots:
            return

        num_roots = len(roots)
        base_quota = self.batch_size // num_roots
        remainder = self.batch_size % num_roots

        rng = np.random.default_rng(self.seed)

        # Precompute per-source sampling probabilities
        probs_by_root = {}
        for root in roots:
            idxs = groups[root]
            labels = np.array([1 if self.dataset.records[i].get("label") == "fake" else 0 for i in idxs], dtype=np.int8)
            pos_count = labels.sum()
            neg_count = labels.size - pos_count
            pos_ratio = min(max(self.pos_ratio, 0.0), 1.0)
            neg_ratio = 1.0 - pos_ratio
            # Avoid divide-by-zero
            pos_count = max(1, int(pos_count))
            neg_count = max(1, int(neg_count))
            weights = np.zeros_like(labels, dtype=np.float64)
            weights[labels == 1] = pos_ratio / pos_count
            weights[labels == 0] = neg_ratio / neg_count
            # Normalize to sum to 1
            weights = weights / weights.sum()
            probs_by_root[root] = (np.array(idxs, dtype=np.int64), weights)

        # Compute number of batches similarly to other samplers
        batches_est = 0
        for i, root in enumerate(roots):
            per_batch = base_quota + (1 if i < remainder else 0)
            if per_batch <= 0:
                continue
            batches_for_root = math.ceil(len(groups[root]) / per_batch)
            batches_est = max(batches_est, batches_for_root)

        if batches_est == 0:
            return

        for _ in range(batches_est):
            batch = []
            for i, root in enumerate(roots):
                per = base_quota + (1 if i < remainder else 0)
                idxs, probs = probs_by_root[root]
                # Sample with replacement according to per-source class-aware probs
                chosen = rng.choice(idxs, size=per, replace=True, p=probs)
                batch.extend(int(x) for x in chosen)
            # If some rounding causes batch size mismatch, pad by sampling from first root
            if len(batch) < self.batch_size:
                root = roots[0]
                idxs, probs = probs_by_root[root]
                extra = self.batch_size - len(batch)
                chosen = rng.choice(idxs, size=extra, replace=True, p=probs)
                batch.extend(int(x) for x in chosen)
            yield batch

    def __len__(self):
        # Conservative length
        groups = {}
        for idx, rec in enumerate(self.dataset.records):
            root = rec.get("prepared_root", "__none__")
            groups.setdefault(root, []).append(idx)
        roots = list(groups.keys())
        if not roots:
            return 0
        num_roots = len(roots)
        base_quota = self.batch_size // num_roots
        remainder = self.batch_size % num_roots
        batches_est = 0
        for i, root in enumerate(roots):
            per_batch = base_quota + (1 if i < remainder else 0)
            if per_batch <= 0:
                continue
            batches_for_root = math.ceil(len(groups[root]) / per_batch)
            batches_est = max(batches_est, batches_for_root)
        return batches_est
    

if __name__ == "__main__":
    """
    Prepare multiple datasets and write a combined manifest for downstream
    multi-dataset training. This example prepares the three datasets used in
    the repository examples.
    """
    datasets = [
        DatasetStructureConfig(
            dataset_root="./datasets",
            dataset_name="CASIA2",
            real_subdir="real",
            fake_subdir="fake",
            mask_subdir="mask",
            mask_suffix="_gt",
            prepared_root="prepared",
        ),
        DatasetStructureConfig(
            dataset_root="./datasets",
            dataset_name="FantasticReality",
            real_subdir="ColorRealImages",
            fake_subdir="ColorFakeImages",
            mask_subdir="masks",
            mask_suffix="",
            prepared_root="prepared",
        ),
        DatasetStructureConfig(
            dataset_root="./datasets",
            dataset_name="COVERAGE",
            real_subdir="real",
            fake_subdir="fake",
            mask_subdir="mask",
            mask_suffix="forged",
            prepared_root="prepared",
        ),
    ]

    # Per-dataset split configs (honour dataset-specific seeds)
    per_dataset_splits = {
        "CASIA2": SplitConfig(train=0.8, val=0.2, test=0.0, seed=6),
        "FantasticReality": SplitConfig(train=0.8, val=0.2, test=0.0, seed=10),
        "COVERAGE": SplitConfig(train=0.0, val=0.0, test=1.0, seed=4),
    }

    # Example preparation + augmentation configuration tuned to keep
    # storage low while retaining a single influential augmentation.
    # We produce one augmented variant per sample using a random crop
    # (scale jitter). Other augmentations are disabled to avoid extra files.
    prep_cfg = PreparationConfig(target_sizes=(320,), normalization_mode="zero_one", compute_high_pass=True)
    augment_cfg = AugmentationConfig(
        enable=True,
        copies_per_sample=1,
        enable_flips=True,
        enable_rotations=False,
        enable_random_crop=False,
        enable_color_jitter=False,
        enable_noise=False,
        max_rotation_degrees=0,
        crop_scale_range=(0.9, 1.0),
    )

    combined_dfs = []
    prepared_roots = []
    for ds in datasets:
        print(f"Preparing dataset '{ds.dataset_name}'...")
        split_cfg = per_dataset_splits.get(ds.dataset_name, SplitConfig())
        pipeline = DataPreparationPipeline(ds, split_cfg, prep_cfg, augment_cfg)
        df = pipeline.prepare()
        prepared_root = ds.prepared_path()
        prepared_roots.append(str(prepared_root))
        if not df.empty:
            df = df.copy()
            df["dataset"] = ds.dataset_name
            df["prepared_root"] = str(prepared_root)
            combined_dfs.append(df)

    # Write a combined manifest if any dataset produced artifacts
    if combined_dfs:
        combined = pd.concat(combined_dfs, ignore_index=True)
        # Choose a combined parent: prefer the common parent of all prepared roots
        parents = {str(Path(p).parent) for p in prepared_roots}
        if len(parents) == 1:
            combined_parent = Path(next(iter(parents)))
        else:
            combined_parent = Path(prepared_roots[0]).parent
        combined_path = combined_parent / "combined_manifest.parquet"
        combined.to_parquet(combined_path, index=False)
        print(f"Wrote combined manifest to {combined_path}")
    else:
        print("No artifacts generated for the provided datasets.")