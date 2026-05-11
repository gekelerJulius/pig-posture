"""Torch datasets for single-bounding-box posture classification."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from PIL import Image
from PIL import ImageFilter
from torch.utils.data import Dataset

from pigpose.data.tone import load_tone_stats, maybe_match_source_tone

IMAGENET_NEUTRAL_RGB = (124, 116, 104)
BLACK_RGB = (0, 0, 0)
MASK_BACKGROUND_MODES = ("none", "attenuate", "hard", "soft", "random", "random_attenuate")
RANDOM_MASK_BACKGROUND_MODES = ("none", "attenuate", "soft")


class SingleBboxDataset(Dataset):
    """Load one cropped pig bounding box per dataset item."""

    def __init__(
        self,
        rows: list[dict[str, Any]],
        transform: Any | None = None,
        transform_by_row: Any | None = None,
        bbox_context: float = 0.0,
        random_crop_margin: float = 0.0,
        overlap_reference_rows: list[dict[str, Any]] | None = None,
        horizontal_flip_probability: float = 0.0,
        context_transform: Any | None = None,
        metadata_vocab: dict[str, dict[str, int]] | None = None,
        include_consistency_view: bool = False,
        mask_dir: Path | str | None = None,
        mask_dir_root: Path | str | None = None,
        mask_background_attenuation: float = 0.0,
        mask_background_mode: str = "none",
        mask_neutral_rgb: tuple[int, int, int] = IMAGENET_NEUTRAL_RGB,
        mask_fill_rgb: tuple[int, int, int] = BLACK_RGB,
        mask_cache_size: int = 64,
        include_foreground_view: bool = False,
        foreground_mask_mode: str = "hard",
        include_mask_metadata: bool = True,
        tone_stats_path: Path | str | None = None,
        tone_match_probability: float = 0.0,
        tone_match_strength_range: tuple[float, float] = (1.0, 1.0),
    ) -> None:
        if bbox_context < 0:
            raise ValueError("bbox_context must be non-negative.")
        if random_crop_margin < 0:
            raise ValueError("random_crop_margin must be non-negative.")
        if not 0 <= horizontal_flip_probability <= 1:
            raise ValueError("horizontal_flip_probability must be between 0 and 1.")
        if not 0 <= mask_background_attenuation <= 1:
            raise ValueError("mask_background_attenuation must be between 0 and 1.")
        if mask_background_mode not in MASK_BACKGROUND_MODES:
            raise ValueError(f"mask_background_mode must be one of: {', '.join(MASK_BACKGROUND_MODES)}.")
        if foreground_mask_mode not in {"hard", "soft", "attenuate"}:
            raise ValueError("foreground_mask_mode must be one of: hard, soft, attenuate.")
        if mask_cache_size < 0:
            raise ValueError("mask_cache_size must be non-negative.")
        if not 0 <= tone_match_probability <= 1:
            raise ValueError("tone_match_probability must be between 0 and 1.")
        if len(tone_match_strength_range) != 2:
            raise ValueError("tone_match_strength_range must contain exactly two values.")
        tone_match_strength_min, tone_match_strength_max = tone_match_strength_range
        if not 0 <= tone_match_strength_min <= tone_match_strength_max <= 1:
            raise ValueError("tone_match_strength_range must satisfy 0 <= min <= max <= 1.")
        if mask_dir is not None and mask_dir_root is not None:
            raise ValueError("Use either mask_dir or mask_dir_root, not both.")
        self.rows = rows
        self.transform = transform
        self.transform_by_row = transform_by_row
        self.bbox_context = bbox_context
        self.random_crop_margin = random_crop_margin
        self.horizontal_flip_probability = horizontal_flip_probability
        self.context_transform = context_transform
        self.metadata_vocab = metadata_vocab or {}
        self.include_consistency_view = include_consistency_view
        self.mask_dir = Path(mask_dir) if mask_dir is not None else None
        self.mask_dir_root = Path(mask_dir_root) if mask_dir_root is not None else None
        self.mask_background_attenuation = mask_background_attenuation
        self.mask_background_mode = mask_background_mode
        self.mask_neutral_rgb = mask_neutral_rgb
        self.mask_fill_rgb = mask_fill_rgb
        self.mask_cache_size = mask_cache_size
        self.include_foreground_view = include_foreground_view
        self.foreground_mask_mode = foreground_mask_mode
        self.include_mask_metadata = include_mask_metadata
        self.tone_match_probability = tone_match_probability
        self.tone_match_strength_range = (tone_match_strength_min, tone_match_strength_max)
        self._tone_stats = load_tone_stats(tone_stats_path)
        self._mask_cache: OrderedDict[Path, Image.Image] = OrderedDict()
        self._mask_paths_by_row_id = (
            build_mask_paths_by_row_id(rows, self.mask_dir, self.mask_dir_root)
            if self._requires_masks()
            else {}
        )
        self.overlap_reference_rows = overlap_reference_rows or rows
        self._rows_by_image_id: dict[str, list[dict[str, Any]]] = {}
        for row in self.overlap_reference_rows:
            self._rows_by_image_id.setdefault(row["image_id"], []).append(row)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        primary = self._build_view(row)
        item = {
            "image": primary["image"],
            "row_id": row["row_id"],
            "image_id": row["image_id"],
            "horizontally_flipped": primary["horizontally_flipped"],
        }
        if self.context_transform is not None:
            item["context_image"] = primary["context_image"]
        if self.include_foreground_view:
            item["foreground_image"] = primary["foreground_image"]
        for metadata_key in ("pen", "camera_type", "camera", "view_id"):
            if metadata_key in row and row[metadata_key] is not None:
                item[metadata_key] = row[metadata_key]
        if "domain" in row:
            item["domain"] = row["domain"]
        item["is_pseudo"] = bool(row.get("is_pseudo", False))
        item["pseudo_confidence"] = float(row.get("pseudo_confidence", 1.0))
        item["pseudo_margin"] = float(row.get("pseudo_margin", 0.0))
        item["pseudo_entropy"] = float(row.get("pseudo_entropy", 0.0))
        item["pseudo_agreement"] = float(row.get("pseudo_agreement", 1.0))
        item["pseudo_quality_weight"] = float(row.get("pseudo_quality_weight", 1.0))
        soft_target = row.get("pseudo_soft_target")
        item["pseudo_has_soft_target"] = bool(row.get("pseudo_has_soft_target", soft_target is not None))
        item["pseudo_soft_target"] = torch.tensor(
            soft_target if soft_target is not None else [0.0, 0.0, 0.0, 0.0, 0.0],
            dtype=torch.float32,
        )
        item["sample_bucket"] = str(row.get("sample_bucket", "pseudo" if row.get("is_pseudo") else row.get("domain", "unknown")))
        item["sample_loss_weight"] = float(row.get("sample_loss_weight", 1.0))
        metadata_ids = {
            key: self._metadata_id(key, row)
            for key in self.metadata_vocab
        }
        if metadata_ids:
            item["metadata"] = torch.tensor(
                [metadata_ids[key] for key in self.metadata_vocab],
                dtype=torch.long,
            )
            item["metadata_continuous"] = torch.tensor(
                self._continuous_metadata(row, primary["crop_scale"], primary["mask_features"]),
                dtype=torch.float32,
            )
        if primary["label"] is not None:
            item["label"] = primary["label"]
        if self.include_consistency_view:
            consistency = self._build_view(row)
            item["image_consistency"] = consistency["image"]
            item["horizontally_flipped_consistency"] = consistency["horizontally_flipped"]
            if self.context_transform is not None:
                item["context_image_consistency"] = consistency["context_image"]
            if self.include_foreground_view:
                item["foreground_image_consistency"] = consistency["foreground_image"]
            if metadata_ids:
                item["metadata_continuous_consistency"] = torch.tensor(
                    self._continuous_metadata(row, consistency["crop_scale"], consistency["mask_features"]),
                    dtype=torch.float32,
                )
            if consistency["label"] is not None:
                item["label_consistency"] = consistency["label"]
        return item

    def _build_view(self, row: dict[str, Any]) -> dict[str, Any]:
        image_path = Path(row["image_path"])
        with Image.open(image_path) as opened_image:
            full_image = opened_image.convert("RGB")
            full_image = maybe_match_source_tone(
                full_image,
                row,
                self._tone_stats,
                probability=self.tone_match_probability,
                strength_range=self.tone_match_strength_range,
            )
            crop_box = _expanded_crop_box(
                row["bbox"],
                full_image.size[0],
                full_image.size[1],
                self.bbox_context,
                random_crop_margin=self.random_crop_margin,
            )
            image, crop_scale = _crop_from_image(
                full_image,
                image_path,
                row["bbox"],
                self.bbox_context,
                crop_box=crop_box,
                mask_path=self._mask_paths_by_row_id.get(str(row["row_id"])),
                mask_background_attenuation=self.mask_background_attenuation,
                mask_background_mode=self.mask_background_mode,
                mask_neutral_rgb=self.mask_neutral_rgb,
                mask_fill_rgb=self.mask_fill_rgb,
                mask_cache=self._mask_cache,
                mask_cache_size=self.mask_cache_size,
            )
            foreground_image = None
            if self.include_foreground_view:
                foreground_image, _ = _crop_from_image(
                    full_image,
                    image_path,
                    row["bbox"],
                    self.bbox_context,
                    crop_box=crop_box,
                    mask_path=self._mask_paths_by_row_id.get(str(row["row_id"])),
                    mask_background_attenuation=self.mask_background_attenuation,
                    mask_background_mode=self.foreground_mask_mode,
                    mask_neutral_rgb=self.mask_neutral_rgb,
                    mask_fill_rgb=self.mask_fill_rgb,
                    mask_cache=self._mask_cache,
                    mask_cache_size=self.mask_cache_size,
                )
            mask_features = self._mask_features(row, crop_scale)
            context_image = full_image.copy() if self.context_transform is not None else None
        label = row.get("class_id")
        should_flip = self._should_horizontally_flip()
        if should_flip:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if label is not None:
                label = horizontally_flipped_label(label)
            if context_image is not None:
                context_image = context_image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if foreground_image is not None:
                foreground_image = foreground_image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        transform = self._transform_for_row(row)
        if transform is not None:
            image = transform(image)
            if foreground_image is not None:
                foreground_image = transform(foreground_image)
        view: dict[str, Any] = {
            "image": image,
            "label": label,
            "crop_scale": crop_scale,
            "mask_features": mask_features,
            "horizontally_flipped": should_flip,
        }
        if self.context_transform is not None and context_image is not None:
            view["context_image"] = self.context_transform(context_image)
        if foreground_image is not None:
            view["foreground_image"] = foreground_image
        return view

    def _requires_masks(self) -> bool:
        return (
            self.mask_background_mode != "none"
            or self.include_foreground_view
            or self.include_mask_metadata
        )

    def _should_horizontally_flip(self) -> bool:
        return (
            self.horizontal_flip_probability > 0
            and random.random() < self.horizontal_flip_probability
        )

    def _metadata_id(self, key: str, row: dict[str, Any]) -> int:
        vocab = self.metadata_vocab[key]
        return vocab.get(str(self._metadata_value(key, row)), vocab.get("unknown", 0))

    def _metadata_value(self, key: str, row: dict[str, Any]) -> Any:
        value = row.get(key)
        if value is not None:
            return value
        if key == "camera_id":
            return row.get("view_id") or _join_known(row.get("camera_type"), row.get("camera"))
        if key == "sensor_pen_domain_id":
            return _join_known(row.get("camera_type"), row.get("pen"), row.get("domain"))
        return "unknown"

    def _transform_for_row(self, row: dict[str, Any]) -> Any | None:
        if self.transform_by_row is not None:
            return self.transform_by_row(row)
        return self.transform

    def _continuous_metadata(self, row: dict[str, Any], crop_scale: float, mask_features: list[float]) -> list[float]:
        bbox_x, bbox_y, bbox_width, bbox_height = [float(value) for value in row["bbox"]]
        width = max(1.0, float(row["width"]))
        height = max(1.0, float(row["height"]))
        return [
            bbox_x / width,
            bbox_y / height,
            bbox_width / width,
            bbox_height / height,
            float(len(self._rows_by_image_id.get(row["image_id"], []))),
            crop_scale,
            *mask_features,
        ]

    def _mask_features(self, row: dict[str, Any], crop_scale: float) -> list[float]:
        if not self.include_mask_metadata:
            return [0.0] * 7
        mask_path = self._mask_paths_by_row_id.get(str(row["row_id"]))
        if mask_path is None or not mask_path.is_file():
            return [0.0] * 7
        mask_image = load_mask_image(mask_path, mask_cache=self._mask_cache, mask_cache_size=self.mask_cache_size)
        mask = np.asarray(mask_image) > 0
        image_area = max(1.0, float(row["width"]) * float(row["height"]))
        return mask_geometry_features(mask, image_area=image_area, crop_scale=crop_scale)


def horizontally_flipped_label(label: int) -> int:
    """Map posture label after a horizontal image flip."""

    if label == 0:
        return 1
    if label == 1:
        return 0
    return label


def _load_crop(
    image_path: Path,
    bbox: list[float],
    bbox_context: float = 0.0,
    random_crop_margin: float = 0.0,
    mask_path: Path | None = None,
    mask_background_attenuation: float = 0.0,
    mask_background_mode: str = "none",
    mask_neutral_rgb: tuple[int, int, int] = IMAGENET_NEUTRAL_RGB,
    mask_fill_rgb: tuple[int, int, int] = BLACK_RGB,
) -> tuple[Image.Image, float]:
    with Image.open(image_path) as image:
        return _crop_from_image(
            image.convert("RGB"),
            image_path,
            bbox,
            bbox_context,
            random_crop_margin=random_crop_margin,
            mask_path=mask_path,
            mask_background_attenuation=mask_background_attenuation,
            mask_background_mode=mask_background_mode,
            mask_neutral_rgb=mask_neutral_rgb,
            mask_fill_rgb=mask_fill_rgb,
        )


def _crop_from_image(
    image: Image.Image,
    image_path: Path,
    bbox: list[float],
    bbox_context: float = 0.0,
    random_crop_margin: float = 0.0,
    mask_path: Path | None = None,
    mask_background_attenuation: float = 0.0,
    mask_background_mode: str = "none",
    mask_neutral_rgb: tuple[int, int, int] = IMAGENET_NEUTRAL_RGB,
    mask_fill_rgb: tuple[int, int, int] = BLACK_RGB,
    mask_cache: OrderedDict[Path, Image.Image] | None = None,
    mask_cache_size: int = 64,
    crop_box: tuple[int, int, int, int] | None = None,
) -> tuple[Image.Image, float]:
    width, height = image.size
    if crop_box is None:
        crop_box = _expanded_crop_box(
            bbox,
            width,
            height,
            bbox_context,
            random_crop_margin=random_crop_margin,
        )
    x1, y1, x2, y2 = crop_box
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid clipped bbox for {image_path}: {bbox}")
    crop = image.crop((x1, y1, x2, y2))
    resolved_mask_mode = resolve_mask_background_mode(mask_background_mode)
    if resolved_mask_mode != "none":
        if mask_path is None:
            raise FileNotFoundError(
                f"Mask path is required when mask background mode is enabled for {image_path}."
            )
        crop = apply_mask_background_mode(
            crop,
            mask_path=mask_path,
            crop_box=crop_box,
            mode=resolved_mask_mode,
            attenuation=mask_background_attenuation,
            neutral_rgb=mask_neutral_rgb,
            fill_rgb=mask_fill_rgb,
            mask_cache=mask_cache,
            mask_cache_size=mask_cache_size,
        )
    bbox_area = max(1.0, float(bbox[2]) * float(bbox[3]))
    crop_scale = ((x2 - x1) * (y2 - y1)) / bbox_area
    return crop, float(crop_scale)


def resolve_mask_background_mode(mode: str) -> str:
    if mode not in MASK_BACKGROUND_MODES:
        raise ValueError(f"mask_background_mode must be one of: {', '.join(MASK_BACKGROUND_MODES)}.")
    if mode == "random":
        return random.choice(RANDOM_MASK_BACKGROUND_MODES)
    if mode == "random_attenuate":
        return "attenuate" if random.random() < 0.5 else "none"
    return mode


def build_mask_paths_by_row_id(
    rows: list[dict[str, Any]],
    mask_dir: Path | None,
    mask_dir_root: Path | None,
) -> dict[str, Path]:
    if mask_dir is None and mask_dir_root is None:
        raise ValueError("mask_dir or mask_dir_root is required when mask attenuation is enabled.")

    counts_by_image: dict[str, int] = {}
    paths: dict[str, Path] = {}
    for row in rows:
        image_id = str(row["image_id"])
        mask_index = mask_index_for_row(row)
        if mask_index is None:
            counts_by_image[image_id] = counts_by_image.get(image_id, 0) + 1
            mask_index = counts_by_image[image_id]
        resolved_mask_dir = mask_dir or mask_dir_for_row(row, _require_path(mask_dir_root))
        paths[str(row["row_id"])] = (
            resolved_mask_dir
            / f"{Path(image_id).stem}_mask_{mask_index:02d}.png"
        )
    return paths


def mask_index_for_row(row: dict[str, Any]) -> int | None:
    value = row.get("instance_index")
    if value is None:
        return None
    try:
        return int(value) + 1
    except (TypeError, ValueError):
        return None


def mask_dir_for_row(row: dict[str, Any], mask_dir_root: Path) -> Path:
    image_parent = Path(row["image_path"]).parent.name
    if not image_parent.endswith("_images"):
        raise ValueError(f"Cannot infer mask split from image directory: {row['image_path']}")
    return mask_dir_root / f"{image_parent.removesuffix('_images')}_masks"


def _require_path(path: Path | None) -> Path:
    if path is None:
        raise ValueError("Expected a path.")
    return path


def attenuate_non_mask_regions(
    crop: Image.Image,
    mask_path: Path,
    crop_box: tuple[int, int, int, int],
    attenuation: float,
    neutral_rgb: tuple[int, int, int] = IMAGENET_NEUTRAL_RGB,
    fill_rgb: tuple[int, int, int] = BLACK_RGB,
    mask_cache: OrderedDict[Path, Image.Image] | None = None,
    mask_cache_size: int = 64,
) -> Image.Image:
    return apply_mask_background_mode(
        crop,
        mask_path=mask_path,
        crop_box=crop_box,
        mode="attenuate",
        attenuation=attenuation,
        neutral_rgb=neutral_rgb,
        fill_rgb=fill_rgb,
        mask_cache=mask_cache,
        mask_cache_size=mask_cache_size,
    )


def apply_mask_background_mode(
    crop: Image.Image,
    mask_path: Path,
    crop_box: tuple[int, int, int, int],
    mode: str,
    attenuation: float,
    neutral_rgb: tuple[int, int, int] = IMAGENET_NEUTRAL_RGB,
    fill_rgb: tuple[int, int, int] = BLACK_RGB,
    mask_cache: OrderedDict[Path, Image.Image] | None = None,
    mask_cache_size: int = 64,
) -> Image.Image:
    if mode == "none":
        return crop
    mask_image = load_mask_image(mask_path, mask_cache=mask_cache, mask_cache_size=mask_cache_size)
    mask_crop = mask_image.crop(crop_box)

    pixels = np.asarray(crop).copy()
    if mode == "soft":
        target_mask_float = np.asarray(mask_crop.filter(ImageFilter.GaussianBlur(radius=2.0)), dtype=np.float32) / 255.0
    else:
        target_mask_float = (np.asarray(mask_crop) > 0).astype(np.float32)
    target_mask = target_mask_float > 0
    if target_mask.shape != pixels.shape[:2]:
        raise ValueError(
            f"Mask crop shape {target_mask.shape} does not match image crop shape {pixels.shape[:2]}: {mask_path}"
        )
    pixels_float = pixels.astype(np.float32)
    neutral = np.asarray(neutral_rgb, dtype=np.float32)
    fill = np.asarray(fill_rgb, dtype=np.float32)
    if mode == "attenuate":
        background_mask = ~target_mask
        if not background_mask.any():
            return crop
        pixels_float[background_mask] = (
            ((1.0 - attenuation) * pixels_float[background_mask]) + (attenuation * neutral)
        )
    elif mode in {"hard", "soft"}:
        alpha = target_mask_float[..., None]
        pixels_float = (pixels_float * alpha) + (fill * (1.0 - alpha))
    else:
        raise ValueError("mode must be one of: none, attenuate, hard, soft.")
    pixels = np.rint(pixels_float).clip(0, 255).astype(np.uint8)
    return Image.fromarray(pixels, mode="RGB")


def mask_geometry_features(mask: np.ndarray, image_area: float, crop_scale: float) -> list[float]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return [0.0] * 7
    area = float(xs.size)
    x1 = float(xs.min())
    x2 = float(xs.max() + 1)
    y1 = float(ys.min())
    y2 = float(ys.max() + 1)
    bbox_width = max(1.0, x2 - x1)
    bbox_height = max(1.0, y2 - y1)
    width = max(1.0, float(mask.shape[1]))
    height = max(1.0, float(mask.shape[0]))
    centered_x = xs.astype(np.float32) - float(xs.mean())
    centered_y = ys.astype(np.float32) - float(ys.mean())
    if xs.size > 1:
        covariance = np.cov(np.stack([centered_x, centered_y], axis=0))
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        major = eigenvectors[:, int(np.argmax(eigenvalues))]
        angle = float(np.arctan2(major[1], major[0]))
    else:
        angle = 0.0
    return [
        area / max(1.0, image_area),
        (area * max(1.0, crop_scale)) / max(1.0, image_area),
        bbox_width / bbox_height,
        float(xs.mean()) / width,
        float(ys.mean()) / height,
        float(np.sin(angle)),
        float(np.cos(angle)),
    ]


def load_mask_image(
    mask_path: Path,
    mask_cache: OrderedDict[Path, Image.Image] | None = None,
    mask_cache_size: int = 64,
) -> Image.Image:
    if not mask_path.is_file():
        raise FileNotFoundError(f"Required SAM3 mask does not exist: {mask_path}")
    if mask_cache is not None and mask_cache_size > 0:
        cached = mask_cache.get(mask_path)
        if cached is not None:
            mask_cache.move_to_end(mask_path)
            return cached
    with Image.open(mask_path) as mask_image:
        loaded = mask_image.convert("L").copy()
    if mask_cache is not None and mask_cache_size > 0:
        mask_cache[mask_path] = loaded
        mask_cache.move_to_end(mask_path)
        while len(mask_cache) > mask_cache_size:
            mask_cache.popitem(last=False)
    return loaded


def _expanded_crop_box(
    bbox: list[float],
    image_width: int,
    image_height: int,
    bbox_context: float,
    random_crop_margin: float = 0.0,
) -> tuple[int, int, int, int]:
    x_min, y_min, box_width, box_height = bbox
    left_context = bbox_context
    right_context = bbox_context
    top_context = bbox_context
    bottom_context = bbox_context
    if random_crop_margin > 0:
        left_context += random.random() * random_crop_margin
        right_context += random.random() * random_crop_margin
        top_context += random.random() * random_crop_margin
        bottom_context += random.random() * random_crop_margin
    x1 = max(0, min(image_width, round(x_min - (box_width * left_context))))
    y1 = max(0, min(image_height, round(y_min - (box_height * top_context))))
    x2 = max(0, min(image_width, round(x_min + box_width + (box_width * right_context))))
    y2 = max(0, min(image_height, round(y_min + box_height + (box_height * bottom_context))))
    return x1, y1, x2, y2


def _join_known(*values: Any) -> str:
    if any(value is None for value in values):
        return "unknown"
    return "_".join(str(value) for value in values)
