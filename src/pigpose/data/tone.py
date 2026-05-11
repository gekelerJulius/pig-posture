"""Source-to-target tone statistics and matching for PigPose RGB images."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def load_tone_stats(path: Path | str | None) -> dict[str, Any] | None:
    if path is None:
        return None
    stats_path = Path(path)
    if not stats_path.is_file():
        return None
    with stats_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def maybe_match_source_tone(
    image: Image.Image,
    row: dict[str, Any],
    tone_stats: dict[str, Any] | None,
    probability: float,
    strength_range: tuple[float, float],
) -> Image.Image:
    if tone_stats is None or str(row.get("domain")) != "source":
        return image
    if probability <= 0 or random.random() >= probability:
        return image
    source_stats = tone_stats_for_row(tone_stats, row, domain="source")
    target_stats = tone_stats_for_row(tone_stats, row, domain="target")
    if source_stats is None or target_stats is None:
        return image
    strength_min, strength_max = strength_range
    strength = random.uniform(strength_min, strength_max)
    return match_image_tone_lab(image, source_stats, target_stats, strength=strength)


def tone_stats_for_row(
    tone_stats: dict[str, Any] | None,
    row: dict[str, Any],
    domain: str,
    min_exact_count: int = 8,
) -> dict[str, Any] | None:
    if tone_stats is None:
        return None
    exact_key = tone_exact_key(row, domain)
    camera_key = tone_camera_key(row, domain)
    global_key = tone_global_key(domain)
    exact_stats = tone_stats.get("exact_subgroup", {}).get(exact_key)
    if exact_stats is not None and int(exact_stats.get("count", 0)) >= min_exact_count:
        return exact_stats
    for group_name, key in (("camera_view", camera_key), ("global", global_key)):
        stats = tone_stats.get(group_name, {}).get(key)
        if stats is not None and int(stats.get("count", 0)) > 0:
            return stats
    if exact_stats is not None and int(exact_stats.get("count", 0)) > 0:
        return exact_stats
    return None


def match_image_tone_lab(
    image: Image.Image,
    source_stats: dict[str, Any],
    target_stats: dict[str, Any],
    strength: float,
) -> Image.Image:
    strength = max(0.0, min(1.0, float(strength)))
    if strength == 0:
        return image
    cv2 = _import_cv2()
    rgb_pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
    lab_pixels = cv2.cvtColor(rgb_pixels, cv2.COLOR_RGB2LAB).astype(np.float32)
    source_mean = np.asarray(source_stats["mean"], dtype=np.float32).reshape(1, 1, 3)
    source_std = np.asarray(source_stats["std"], dtype=np.float32).reshape(1, 1, 3).clip(min=1e-3)
    target_mean = np.asarray(target_stats["mean"], dtype=np.float32).reshape(1, 1, 3)
    target_std = np.asarray(target_stats["std"], dtype=np.float32).reshape(1, 1, 3).clip(min=1e-3)
    matched_lab = (lab_pixels - source_mean) / source_std * target_std + target_mean
    blended_lab = lab_pixels * (1.0 - strength) + matched_lab * strength
    blended_lab = np.clip(blended_lab, 0, 255).astype(np.uint8)
    matched_rgb = cv2.cvtColor(blended_lab, cv2.COLOR_LAB2RGB)
    return Image.fromarray(matched_rgb, mode="RGB")


def image_lab_mean_std(path: Path) -> tuple[np.ndarray, np.ndarray]:
    cv2 = _import_cv2()
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
    lab_pixels = cv2.cvtColor(pixels, cv2.COLOR_RGB2LAB).astype(np.float32).reshape(-1, 3)
    return lab_pixels.mean(axis=0), lab_pixels.std(axis=0).clip(min=1e-3)


def tone_exact_key(row: dict[str, Any], domain: str) -> str:
    return "|".join(
        [
            str(domain),
            str(row.get("pen") or "unknown"),
            str(row.get("camera_type") or "unknown"),
            str(row.get("camera") or "unknown"),
        ]
    )


def tone_camera_key(row: dict[str, Any], domain: str) -> str:
    return "|".join(
        [
            str(domain),
            str(row.get("camera_type") or "unknown"),
            str(row.get("camera") or "unknown"),
        ]
    )


def tone_global_key(domain: str) -> str:
    return str(domain)


def _import_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for LAB tone matching. Install opencv-python.") from exc
    return cv2
