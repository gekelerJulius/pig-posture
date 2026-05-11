"""Precompute LAB tone statistics for source-to-target style matching."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pigpose.config import DATA_DIR  # noqa: E402
from pigpose.data.datamodule import load_train2_source_target_rows  # noqa: E402
from pigpose.data.loaders import load_split  # noqa: E402
from pigpose.data.tone import (  # noqa: E402
    image_lab_mean_std,
    match_image_tone_lab,
    tone_camera_key,
    tone_exact_key,
    tone_global_key,
    tone_stats_for_row,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output", type=Path, default=DATA_DIR / "tone_stats.json")
    parser.add_argument("--preview-dir", type=Path, default=None)
    parser.add_argument("--preview-count", type=int, default=12)
    parser.add_argument("--preview-strength", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = tone_rows(args.data_dir)
    print(f"Found {len(rows)} unique annotated images")
    stats = build_tone_stats(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(stats, file, indent=2, sort_keys=True)
        file.write("\n")
    preview_dir = args.preview_dir or args.output.with_suffix("").parent / f"{args.output.stem}_previews"
    write_tone_previews(
        rows,
        stats,
        preview_dir=preview_dir,
        count=args.preview_count,
        strength=args.preview_strength,
    )
    print(f"Wrote tone stats for {stats['summary']['images']} images to {args.output}")
    print(f"Wrote tone previews to {preview_dir}")


def tone_rows(data_dir: Path) -> list[dict[str, Any]]:
    source_rows, target_rows = load_train2_source_target_rows(data_dir=data_dir)
    test_rows = [{**row, "domain": "target"} for row in load_split("test", data_dir=data_dir).annotations]
    rows_by_image: dict[str, dict[str, Any]] = {}
    for row in [*source_rows, *target_rows, *test_rows]:
        rows_by_image.setdefault(str(row["image_path"]), row)
    return list(rows_by_image.values())


def build_tone_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    accumulators = {
        "exact_subgroup": {},
        "camera_view": {},
        "global": {},
    }
    image_count = 0
    skipped_count = 0
    progress = tqdm(
        rows,
        desc="Computing LAB tone stats",
        unit="image",
        dynamic_ncols=True,
    )
    for row in progress:
        image_path = Path(row["image_path"])
        if not image_path.is_file():
            skipped_count += 1
            progress.set_postfix(images=image_count, skipped=skipped_count)
            continue
        mean, std = image_lab_mean_std(image_path)
        image_count += 1
        domain = str(row.get("domain") or "unknown")
        update_accumulator(accumulators["exact_subgroup"], tone_exact_key(row, domain), mean, std)
        update_accumulator(accumulators["camera_view"], tone_camera_key(row, domain), mean, std)
        update_accumulator(accumulators["global"], tone_global_key(domain), mean, std)
        progress.set_postfix(images=image_count, skipped=skipped_count)
    return {
        "version": 2,
        "channels": "lab",
        "summary": {"images": image_count},
        **{
            group_name: {key: finalize_accumulator(value) for key, value in sorted(group.items())}
            for group_name, group in accumulators.items()
        },
    }


def write_tone_previews(
    rows: list[dict[str, Any]],
    stats: dict[str, Any],
    preview_dir: Path,
    count: int,
    strength: float,
) -> None:
    if count <= 0:
        return
    preview_dir.mkdir(parents=True, exist_ok=True)
    source_rows = [row for row in rows if str(row.get("domain")) == "source" and Path(row["image_path"]).is_file()]
    preview_rows = source_rows[:count]
    progress = tqdm(
        enumerate(preview_rows),
        total=len(preview_rows),
        desc="Writing tone previews",
        unit="preview",
        dynamic_ncols=True,
    )
    written_count = 0
    skipped_count = 0
    for index, row in progress:
        source_stats = tone_stats_for_row(stats, row, domain="source")
        target_stats = tone_stats_for_row(stats, row, domain="target")
        if source_stats is None or target_stats is None:
            skipped_count += 1
            progress.set_postfix(written=written_count, skipped=skipped_count)
            continue
        with Image.open(row["image_path"]) as opened:
            original = opened.convert("RGB").resize((320, 180))
        matched = match_image_tone_lab(original, source_stats, target_stats, strength=strength)
        canvas = Image.new("RGB", (640, 204), "white")
        canvas.paste(original, (0, 24))
        canvas.paste(matched, (320, 24))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 6), "source", fill=(0, 0, 0))
        draw.text((328, 6), "matched", fill=(0, 0, 0))
        canvas.save(preview_dir / f"tone_preview_{index:02d}.jpg", quality=92)
        written_count += 1
        progress.set_postfix(written=written_count, skipped=skipped_count)


def update_accumulator(
    group: dict[str, dict[str, Any]],
    key: str,
    mean: np.ndarray,
    std: np.ndarray,
) -> None:
    item = group.setdefault(
        key,
        {
            "count": 0,
            "mean_sum": np.zeros(3, dtype=np.float64),
            "std_sum": np.zeros(3, dtype=np.float64),
        },
    )
    item["count"] += 1
    item["mean_sum"] += mean
    item["std_sum"] += std


def finalize_accumulator(item: dict[str, Any]) -> dict[str, Any]:
    count = max(1, int(item["count"]))
    return {
        "count": count,
        "mean": (item["mean_sum"] / count).round(4).tolist(),
        "std": (item["std_sum"] / count).round(4).tolist(),
    }


if __name__ == "__main__":
    main()
