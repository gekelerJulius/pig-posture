"""Export hard training examples from a manifest CSV as labeled crop images."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re
import textwrap

from PIL import Image, ImageDraw, ImageFont

MASK_OVERLAY_RGB = (105, 70, 230)
ROW_INSTANCE_PATTERN = re.compile(r"_(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/hard_examples/train_hard_examples.csv"),
        help="Hard-example CSV containing image_path, bbox, label, and reason columns.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/hard_examples/crops"),
        help="Directory where labeled crop images will be written.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of rows to export.")
    parser.add_argument("--padding", type=float, default=0.15, help="Relative bbox padding added around each crop.")
    parser.add_argument(
        "--mask-dir-root",
        type=Path,
        default=Path("Data"),
        help="Root directory used to locate SAM mask folders such as train2_masks.",
    )
    parser.add_argument(
        "--mask-overlay",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overlay the target pig SAM mask in transparent blue-violet.",
    )
    parser.add_argument("--mask-alpha", type=float, default=0.38, help="Mask overlay opacity in [0, 1].")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_manifest(args.manifest)
    if args.limit is not None:
        rows = rows[: args.limit]
    if not 0 <= args.mask_alpha <= 1:
        raise SystemExit("--mask-alpha must be in [0, 1].")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        export_row(
            row,
            args.output_dir,
            padding=args.padding,
            mask_dir_root=args.mask_dir_root,
            mask_overlay=args.mask_overlay,
            mask_alpha=args.mask_alpha,
        )
    print(f"Exported {len(rows)} hard-example crops to {args.output_dir}")


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def export_row(
    row: dict[str, str],
    output_dir: Path,
    padding: float,
    mask_dir_root: Path,
    mask_overlay: bool,
    mask_alpha: float,
) -> None:
    image_path = Path(row["image_path"])
    bbox = [float(row[key]) for key in ("bbox_x", "bbox_y", "bbox_w", "bbox_h")]
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
        crop_box = padded_box(bbox, image.size, padding)
        crop = image.crop(crop_box)
        mask_path = mask_path_for_row(row, mask_dir_root)
        if mask_overlay and mask_path is not None and mask_path.is_file():
            with Image.open(mask_path) as opened_mask:
                mask_crop = opened_mask.convert("L").crop(crop_box)
            crop = overlay_mask(crop, mask_crop, alpha=mask_alpha)
    annotated = annotate_crop(crop, row)
    rank = int(float(row["rank"]))
    safe_row_id = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in row["row_id"])
    annotated.save(output_dir / f"{rank:03d}_{safe_row_id}.jpg", quality=95)


def padded_box(bbox: list[float], image_size: tuple[int, int], padding: float) -> tuple[int, int, int, int]:
    x, y, width, height = bbox
    pad_x = width * padding
    pad_y = height * padding
    image_width, image_height = image_size
    left = max(0, int(round(x - pad_x)))
    top = max(0, int(round(y - pad_y)))
    right = min(image_width, int(round(x + width + pad_x)))
    bottom = min(image_height, int(round(y + height + pad_y)))
    return left, top, right, bottom


def mask_path_for_row(row: dict[str, str], mask_dir_root: Path) -> Path | None:
    image_path = Path(row["image_path"])
    image_parent = image_path.parent.name
    if not image_parent.endswith("_images"):
        return None
    match = ROW_INSTANCE_PATTERN.search(row["row_id"])
    if match is None:
        return None
    mask_index = int(match.group(1)) + 1
    mask_dir = mask_dir_root / f"{image_parent.removesuffix('_images')}_masks"
    return mask_dir / f"{image_path.stem}_mask_{mask_index:02d}.png"


def overlay_mask(crop: Image.Image, mask_crop: Image.Image, alpha: float) -> Image.Image:
    mask = mask_crop.point(lambda value: int(alpha * 255) if value > 0 else 0)
    overlay = Image.new("RGB", crop.size, color=MASK_OVERLAY_RGB)
    blended = crop.copy()
    blended.paste(overlay, mask=mask)
    return blended


def annotate_crop(crop: Image.Image, row: dict[str, str]) -> Image.Image:
    font = ImageFont.load_default()
    lines = [
        f"#{row['rank']} {row['true_class_name']} ({row['true_class_id']})",
        f"reasons: {row['hard_reasons']}",
        f"min_conf={float(row['min_confidence']):.3f} val_wrong={row['val_wrong_count']} fragile={row['fragile_changed_count']}",
        f"preds: {row['pred_classes_seen']}",
    ]
    wrapped_lines = []
    for line in lines:
        wrapped_lines.extend(textwrap.wrap(line, width=92) or [""])
    line_height = 14
    pad = 8
    banner_height = pad * 2 + line_height * len(wrapped_lines)
    canvas = Image.new("RGB", (crop.width, crop.height + banner_height), color=(255, 255, 255))
    canvas.paste(crop, (0, banner_height))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, crop.width, banner_height), fill=(255, 255, 255))
    for index, line in enumerate(wrapped_lines):
        draw.text((pad, pad + index * line_height), line, fill=(0, 0, 0), font=font)
    return canvas


if __name__ == "__main__":
    main()
