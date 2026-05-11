"""Write side-by-side previews for horizontal flip label mapping."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pigpose.config import DATA_DIR, OUTPUTS_DIR, timestamped_run_dir  # noqa: E402
from pigpose.data.loaders import load_classes, load_split  # noqa: E402
from pigpose.data.single_bbox_dataset import horizontally_flipped_label  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR / "label_flip_review")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--class-id", type=int, action="append", default=None, help="Restrict to one or more class IDs.")
    parser.add_argument("--crop-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = timestamped_run_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [row for row in load_split("train2", data_dir=args.data_dir).annotations if "class_id" in row]
    if args.class_id is not None:
        class_ids = set(args.class_id)
        rows = [row for row in rows if int(row["class_id"]) in class_ids]
    selected_rows = select_rows(rows, count=args.count, seed=args.seed)
    class_names = load_classes(args.data_dir)
    for index, row in enumerate(selected_rows):
        canvas = render_flip_preview(row, class_names=class_names, crop_size=args.crop_size)
        output_path = output_dir / f"flip_label_{index:03d}_row_{row['row_id']}.jpg"
        write_image(output_path, canvas)
    print(f"Wrote {len(selected_rows)} flip-label previews to {output_dir}")


def select_rows(rows: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    if count <= 0:
        return []
    rng = random.Random(seed)
    rows_by_class: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_class.setdefault(int(row["class_id"]), []).append(row)
    for class_rows in rows_by_class.values():
        rng.shuffle(class_rows)
    selected: list[dict[str, Any]] = []
    preferred_classes = [0, 1, *sorted(class_id for class_id in rows_by_class if class_id not in {0, 1})]
    while len(selected) < count:
        added = False
        for class_id in preferred_classes:
            class_rows = rows_by_class.get(class_id, [])
            if not class_rows:
                continue
            selected.append(class_rows.pop())
            added = True
            if len(selected) >= count:
                break
        if not added:
            break
    return selected


def render_flip_preview(row: dict[str, Any], class_names: dict[int, str], crop_size: int) -> Image.Image:
    with Image.open(row["image_path"]) as opened:
        image = opened.convert("RGB")
        crop = image.crop(crop_box(row["bbox"], image.size)).resize((crop_size, crop_size), Image.Resampling.BILINEAR)
    flipped = crop.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    label = int(row["class_id"])
    flipped_label = horizontally_flipped_label(label)
    panel_width = crop_size
    header_height = 54
    footer_height = 42
    gap = 12
    canvas = Image.new("RGB", (panel_width * 2 + gap, header_height + crop_size + footer_height), (245, 245, 245))
    canvas.paste(crop, (0, header_height))
    canvas.paste(flipped, (panel_width + gap, header_height))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw_text(draw, (8, 8), f"original: {label} {class_names.get(label, str(label))}", font)
    draw_text(draw, (panel_width + gap + 8, 8), f"flipped: {flipped_label} {class_names.get(flipped_label, str(flipped_label))}", font)
    draw_text(draw, (8, header_height + crop_size + 10), f"row_id={row['row_id']} image={row['image_id']}", font)
    return canvas


def crop_box(bbox: list[float], image_size: tuple[int, int]) -> tuple[int, int, int, int]:
    image_width, image_height = image_size
    x, y, width, height = [float(value) for value in bbox]
    x1 = max(0, min(image_width, round(x)))
    y1 = max(0, min(image_height, round(y)))
    x2 = max(0, min(image_width, round(x + width)))
    y2 = max(0, min(image_height, round(y + height)))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid bbox for preview: {bbox}")
    return x1, y1, x2, y2


def draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font: ImageFont.ImageFont) -> None:
    draw.text((xy[0] + 1, xy[1] + 1), text, fill=(255, 255, 255), font=font)
    draw.text(xy, text, fill=(20, 20, 20), font=font)


def write_image(path: Path, image: Image.Image) -> None:
    try:
        import cv2

        cv2.imwrite(str(path), np.asarray(image)[:, :, ::-1])
    except ImportError:
        image.save(path, quality=92)


if __name__ == "__main__":
    main()
