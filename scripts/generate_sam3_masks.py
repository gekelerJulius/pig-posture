"""Generate full-image SAM3 pig masks from PigPose bbox annotations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pigpose.config import DATA_DIR  # noqa: E402
from pigpose.data.loaders import load_split  # noqa: E402


DEFAULT_SPLITS = ("train1", "train2", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS), choices=DEFAULT_SPLITS)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--model", default="/home/juli/Projects/PigPose/sam3.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = load_sam_model(args.model)
    total_written = 0
    total_skipped = 0
    for split in args.splits:
        written, skipped = generate_split_masks(
            split=split,
            data_dir=args.data_dir,
            model=model,
            conf=args.conf,
            overwrite=args.overwrite,
        )
        total_written += written
        total_skipped += skipped
    print(f"Wrote {total_written} mask files; skipped {total_skipped} existing mask files.")


def load_sam_model(model_path: str) -> Any:
    from ultralytics import SAM

    return SAM(model_path)


def generate_split_masks(
    *,
    split: str,
    data_dir: Path,
    model: Any,
    conf: float,
    overwrite: bool = False,
) -> tuple[int, int]:
    dataset_split = load_split(split, data_dir=data_dir)
    rows_by_image = group_rows_by_image(dataset_split.annotations)
    mask_dir = data_dir / f"{split}_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0

    progress = tqdm(rows_by_image.items(), desc=f"{split}: Generating", unit="image")
    for image_id, rows in progress:
        output_paths = mask_output_paths(mask_dir, image_id, len(rows))
        existing = [path.exists() for path in output_paths]
        if all(existing) and not overwrite:
            skipped += len(output_paths)
            progress.set_description(f"{split}: Skipping {image_id}")
            continue

        progress.set_description(f"{split}: Generating {image_id}")
        masks = generate_image_masks(
            model=model,
            image_path=dataset_split.image_dir / image_id,
            rows=rows,
            conf=conf,
        )
        for mask, output_path, exists in zip(masks, output_paths, existing, strict=True):
            if exists and not overwrite:
                skipped += 1
                continue
            save_binary_mask(mask, output_path)
            written += 1

    return written, skipped


def group_rows_by_image(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    rows_by_image: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_image.setdefault(str(row["image_id"]), []).append(row)
    return rows_by_image


def mask_output_paths(mask_dir: Path, image_id: str, count: int) -> list[Path]:
    stem = Path(image_id).stem
    return [mask_dir / f"{stem}_mask_{index:02d}.png" for index in range(1, count + 1)]


def generate_image_masks(
    *,
    model: Any,
    image_path: Path,
    rows: Sequence[dict[str, Any]],
    conf: float,
) -> list[np.ndarray]:
    if not image_path.is_file():
        raise FileNotFoundError(f"Image does not exist: {image_path}")

    boxes = [
        bbox_xywh_to_xyxy(row["bbox"], int(row["width"]), int(row["height"]))
        for row in rows
    ]
    results = model(str(image_path), bboxes=boxes, task="segment", conf=conf)
    candidate_masks = masks_from_results(results)
    selected_masks = select_masks_for_boxes(candidate_masks, boxes, rows, image_path.name)
    return selected_masks


def masks_from_results(results: Any) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    for result in ensure_sequence(results):
        result_masks = getattr(result, "masks", None)
        if result_masks is None:
            continue
        data = getattr(result_masks, "data", None)
        if data is None:
            continue
        array = tensor_to_numpy(data)
        if array.ndim == 2:
            array = array[None, :, :]
        if array.ndim != 3:
            raise ValueError(f"Expected SAM masks with shape [N,H,W], got {array.shape}.")
        masks.extend(np.asarray(mask > 0, dtype=bool) for mask in array)
    return masks


def ensure_sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, (list, tuple)):
        return value
    return [value]


def tensor_to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def select_masks_for_boxes(
    candidate_masks: Sequence[np.ndarray],
    boxes: Sequence[Sequence[float]],
    rows: Sequence[dict[str, Any]],
    image_name: str,
) -> list[np.ndarray]:
    if len(candidate_masks) == len(boxes):
        selected = [np.asarray(mask, dtype=bool) for mask in candidate_masks]
    elif len(boxes) > 0 and len(candidate_masks) > len(boxes) and len(candidate_masks) % len(boxes) == 0:
        candidates_per_box = len(candidate_masks) // len(boxes)
        selected = []
        for index, box in enumerate(boxes):
            start = index * candidates_per_box
            candidates = candidate_masks[start:start + candidates_per_box]
            selected.append(best_mask_for_box(candidates, box))
    else:
        selected = assign_masks_to_boxes(candidate_masks, boxes)

    for mask, row in zip(selected, rows, strict=True):
        if mask.sum() == 0:
            raise RuntimeError(
                f"SAM3 returned an empty mask for image {image_name}, row_id={row['row_id']}."
            )
    return selected


def assign_masks_to_boxes(
    candidate_masks: Sequence[np.ndarray],
    boxes: Sequence[Sequence[float]],
) -> list[np.ndarray]:
    if len(candidate_masks) < len(boxes):
        raise RuntimeError(
            f"SAM3 returned {len(candidate_masks)} masks for {len(boxes)} prompted boxes."
        )

    unused = set(range(len(candidate_masks)))
    selected: list[np.ndarray] = []
    for box in boxes:
        best_index = max(
            unused,
            key=lambda mask_index: bbox_mask_score(candidate_masks[mask_index], box),
        )
        unused.remove(best_index)
        selected.append(np.asarray(candidate_masks[best_index], dtype=bool))
    return selected


def best_mask_for_box(
    candidate_masks: Sequence[np.ndarray],
    box: Sequence[float],
) -> np.ndarray:
    if not candidate_masks:
        raise RuntimeError("SAM3 returned no candidate masks for a prompted box.")
    return np.asarray(
        max(candidate_masks, key=lambda mask: bbox_mask_score(mask, box)),
        dtype=bool,
    )


def bbox_xywh_to_xyxy(
    bbox: Sequence[float],
    image_width: int,
    image_height: int,
) -> list[int]:
    x, y, width, height = [float(value) for value in bbox]
    x1 = round(x)
    y1 = round(y)
    x2 = round(x + width)
    y2 = round(y + height)
    return [
        max(0, min(image_width, x1)),
        max(0, min(image_height, y1)),
        max(0, min(image_width, x2)),
        max(0, min(image_height, y2)),
    ]


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return float(intersection / union) if union else 0.0


def mask_box(mask: np.ndarray) -> list[int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]


def mask_inside_fraction(mask: np.ndarray, bbox: Sequence[float]) -> float:
    x1, y1, x2, y2 = [int(round(value)) for value in bbox]
    total = float((mask > 0).sum())
    if total == 0:
        return 0.0
    inside = float((mask[y1:y2, x1:x2] > 0).sum())
    return inside / total


def bbox_mask_score(mask: np.ndarray, bbox: Sequence[float]) -> float:
    return box_iou(mask_box(mask), bbox) + 0.5 * mask_inside_fraction(mask, bbox)


def save_binary_mask(mask: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.where(mask > 0, 255, 0).astype(np.uint8)
    Image.fromarray(pixels, mode="L").save(output_path)


if __name__ == "__main__":
    main()
