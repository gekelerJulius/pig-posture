"""Review generated SAM3 masks row-by-row with OpenCV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pigpose.config import DATA_DIR  # noqa: E402
from pigpose.data.loaders import load_split  # noqa: E402


BLUE_BGR = np.array([255, 0, 0], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train1", choices=["train1", "train2", "test"])
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--mask-dir", type=Path, default=None)
    parser.add_argument("--start", type=int, default=0, help="Zero-based row index to start from.")
    parser.add_argument("--alpha", type=float, default=0.18, help="Mask overlay opacity.")
    parser.add_argument("--thickness", type=int, default=1, help="Bbox rectangle thickness.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cv2 = import_cv2()
    split = load_split(args.split, data_dir=args.data_dir)
    mask_dir = args.mask_dir or (args.data_dir / f"{args.split}_masks")
    row_views = build_row_views(split.annotations, mask_dir)

    if args.start < 0 or args.start >= len(row_views):
        raise ValueError(f"--start must be between 0 and {len(row_views) - 1}.")

    window_name = "PigPose SAM3 mask review"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        for index, row_view in enumerate(row_views[args.start:], start=args.start):
            image = cv2.imread(str(row_view["image_path"]), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f"Could not read image: {row_view['image_path']}")
            mask = cv2.imread(str(row_view["mask_path"]), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(
                    f"Could not read mask for row_id={row_view['row_id']}: {row_view['mask_path']}"
                )

            rendered = render_overlay(
                image=image,
                mask=mask,
                bbox=row_view["bbox"],
                alpha=args.alpha,
                thickness=args.thickness,
                cv2=cv2,
            )
            cv2.setWindowTitle(
                window_name,
                f"{args.split} {index + 1}/{len(row_views)} {row_view['row_id']}",
            )
            cv2.imshow(window_name, rendered)
            key = cv2.waitKey(0) & 0xFF
            if key in {27, ord("q"), ord("Q")}:
                break
    finally:
        cv2.destroyAllWindows()


def import_cv2() -> Any:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "OpenCV is required for mask review. Install it with `uv add opencv-python` "
            "or run with an environment that provides `cv2`."
        ) from error
    return cv2


def build_row_views(rows: Iterable[dict[str, Any]], mask_dir: Path) -> list[dict[str, Any]]:
    per_image_counts: dict[str, int] = {}
    row_views: list[dict[str, Any]] = []
    for row in rows:
        image_id = str(row["image_id"])
        per_image_counts[image_id] = per_image_counts.get(image_id, 0) + 1
        mask_index = per_image_counts[image_id]
        row_views.append(
            {
                "row_id": row["row_id"],
                "image_path": row["image_path"],
                "mask_path": mask_path_for_row(mask_dir, image_id, mask_index),
                "bbox": row["bbox"],
            }
        )
    return row_views


def mask_path_for_row(mask_dir: Path, image_id: str, mask_index: int) -> Path:
    return mask_dir / f"{Path(image_id).stem}_mask_{mask_index:02d}.png"


def render_overlay(
    *,
    image: np.ndarray,
    mask: np.ndarray,
    bbox: list[float],
    alpha: float,
    thickness: int,
    cv2: Any,
) -> np.ndarray:
    if image.shape[:2] != mask.shape[:2]:
        raise ValueError(
            f"Image and mask dimensions differ: image={image.shape[:2]}, mask={mask.shape[:2]}"
        )
    if not 0 <= alpha <= 1:
        raise ValueError("--alpha must be between 0 and 1.")

    rendered = image.copy()
    mask_pixels = mask > 0
    if mask_pixels.any():
        rendered_float = rendered.astype(np.float32)
        rendered_float[mask_pixels] = (
            (1.0 - alpha) * rendered_float[mask_pixels] + alpha * BLUE_BGR
        )
        rendered = np.rint(rendered_float).clip(0, 255).astype(np.uint8)

    x1, y1, x2, y2 = bbox_xywh_to_xyxy(bbox, image.shape[1], image.shape[0])
    cv2.rectangle(rendered, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=thickness)
    return rendered


def bbox_xywh_to_xyxy(bbox: list[float], image_width: int, image_height: int) -> tuple[int, int, int, int]:
    x, y, width, height = [float(value) for value in bbox]
    x1 = max(0, min(image_width, round(x)))
    y1 = max(0, min(image_height, round(y)))
    x2 = max(0, min(image_width, round(x + width)))
    y2 = max(0, min(image_height, round(y + height)))
    return x1, y1, x2, y2


if __name__ == "__main__":
    main()
