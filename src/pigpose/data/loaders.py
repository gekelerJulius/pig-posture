"""Load the Kaggle-style PigPose data files described in README.md."""

from __future__ import annotations

import ast
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from pigpose.config import DATA_DIR


@dataclass(frozen=True)
class DatasetSplit:
    """A loaded annotation split and its matching image directory."""

    name: str
    annotations: list[dict[str, Any]]
    image_dir: Path


def load_classes(data_dir: Path = DATA_DIR) -> dict[int, str]:
    """Load posture class ID to label mapping."""

    class_file = data_dir / "pig_posture_classes.txt"
    with class_file.open("r", encoding="utf-8") as file:
        return {
            class_id: line.strip() for class_id, line in enumerate(file) if line.strip()
        }


def load_split(split: str, data_dir: Path = DATA_DIR) -> DatasetSplit:
    """Load one of train1, train2, or test from the Data directory."""

    if split not in {"train1", "train2", "test"}:
        raise ValueError(f"Unsupported split: {split}")

    csv_path = data_dir / f"{split}.csv"
    image_dir = data_dir / f"{split}_images"
    annotations = [_normalise_row(row, split, image_dir) for row in _read_csv(csv_path)]
    return DatasetSplit(name=split, annotations=annotations, image_dir=image_dir)


def load_all_splits(data_dir: Path = DATA_DIR) -> dict[str, DatasetSplit]:
    """Load all available dataset splits."""

    return {
        split: load_split(split, data_dir=data_dir)
        for split in ("train1", "train2", "test")
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _normalise_row(row: dict[str, str], split: str, image_dir: Path) -> dict[str, Any]:
    bbox = ast.literal_eval(row["bbox"])
    parsed_name = parse_image_filename(row["image_id"])
    item: dict[str, Any] = {
        "split": split,
        "row_id": row["row_id"],
        "instance_index": _parse_instance_index(row["row_id"]),
        "image_id": row["image_id"],
        "image_path": image_dir / row["image_id"],
        "width": int(row["width"]),
        "height": int(row["height"]),
        "bbox": [float(value) for value in bbox],
        "bbox_area": float(bbox[2]) * float(bbox[3]),
        "bbox_area_ratio": (float(bbox[2]) * float(bbox[3]))
        / (int(row["width"]) * int(row["height"])),
        **parsed_name,
    }
    if "class_id" in row and row["class_id"] != "":
        item["class_id"] = int(row["class_id"])
    return item


def parse_image_filename(image_id: str) -> dict[str, Any]:
    """Parse filenames like pen1_orb_cam1_20250108_085204.jpg."""

    stem = Path(image_id).stem
    parts = stem.split("_")
    if len(parts) != 5:
        return {
            "pen": None,
            "camera_type": None,
            "camera": None,
            "view_id": None,
            "scene_id": Path(image_id).stem,
            "captured_at": None,
        }

    pen, camera_type, camera, date_part, time_part = parts
    captured_at = datetime.strptime(f"{date_part}_{time_part}", "%Y%m%d_%H%M%S")
    view_id = f"{camera_type}_{camera}"
    scene_id = f"{pen}_{date_part}_{time_part}"
    return {
        "pen": pen,
        "camera_type": camera_type,
        "camera": camera,
        "view_id": view_id,
        "scene_id": scene_id,
        "captured_at": captured_at,
    }


def _parse_instance_index(row_id: str) -> str:
    return row_id.rsplit("_", maxsplit=1)[-1]
