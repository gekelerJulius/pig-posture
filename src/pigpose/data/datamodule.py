"""Data loading for the compact PigPose training path."""

from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Sampler

from pigpose.config import DATA_DIR
from pigpose.data.loaders import load_split

METADATA_KEYS = ("camera_id", "sensor_pen_domain_id")
CONTINUOUS_METADATA_KEYS = (
    "bbox_x_norm",
    "bbox_y_norm",
    "bbox_w_norm",
    "bbox_h_norm",
    "num_pigs_in_image",
    "crop_scale",
    "mask_area_norm",
    "mask_crop_area_norm",
    "mask_bbox_aspect_ratio",
    "mask_centroid_x_norm",
    "mask_centroid_y_norm",
    "mask_orientation_sin",
    "mask_orientation_cos",
)
SINGLEPASS_CONTINUOUS_METADATA_KEYS = (
    "bbox_x1_norm",
    "bbox_y1_norm",
    "bbox_x2_norm",
    "bbox_y2_norm",
    "bbox_w_norm",
    "bbox_h_norm",
    "bbox_aspect_ratio",
    "num_pigs_in_image",
    "crop_scale",
    "mask_area_norm",
    "mask_bbox_aspect_ratio",
    "mask_centroid_x_norm",
    "mask_centroid_y_norm",
    "mask_orientation_sin",
    "mask_orientation_cos",
)
DEFAULT_METADATA_VOCAB = {
    "camera_id": {
        "unknown": 0,
        "orb_cam1": 1,
        "orb_cam2": 2,
        "tur_cam1": 3,
        "tur_cam2": 4,
    },
    "sensor_pen_domain_id": {
        value: index
        for index, value in enumerate(
            [
                "unknown",
                *[
                    f"{sensor}_{pen}_{domain}"
                    for domain in ("source", "target")
                    for pen in ("pen1", "pen2")
                    for sensor in ("orb", "tur")
                ],
            ]
        )
    },
}


class TargetFractionBatchSampler(Sampler[list[int]]):
    """Sample fixed source/target proportions from a source+target ConcatDataset."""

    def __init__(
        self,
        source_rows: list[dict[str, Any]],
        target_rows: list[dict[str, Any]],
        batch_size: int,
        target_fraction: float,
        seed: int,
        source_weights: list[float] | None = None,
        target_class_weight_overrides: dict[int, float] | None = None,
        target_subgroup_weight_overrides: dict[str, float] | None = None,
    ) -> None:
        if batch_size < 2:
            raise ValueError("batch_size must be at least 2.")
        if not source_rows or not target_rows:
            raise ValueError("source and target rows must be non-empty.")
        self.source_by_class = _indices_by_class(source_rows, offset=0)
        self.source_weights_by_class = _weights_by_class(source_rows, source_weights)
        self.target_by_class = _indices_by_class(target_rows, offset=len(source_rows))
        self.target_rows = target_rows
        self.target_row_ids = [str(row["row_id"]) for row in target_rows]
        self.source_classes = sorted(self.source_by_class)
        self.target_indices = [len(source_rows) + index for index in range(len(target_rows))]
        self.base_target_weights = torch.tensor(
            build_target_sample_weights(
                target_rows,
                class_weight_overrides=target_class_weight_overrides or DEFAULT_TARGET_CLASS_WEIGHTS,
                subgroup_weight_overrides=target_subgroup_weight_overrides or DEFAULT_TARGET_SUBGROUP_WEIGHTS,
            ),
            dtype=torch.double,
        )
        self.target_hard_multipliers = torch.ones(len(target_rows), dtype=torch.double)
        self.target_weights = self.base_target_weights.clone()
        self.batch_size = batch_size
        self.target_count = min(batch_size - 1, max(1, round(batch_size * target_fraction)))
        self.source_count = batch_size - self.target_count
        self.num_batches = max(1, round(len(target_rows) / self.target_count))
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        for _ in range(self.num_batches):
            batch = [
                _sample_class_balanced(self.source_by_class, self.source_classes, rng, self.source_weights_by_class)
                for _ in range(self.source_count)
            ]
            batch.extend(
                _sample_weighted(self.target_indices, self.target_weights, rng)
                for _ in range(self.target_count)
            )
            rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return self.num_batches

    def update_target_hard_multipliers(self, multipliers_by_row_id: dict[str, float]) -> None:
        multipliers = []
        for row_id in self.target_row_ids:
            multipliers.append(max(1.0, float(multipliers_by_row_id.get(row_id, 1.0))))
        self.target_hard_multipliers = torch.tensor(multipliers, dtype=torch.double)
        self.target_weights = self.base_target_weights * self.target_hard_multipliers


def load_train2_source_target_rows(
    data_dir: Path = DATA_DIR,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load train2 and mark rows as source/target using the explicit target image CSV."""

    target_image_ids = load_target_image_ids(data_dir)
    rows = [row for row in load_split("train2", data_dir=data_dir).annotations if "class_id" in row]
    rows = _with_num_pigs_in_image(rows)
    source_rows = [{**row, "domain": "source"} for row in rows if row["image_id"] not in target_image_ids]
    target_rows = [{**row, "domain": "target"} for row in rows if row["image_id"] in target_image_ids]
    if not source_rows or not target_rows:
        raise ValueError("train2 must contain both source and target-domain rows.")
    return source_rows, target_rows


def load_target_image_ids(data_dir: Path = DATA_DIR) -> set[str]:
    path = data_dir / "train2_test_domain_images.csv"
    with path.open("r", encoding="utf-8", newline="") as file:
        return {
            _clean_csv_row(row)["image_id"]
            for row in csv.DictReader(file)
            if _clean_csv_row(row).get("image_id")
        }


def load_pseudo_label_rows(
    pseudo_label_dir: Path,
    loss_weight: float,
    max_loss_weight: float | None = None,
    soft_targets: bool = True,
    soft_temperature: float = 1.0,
    data_dir: Path = DATA_DIR,
) -> list[dict[str, Any]]:
    path = pseudo_label_dir / "pseudo_labels.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Pseudo-label CSV does not exist: {path}")
    use_quality_weight = max_loss_weight is not None
    max_loss_weight = loss_weight if max_loss_weight is None else max_loss_weight
    soft_target_by_row_id = (
        load_pseudo_soft_targets(pseudo_label_dir, temperature=soft_temperature)
        if soft_targets
        else {}
    )
    test_rows = {
        row["row_id"]: row
        for row in _with_num_pigs_in_image(load_split("test", data_dir=data_dir).annotations)
    }
    pseudo_rows = []
    with path.open("r", encoding="utf-8", newline="") as file:
        for item in csv.DictReader(file):
            row_id = item["row_id"]
            if row_id not in test_rows:
                raise KeyError(f"Pseudo label row_id is not present in test.csv: {row_id}")
            confidence = float(item["confidence"])
            margin = _optional_float(item.get("margin"))
            entropy = _optional_float(item.get("entropy"))
            agreement = _optional_float(item.get("agreement"), default=1.0)
            min_member_confidence = _optional_float(item.get("min_member_confidence"), default=confidence)
            mean_member_confidence = _optional_float(item.get("mean_member_confidence"), default=confidence)
            quality = pseudo_quality_weight(
                confidence=confidence,
                margin=margin,
                entropy=entropy,
                agreement=agreement,
                min_member_confidence=min_member_confidence,
                mean_member_confidence=mean_member_confidence,
            )
            pseudo_rows.append(
                {
                    **test_rows[row_id],
                    "class_id": int(item["class_id"]),
                    "domain": "target",
                    "is_pseudo": True,
                    "pseudo_confidence": confidence,
                    "pseudo_margin": 0.0 if margin is None else margin,
                    "pseudo_entropy": 0.0 if entropy is None else entropy,
                    "pseudo_agreement": agreement,
                    "pseudo_min_member_confidence": min_member_confidence,
                    "pseudo_mean_member_confidence": mean_member_confidence,
                    "pseudo_quality_weight": quality,
                    "pseudo_has_soft_target": row_id in soft_target_by_row_id,
                    "pseudo_soft_target": soft_target_by_row_id.get(row_id),
                    "sample_loss_weight": (
                        loss_weight + (max_loss_weight - loss_weight) * quality
                        if use_quality_weight
                        else loss_weight * confidence
                    ),
                }
            )
    return pseudo_rows


def load_pseudo_soft_targets(pseudo_label_dir: Path, temperature: float) -> dict[str, list[float]]:
    logits_path = pseudo_label_dir / "logits.pt"
    if not logits_path.is_file():
        return {}
    if temperature <= 0:
        raise ValueError("pseudo soft target temperature must be positive.")
    payload = torch.load(logits_path, map_location="cpu")
    row_ids = [str(row_id) for row_id in payload.get("row_ids", [])]
    logits = payload.get("logits")
    if logits is None or len(row_ids) != int(logits.shape[0]):
        return {}
    probabilities = torch.softmax(logits.float() / float(temperature), dim=1)
    return {
        row_id: probabilities[index].tolist()
        for index, row_id in enumerate(row_ids)
    }


def _optional_float(value: Any, default: float | None = None) -> float | None:
    if value in (None, ""):
        return default
    return float(value)


def pseudo_quality_weight(
    *,
    confidence: float,
    margin: float | None,
    entropy: float | None,
    agreement: float | None,
    min_member_confidence: float | None,
    mean_member_confidence: float | None,
) -> float:
    margin_score = 1.0 if margin is None else min(1.0, max(0.0, margin / 0.85))
    entropy_score = 1.0 if entropy is None else min(1.0, max(0.0, 1.0 - entropy / math.log(5)))
    agreement_score = 1.0 if agreement is None else min(1.0, max(0.0, agreement))
    min_member_score = confidence if min_member_confidence is None else min(1.0, max(0.0, min_member_confidence))
    mean_member_score = confidence if mean_member_confidence is None else min(1.0, max(0.0, mean_member_confidence))
    confidence_score = min(1.0, max(0.0, confidence))
    quality = (
        confidence_score
        * confidence_score
        * margin_score
        * entropy_score
        * agreement_score
        * (0.35 + 0.65 * mean_member_score)
        * (0.60 + 0.40 * min_member_score)
    )
    return min(1.0, max(0.0, quality))


def summarize_pseudo_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"rows": 0}
    confidences = sorted(float(row.get("pseudo_confidence", 0.0)) for row in rows)
    return {
        "rows": len(rows),
        "class_counts": dict(sorted(Counter(int(row["class_id"]) for row in rows).items())),
        "confidence_min": confidences[0],
        "confidence_median": confidences[len(confidences) // 2],
        "confidence_max": confidences[-1],
        "soft_target_rows": sum(bool(row.get("pseudo_has_soft_target")) for row in rows),
    }


@dataclass(frozen=True)
class GroupedValidationSplit:
    train_rows: list[dict[str, Any]]
    val_rows: list[dict[str, Any]]
    val_group_keys: list[str]


def split_rows_by_group_fold(
    rows: list[dict[str, Any]],
    fold_count: int,
    fold_index: int,
) -> GroupedValidationSplit:
    """Split target rows by deterministic subgroup-session folds."""

    if fold_count < 2:
        raise ValueError("fold_count must be at least 2.")
    if not 0 <= fold_index < fold_count:
        raise ValueError("fold_index must be in [0, fold_count).")

    rows_by_group: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_group.setdefault(validation_group_key(row), []).append(row)
    if fold_count > len(rows_by_group):
        raise ValueError("fold_count cannot exceed the number of validation groups.")

    group_assignments = assign_validation_groups_to_folds(rows_by_group, fold_count)
    val_group_keys = sorted(
        group_key
        for group_key, assigned_fold in group_assignments.items()
        if assigned_fold == fold_index
    )
    val_group_key_set = set(val_group_keys)
    train_rows = [row for row in rows if validation_group_key(row) not in val_group_key_set]
    val_rows = [row for row in rows if validation_group_key(row) in val_group_key_set]
    if not train_rows or not val_rows:
        raise ValueError("Grouped validation split produced an empty train or validation set.")
    if _labels(train_rows) != _labels(rows):
        raise ValueError("Target train fold is missing at least one class.")
    if _labels(val_rows) != _labels(rows):
        raise ValueError("Target validation fold is missing at least one class.")
    return GroupedValidationSplit(train_rows, val_rows, val_group_keys)


def assign_validation_groups_to_folds(
    rows_by_group: dict[str, list[dict[str, Any]]],
    fold_count: int,
) -> dict[str, int]:
    """Assign groups to folds while prioritizing rare-class coverage."""

    fold_sizes = [0 for _ in range(fold_count)]
    fold_labels: list[set[int]] = [set() for _ in range(fold_count)]
    group_count_by_class = Counter(
        class_id
        for group_rows in rows_by_group.values()
        for class_id in {int(row["class_id"]) for row in group_rows}
    )
    assignments: dict[str, int] = {}
    for group_key, group_rows in sorted(
        rows_by_group.items(),
        key=lambda item: _group_assignment_sort_key(item, group_count_by_class),
    ):
        group_labels = {int(row["class_id"]) for row in group_rows}
        fold_index = min(
            range(fold_count),
            key=lambda index: (
                -len(group_labels - fold_labels[index]),
                fold_sizes[index],
                index,
            ),
        )
        assignments[group_key] = fold_index
        fold_sizes[fold_index] += len(group_rows)
        fold_labels[fold_index].update(group_labels)
    return assignments


def _group_assignment_sort_key(
    item: tuple[str, list[dict[str, Any]]],
    group_count_by_class: Counter[int],
) -> tuple[int, int, int, str]:
    group_key, group_rows = item
    group_labels = {int(row["class_id"]) for row in group_rows}
    rarest_class_group_count = min(group_count_by_class[class_id] for class_id in group_labels)
    return (
        rarest_class_group_count,
        -len(group_labels),
        -len(group_rows),
        group_key,
    )


DEFAULT_TARGET_CLASS_WEIGHTS = {
    0: 1.15,
    1: 1.10,
    2: 2.10,
    3: 1.0,
    4: 1.05,
}
DEFAULT_TARGET_SUBGROUP_WEIGHTS = {
    "pen1_tur_cam1": 1.40,
    "pen2_tur_cam2": 1.15,
}


def build_target_sample_weights(
    rows: list[dict[str, Any]],
    class_weight_overrides: dict[int, float],
    subgroup_weight_overrides: dict[str, float],
) -> list[float]:
    """Build target-row probabilities from inverse class frequency and weak-group boosts."""

    class_counts = Counter(int(row["class_id"]) for row in rows)
    weights = []
    for row in rows:
        class_id = int(row["class_id"])
        base = 1.0 / (class_counts[class_id] ** 0.5)
        class_boost = class_weight_overrides.get(class_id, 1.0)
        subgroup_boost = subgroup_weight_overrides.get(subgroup_key(row), 1.0)
        confidence_boost = float(row.get("pseudo_confidence", 1.0)) if row.get("is_pseudo") else 1.0
        weights.append(base * class_boost * subgroup_boost * confidence_boost)
    return weights


def subgroup_key(row: dict[str, Any]) -> str:
    return "_".join([str(row.get("pen")), str(row.get("camera_type")), str(row.get("camera"))])


def validation_group_key(row: dict[str, Any]) -> str:
    captured_at = row.get("captured_at")
    session_key = captured_at.strftime("%Y%m%d_%H") if captured_at is not None else "unknown_session"
    return f"{subgroup_key(row)}_{session_key}"


def _indices_by_class(rows: list[dict[str, Any]], offset: int) -> dict[int, list[int]]:
    indices: dict[int, list[int]] = {}
    for index, row in enumerate(rows):
        indices.setdefault(int(row["class_id"]), []).append(offset + index)
    return indices


def _weights_by_class(
    rows: list[dict[str, Any]],
    weights: list[float] | None,
) -> dict[int, torch.Tensor]:
    if weights is None:
        weights = [1.0 for _ in rows]
    values: dict[int, list[float]] = {}
    for row, weight in zip(rows, weights, strict=True):
        values.setdefault(int(row["class_id"]), []).append(float(weight))
    return {class_id: torch.tensor(class_weights, dtype=torch.double) for class_id, class_weights in values.items()}


def _sample_class_balanced(
    indices_by_class: dict[int, list[int]],
    classes: list[int],
    rng: random.Random,
    weights_by_class: dict[int, torch.Tensor] | None = None,
) -> int:
    selected_class = rng.choice(classes)
    indices = indices_by_class[selected_class]
    if weights_by_class is None:
        return rng.choice(indices)
    weights = weights_by_class[selected_class]
    generator = torch.Generator().manual_seed(rng.randrange(2**63 - 1))
    sampled = torch.multinomial(weights, num_samples=1, replacement=True, generator=generator)
    return indices[int(sampled.item())]


def _sample_weighted(indices: list[int], weights: torch.Tensor, rng: random.Random) -> int:
    generator = torch.Generator().manual_seed(rng.randrange(2**63 - 1))
    sampled = torch.multinomial(weights, num_samples=1, replacement=True, generator=generator)
    return indices[int(sampled.item())]


def _labels(rows: list[dict[str, Any]]) -> set[int]:
    return {int(row["class_id"]) for row in rows}


def _clean_csv_row(row: dict[str, str]) -> dict[str, str]:
    return {key.strip(): value.strip() for key, value in row.items()}


def _with_num_pigs_in_image(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter(str(row["image_id"]) for row in rows)
    return [{**row, "num_pigs_in_image": counts[str(row["image_id"])]} for row in rows]
