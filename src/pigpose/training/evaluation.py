"""Evaluation metrics and artifact writers for PigPose model selection."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from pigpose.data.loaders import load_classes
from pigpose.training.metrics import (
    balanced_accuracy_from_predictions,
    macro_f1_from_predictions,
    macro_f1_present_classes_from_predictions,
)


def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    precision: str = "32-true",
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    """Collect predictions, labels, and source rows for one evaluation loader."""

    preds = []
    labels = []
    rows = loader.dataset.rows
    model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            context_images = batch.get("context_image")
            if context_images is not None:
                context_images = context_images.to(device)
            autocast_enabled = device.type == "cuda" and precision != "32-true"
            with torch.autocast(device_type=device.type, enabled=autocast_enabled):
                if "metadata" in batch and context_images is not None:
                    logits, _ = model(
                        images,
                        context_images,
                        batch["metadata"].to(device),
                        batch.get("metadata_continuous").to(device)
                        if batch.get("metadata_continuous") is not None
                        else None,
                    )
                elif context_images is None:
                    logits, _ = model(images)
                else:
                    logits, _ = model(images, context_images=context_images)
            preds.append(logits.argmax(dim=1).detach().cpu())
            labels.append(batch["label"].detach().cpu().long())
    return torch.cat(preds), torch.cat(labels), rows


def evaluate_predictions(
    preds: torch.Tensor,
    labels: torch.Tensor,
    rows: list[dict[str, Any]],
    num_classes: int = 5,
) -> dict[str, Any]:
    """Compute summary, per-class, subgroup, and confusion metrics."""

    class_names = load_classes()
    confusion = _confusion_matrix(preds, labels, num_classes)
    per_class = _per_class_metrics(confusion, class_names)
    subgroup_metrics = _subgroup_metrics(preds, labels, rows, num_classes)
    macro_f1 = macro_f1_from_predictions(preds, labels, num_classes).item()
    balanced_accuracy = balanced_accuracy_from_predictions(preds, labels, num_classes).item()
    sitting_recall = per_class[2]["recall"]
    left_f1 = per_class[0]["f1"]
    right_f1 = per_class[1]["f1"]
    left_right_confusion = int(confusion[0, 1].item() + confusion[1, 0].item())
    left_right_support = int(confusion[0].sum().item() + confusion[1].sum().item())
    left_right_confusion_rate = (
        left_right_confusion / left_right_support if left_right_support > 0 else 0.0
    )
    sternal_support = int(confusion[4].sum().item()) if confusion.size(0) > 4 else 0
    sternal_to_right = int(confusion[4, 1].item()) if confusion.size(0) > 4 else 0
    sternal_to_right_rate = sternal_to_right / sternal_support if sternal_support > 0 else 0.0
    domain_macro_f1 = {}
    domain_details = {}
    for domain in sorted({row.get("domain", "val") for row in rows}):
        indices = [idx for idx, row in enumerate(rows) if row.get("domain", "val") == domain]
        if indices:
            domain_preds = preds[indices]
            domain_labels = labels[indices]
            domain_confusion = _confusion_matrix(domain_preds, domain_labels, num_classes)
            domain_per_class = _per_class_metrics(domain_confusion, class_names)
            domain_macro_f1[domain] = macro_f1_from_predictions(
                domain_preds,
                domain_labels,
                num_classes,
            ).item()
            domain_details[domain] = {
                "macro_f1": domain_macro_f1[domain],
                "sitting_f1": domain_per_class[2]["f1"],
                "sternal_recall": domain_per_class[4]["recall"],
                "per_class": domain_per_class,
            }
    return {
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_accuracy,
        "domain_macro_f1": domain_macro_f1,
        "domains": domain_details,
        "critical": {
            "sitting_recall": sitting_recall,
            "lateral_lying_left_f1": left_f1,
            "lateral_lying_right_f1": right_f1,
            "left_right_confusion": left_right_confusion,
            "left_right_confusion_rate": left_right_confusion_rate,
            "sternal_to_right_confusion": sternal_to_right,
            "sternal_to_right_rate": sternal_to_right_rate,
        },
        "per_class": per_class,
        "subgroups": subgroup_metrics,
        "confusion_matrix": confusion.tolist(),
    }


def write_metric_artifacts(metrics: dict[str, Any], output_dir: Path) -> None:
    """Write JSON and CSV metric artifacts for one run."""

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics_summary.json").write_text(
        json.dumps(_jsonable(metrics), indent=2) + "\n",
        encoding="utf-8",
    )
    _write_per_class_csv(metrics["per_class"], output_dir / "per_class_metrics.csv")
    _write_subgroup_csv(metrics["subgroups"], output_dir / "subgroup_metrics.csv")
    _write_confusion_csv(metrics["confusion_matrix"], output_dir / "confusion_matrix.csv")


def write_error_diagnostics(
    preds: torch.Tensor,
    labels: torch.Tensor,
    rows: list[dict[str, Any]],
    output_dir: Path,
    logits: torch.Tensor | None = None,
    max_contact_sheet_items: int = 32,
) -> dict[str, Any]:
    diagnostics_dir = output_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    class_names = load_classes()
    probabilities = logits.softmax(dim=1).detach().cpu() if logits is not None else None
    misclassified = []
    confusion_counts: dict[tuple[int, int], int] = {}
    for index, (pred, label, row) in enumerate(zip(preds.tolist(), labels.tolist(), rows, strict=True)):
        pred = int(pred)
        label = int(label)
        if pred == label:
            continue
        confidence = float(probabilities[index, pred].item()) if probabilities is not None else None
        item = {
            "row_id": str(row.get("row_id", "")),
            "image_id": str(row.get("image_id", "")),
            "true_class_id": label,
            "true_class_name": class_names.get(label, str(label)),
            "pred_class_id": pred,
            "pred_class_name": class_names.get(pred, str(pred)),
            "subgroup": _subgroup_key(row),
            "bbox": json.dumps(row.get("bbox", [])),
            "confidence": "" if confidence is None else confidence,
            "image_path": str(row.get("image_path", "")),
        }
        misclassified.append(item)
        confusion_counts[(label, pred)] = confusion_counts.get((label, pred), 0) + 1
    misclassification_path = diagnostics_dir / "misclassifications.csv"
    _write_csv(
        misclassified,
        misclassification_path,
        [
            "row_id",
            "image_id",
            "true_class_id",
            "true_class_name",
            "pred_class_id",
            "pred_class_name",
            "subgroup",
            "bbox",
            "confidence",
            "image_path",
        ],
    )
    top_confusions = [
        {
            "true_class_id": true_id,
            "true_class_name": class_names.get(true_id, str(true_id)),
            "pred_class_id": pred_id,
            "pred_class_name": class_names.get(pred_id, str(pred_id)),
            "count": count,
        }
        for (true_id, pred_id), count in sorted(confusion_counts.items(), key=lambda item: (-item[1], item[0]))
    ]
    top_confusions_path = diagnostics_dir / "top_confusions.csv"
    _write_csv(
        top_confusions,
        top_confusions_path,
        ["true_class_id", "true_class_name", "pred_class_id", "pred_class_name", "count"],
    )
    contact_sheets = {}
    sheet_pairs = [(int(item["true_class_id"]), int(item["pred_class_id"])) for item in top_confusions[:3]]
    if (4, 1) in confusion_counts and (4, 1) not in sheet_pairs:
        sheet_pairs.append((4, 1))
    for true_id, pred_id in sheet_pairs:
        matching_rows = [
            row for row, pred, label in zip(rows, preds.tolist(), labels.tolist(), strict=True)
            if int(label) == true_id and int(pred) == pred_id
        ]
        sheet_path = diagnostics_dir / f"true_{true_id}_pred_{pred_id}.jpg"
        if _write_contact_sheet(matching_rows[:max_contact_sheet_items], sheet_path, class_names, true_id, pred_id):
            contact_sheets[f"true_{true_id}_pred_{pred_id}"] = str(sheet_path)
    return {
        "misclassifications_csv": str(misclassification_path),
        "top_confusions_csv": str(top_confusions_path),
        "contact_sheets": contact_sheets,
    }


def write_prediction_diagnostics(
    preds: torch.Tensor,
    labels: torch.Tensor,
    rows: list[dict[str, Any]],
    output_dir: Path,
    logits: torch.Tensor | None = None,
) -> str:
    """Write one validation prediction row per labeled target object."""

    diagnostics_dir = output_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    class_names = load_classes()
    probabilities = logits.softmax(dim=1).detach().cpu() if logits is not None else None
    records = []
    for index, (pred, label, row) in enumerate(zip(preds.tolist(), labels.tolist(), rows, strict=True)):
        pred_int = int(pred)
        label_int = int(label)
        record = {
            "row_id": str(row.get("row_id", "")),
            "image_id": str(row.get("image_id", "")),
            "true_class_id": label_int,
            "true_class_name": class_names.get(label_int, str(label_int)),
            "pred_class_id": pred_int,
            "pred_class_name": class_names.get(pred_int, str(pred_int)),
            "correct": int(pred_int == label_int),
            "subgroup": _subgroup_key(row),
            "bbox": json.dumps(row.get("bbox", [])),
            "image_path": str(row.get("image_path", "")),
        }
        if probabilities is not None:
            record["pred_confidence"] = float(probabilities[index, pred_int].item())
            for class_id in range(probabilities.size(1)):
                record[f"prob_class_{class_id}"] = float(probabilities[index, class_id].item())
        records.append(record)
    prediction_path = diagnostics_dir / "validation_predictions.csv"
    fieldnames = [
        "row_id",
        "image_id",
        "true_class_id",
        "true_class_name",
        "pred_class_id",
        "pred_class_name",
        "correct",
        "subgroup",
        "bbox",
        "image_path",
    ]
    if probabilities is not None:
        fieldnames.append("pred_confidence")
        fieldnames.extend(f"prob_class_{class_id}" for class_id in range(probabilities.size(1)))
    _write_csv(records, prediction_path, fieldnames)
    return str(prediction_path)


def _per_class_metrics(
    confusion: torch.Tensor,
    class_names: dict[int, str],
) -> list[dict[str, Any]]:
    metrics = []
    for class_id in range(confusion.size(0)):
        tp = confusion[class_id, class_id].float()
        fp = confusion[:, class_id].sum().float() - tp
        fn = confusion[class_id, :].sum().float() - tp
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        metrics.append(
            {
                "class_id": class_id,
                "class_name": class_names.get(class_id, str(class_id)),
                "precision": precision.item(),
                "recall": recall.item(),
                "f1": f1.item(),
                "support": int(confusion[class_id, :].sum().item()),
                "predicted": int(confusion[:, class_id].sum().item()),
            }
        )
    return metrics


def _subgroup_metrics(
    preds: torch.Tensor,
    labels: torch.Tensor,
    rows: list[dict[str, Any]],
    num_classes: int,
) -> list[dict[str, Any]]:
    indices_by_group: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        subgroup = "_".join(
            [
                str(row.get("pen")),
                str(row.get("camera_type")),
                str(row.get("camera")),
            ]
        )
        indices_by_group.setdefault(subgroup, []).append(index)
    return [
        {
            "subgroup": subgroup,
            "macro_f1": macro_f1_from_predictions(
                preds[indices],
                labels[indices],
                num_classes,
            ).item(),
            "macro_f1_all": macro_f1_from_predictions(
                preds[indices],
                labels[indices],
                num_classes,
            ).item(),
            "macro_f1_present": macro_f1_present_classes_from_predictions(
                preds[indices],
                labels[indices],
                num_classes,
            ).item(),
            "support": len(indices),
        }
        for subgroup, indices in sorted(indices_by_group.items())
    ]


def _confusion_matrix(
    preds: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for label, pred in zip(labels.tolist(), preds.tolist()):
        confusion[int(label), int(pred)] += 1
    return confusion


def _safe_div(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    if denominator.item() == 0:
        return numerator.new_zeros(())
    return numerator / denominator


def _write_per_class_csv(rows: list[dict[str, Any]], path: Path) -> None:
    _write_csv(
        rows,
        path,
        ["class_id", "class_name", "precision", "recall", "f1", "support", "predicted"],
    )


def _write_subgroup_csv(rows: list[dict[str, Any]], path: Path) -> None:
    _write_csv(rows, path, ["subgroup", "macro_f1", "macro_f1_all", "macro_f1_present", "support"])


def _write_confusion_csv(matrix: list[list[int]], path: Path) -> None:
    class_names = load_classes()
    rows = []
    for true_id, row in enumerate(matrix):
        item = {"true_class_id": true_id, "true_class_name": class_names.get(true_id, str(true_id))}
        for pred_id, count in enumerate(row):
            item[f"pred_{pred_id}"] = count
        rows.append(item)
    _write_csv(rows, path, ["true_class_id", "true_class_name", *[f"pred_{idx}" for idx in range(len(matrix))]])


def _subgroup_key(row: dict[str, Any]) -> str:
    return "_".join([str(row.get("pen")), str(row.get("camera_type")), str(row.get("camera"))])


def _write_contact_sheet(
    rows: list[dict[str, Any]],
    path: Path,
    class_names: dict[int, str],
    true_id: int,
    pred_id: int,
) -> bool:
    if not rows:
        return False
    tile_size = 192
    columns = min(4, len(rows))
    row_count = int(math.ceil(len(rows) / columns))
    sheet = Image.new("RGB", (columns * tile_size, row_count * tile_size), "white")
    for index, row in enumerate(rows):
        image_path = Path(str(row.get("image_path", "")))
        if not image_path.is_file():
            continue
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        bbox = row.get("bbox", [])
        if len(bbox) == 4:
            x, y, width, height = [float(value) for value in bbox]
            draw.rectangle([x, y, x + width, y + height], outline=(255, 64, 0), width=max(2, image.width // 256))
        label = f"T:{class_names.get(true_id, true_id)} P:{class_names.get(pred_id, pred_id)}"
        draw.rectangle([0, 0, image.width, 22], fill=(255, 255, 255))
        draw.text((4, 4), label, fill=(0, 0, 0))
        image.thumbnail((tile_size, tile_size))
        tile = Image.new("RGB", (tile_size, tile_size), "white")
        tile.paste(image, ((tile_size - image.width) // 2, (tile_size - image.height) // 2))
        x_offset = (index % columns) * tile_size
        y_offset = (index // columns) * tile_size
        sheet.paste(tile, (x_offset, y_offset))
    sheet.save(path, quality=92)
    return True


def _write_csv(rows: list[dict[str, Any]], path: Path, fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value
