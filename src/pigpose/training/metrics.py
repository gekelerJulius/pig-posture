"""Classification and embedding metrics for PigPose training."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean()


def macro_f1_from_predictions(
    preds: torch.Tensor, labels: torch.Tensor, num_classes: int
) -> torch.Tensor:
    scores = []
    for class_id in range(num_classes):
        pred_pos = preds == class_id
        label_pos = labels == class_id
        tp = (pred_pos & label_pos).sum().float()
        fp = (pred_pos & ~label_pos).sum().float()
        fn = (~pred_pos & label_pos).sum().float()
        denom = (2 * tp) + fp + fn
        scores.append(
            torch.where(
                denom > 0, (2 * tp) / denom, torch.tensor(0.0, device=preds.device)
            )
        )
    return torch.stack(scores).mean()


def balanced_accuracy_from_predictions(
    preds: torch.Tensor, labels: torch.Tensor, num_classes: int
) -> torch.Tensor:
    recalls = []
    for class_id in range(num_classes):
        label_pos = labels == class_id
        support = label_pos.sum().float()
        tp = ((preds == class_id) & label_pos).sum().float()
        recalls.append(torch.where(support > 0, tp / support, torch.tensor(0.0, device=preds.device)))
    return torch.stack(recalls).mean()


def macro_f1_present_classes_from_predictions(
    preds: torch.Tensor, labels: torch.Tensor, num_classes: int
) -> torch.Tensor:
    scores = []
    for class_id in range(num_classes):
        pred_pos = preds == class_id
        label_pos = labels == class_id
        if not (pred_pos.any() or label_pos.any()):
            continue
        tp = (pred_pos & label_pos).sum().float()
        fp = (pred_pos & ~label_pos).sum().float()
        fn = (~pred_pos & label_pos).sum().float()
        denom = (2 * tp) + fp + fn
        scores.append(
            torch.where(
                denom > 0, (2 * tp) / denom, torch.tensor(0.0, device=preds.device)
            )
        )
    if not scores:
        return torch.tensor(0.0, device=preds.device)
    return torch.stack(scores).mean()


def per_class_classification_metrics(
    preds: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> dict[int, dict[str, torch.Tensor]]:
    metrics = {}
    for class_id in range(num_classes):
        pred_pos = preds == class_id
        label_pos = labels == class_id
        tp = (pred_pos & label_pos).sum().float()
        fp = (pred_pos & ~label_pos).sum().float()
        fn = (~pred_pos & label_pos).sum().float()
        support = label_pos.sum().float()
        precision_denominator = tp + fp
        recall_denominator = tp + fn
        precision = torch.where(
            precision_denominator > 0,
            tp / precision_denominator,
            torch.tensor(0.0, device=preds.device),
        )
        recall = torch.where(
            recall_denominator > 0,
            tp / recall_denominator,
            torch.tensor(0.0, device=preds.device),
        )
        f1_denominator = precision + recall
        f1 = torch.where(
            f1_denominator > 0,
            (2 * precision * recall) / f1_denominator,
            torch.tensor(0.0, device=preds.device),
        )
        metrics[class_id] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "predicted": pred_pos.sum().float(),
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
        }
    return metrics


def embedding_geometry_metrics(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, torch.Tensor]:
    normalized_embeddings = F.normalize(embeddings, dim=1)
    similarities = normalized_embeddings @ normalized_embeddings.T
    sample_count = labels.size(0)
    self_mask = torch.eye(sample_count, dtype=torch.bool, device=labels.device)

    leave_one_out_similarities = similarities.masked_fill(self_mask, -torch.inf)
    knn_1_predictions = labels[leave_one_out_similarities.argmax(dim=1)]
    prototype_acc = _leave_one_out_prototype_accuracy(normalized_embeddings, labels)

    distances = 1 - similarities
    same_label_mask = labels.unsqueeze(0).eq(labels.unsqueeze(1)) & ~self_mask
    diff_label_mask = labels.unsqueeze(0).ne(labels.unsqueeze(1))
    same_distances = distances[same_label_mask]
    same_distance_p90 = (
        same_distances.float().quantile(0.9)
        if same_distances.numel() > 0
        else embeddings.new_zeros(())
    )
    same_distance_mean = (
        same_distances.mean() if same_distances.numel() > 0 else embeddings.new_zeros(())
    )
    diff_distance_mean = (
        distances[diff_label_mask].mean()
        if diff_label_mask.any()
        else embeddings.new_zeros(())
    )

    classes = torch.unique(labels).sort().values
    if classes.numel() == 0:
        zero = embeddings.new_zeros(())
        return {
            "knn_1_acc": zero,
            "prototype_acc": zero,
            "same_distance_mean": zero,
            "same_distance_p90": zero,
            "center_distance_mean": zero,
            "center_distance_p90": zero,
            "distance_margin_mean": zero,
            "cluster_score": zero,
        }

    prototypes = []
    for class_label in classes:
        class_embeddings = normalized_embeddings[labels.eq(class_label)]
        prototypes.append(F.normalize(class_embeddings.mean(dim=0), dim=0))
    prototypes_tensor = torch.stack(prototypes, dim=0)

    target_indices = torch.empty(labels.size(0), dtype=torch.long, device=labels.device)
    for class_index, class_label in enumerate(classes):
        target_indices[labels.eq(class_label)] = class_index
    own_prototype_distances = 1 - (
        normalized_embeddings * prototypes_tensor[target_indices]
    ).sum(dim=1)

    if classes.numel() > 1:
        prototype_distances = 1 - prototypes_tensor @ prototypes_tensor.T
        prototype_self_mask = torch.eye(
            classes.numel(),
            dtype=torch.bool,
            device=labels.device,
        )
        prototype_distance_min = prototype_distances[~prototype_self_mask].min()
    else:
        prototype_distance_min = embeddings.new_zeros(())

    return {
        "knn_1_acc": knn_1_predictions.eq(labels).float().mean(),
        "prototype_acc": prototype_acc,
        "same_distance_mean": same_distance_mean,
        "same_distance_p90": same_distance_p90,
        "center_distance_mean": own_prototype_distances.mean(),
        "center_distance_p90": own_prototype_distances.float().quantile(0.9),
        "distance_margin_mean": diff_distance_mean - same_distance_mean,
        "cluster_score": prototype_distance_min - same_distance_p90,
    }


def _leave_one_out_prototype_accuracy(
    normalized_embeddings: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    classes = torch.unique(labels).sort().values
    if classes.numel() == 0:
        return normalized_embeddings.new_zeros(())
    predictions = []

    for sample_index, label in enumerate(labels):
        class_prototypes = []
        for class_label in classes:
            class_mask = labels.eq(class_label)
            if class_label == label and class_mask.sum() > 1:
                class_mask = class_mask.clone()
                class_mask[sample_index] = False
            prototype = normalized_embeddings[class_mask].mean(dim=0)
            class_prototypes.append(F.normalize(prototype, dim=0))

        prototypes = torch.stack(class_prototypes, dim=0)
        prototype_similarities = prototypes @ normalized_embeddings[sample_index]
        predictions.append(classes[prototype_similarities.argmax()])

    return torch.stack(predictions).eq(labels).float().mean()
