"""Create pseudo labels for Kaggle test rows from the best available checkpoints."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from pigpose.config import DATA_DIR, OUTPUTS_DIR, timestamped_run_dir  # noqa: E402
from pigpose.data.loaders import load_split  # noqa: E402

from create_submission import (  # noqa: E402
    load_submission_model,
    parse_bbox_contexts,
    prediction_config_for_checkpoint,
    predict_logits,
    resolve_batch_size,
    resolve_class_bias_offsets,
    resolve_device,
    resolve_prediction_configs,
    resolve_tta_policy,
    resolve_tta_views,
    tta_transforms_for_policy,
)


RARE_CLASSES = {0, 1, 2}
CHECKPOINT_SCORE_PATTERN = re.compile(r"val_macro_f1=([0-9]+(?:\.[0-9]+)?)")
LOGGER = logging.getLogger("create_pseudo_labels")


@dataclass(frozen=True)
class CheckpointCandidate:
    path: Path
    score: float
    score_source: str
    run_dir: Path


@dataclass(frozen=True)
class SelectionReport:
    candidates: list[dict[str, Any]]
    selected: list[dict[str, Any]]
    pairwise_diversity: list[dict[str, Any]]
    probe_rows: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, action="append", default=None)
    parser.add_argument("--ensemble-size", type=int, default=3)
    parser.add_argument("--checkpoint-root", type=Path, default=OUTPUTS_DIR / "train")
    parser.add_argument("--selection-mode", choices=["score", "diverse"], default="diverse")
    parser.add_argument("--candidate-pool-size", type=int, default=8)
    parser.add_argument("--selection-min-score", type=float, default=0.80)
    parser.add_argument("--selection-max-score-drop", type=float, default=0.08)
    parser.add_argument("--selection-probe-rows", type=int, default=256)
    parser.add_argument("--selection-diversity-weight", type=float, default=0.35)
    parser.add_argument("--selection-min-diversity", type=float, default=0.0)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--bbox-context", type=float, default=0.10)
    parser.add_argument("--bbox-contexts", default="0.00,0.005,0.10")
    parser.add_argument("--mask-dir-root", type=Path, default=DATA_DIR)
    parser.add_argument("--mask-background-attenuation", type=float, default=0.5)
    parser.add_argument("--mask-background-mode", choices=["none", "attenuate", "hard", "soft"], default="soft")
    parser.add_argument("--preprocessing", choices=["auto", "args"], default="auto")
    parser.add_argument("--class-bias", choices=["auto", "none"], default="auto")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--precision", choices=["16-mixed", "bf16-mixed", "32-true"], default="16-mixed")
    parser.add_argument("--tta-views", type=int, choices=[1, 2, 4], default=2)
    parser.add_argument("--tta-policy", choices=["auto", "base", "base_flip", "base_zoom", "all"], default="base_flip")
    parser.add_argument("--base-threshold", type=float, default=0.90)
    parser.add_argument("--rare-threshold", type=float, default=0.85)
    parser.add_argument("--min-model-votes", type=int, default=3)
    parser.add_argument("--min-member-confidence", type=float, default=0.60)
    parser.add_argument("--min-mean-member-confidence", type=float, default=0.75)
    parser.add_argument("--min-margin", type=float, default=0.10)
    parser.add_argument("--max-entropy", type=float, default=None)
    parser.add_argument("--max-per-class", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    validate_args(args)
    torch.set_float32_matmul_precision("medium")
    output_dir = args.output_dir or timestamped_run_dir(DATA_DIR / "pseudo_labels")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    LOGGER.info("Writing pseudo-label artifacts to %s", output_dir)
    LOGGER.info("Using device: %s", device)
    test_rows = [{**row, "domain": "target"} for row in load_split("test").annotations]
    LOGGER.info("Loaded %d test rows", len(test_rows))
    checkpoint_paths, selection_report = resolve_pseudo_label_checkpoints(args, test_rows, device)
    if not checkpoint_paths:
        raise RuntimeError("No checkpoints were provided or discovered.")
    models = [load_submission_model(path, device) for path in checkpoint_paths]
    bbox_contexts = parse_bbox_contexts(args.bbox_contexts, fallback=args.bbox_context)
    tta_policy = resolve_tta_policy(args.tta_policy, checkpoint_paths)
    prediction_configs = resolve_prediction_configs(args, checkpoint_paths)
    batch_size = resolve_batch_size(args.batch_size, models, prediction_configs)
    tta_views = resolve_tta_views(args.tta_views, models, prediction_configs)
    class_bias_offsets = resolve_class_bias_offsets(args.class_bias, checkpoint_paths)
    log_inference_plan(
        checkpoint_paths=checkpoint_paths,
        tta_policy=tta_policy,
        tta_views=tta_views,
        bbox_contexts=bbox_contexts,
        batch_size=batch_size,
        selection_report=selection_report,
    )
    row_ids, member_logits = predict_member_logits(
        models=models,
        checkpoint_paths=checkpoint_paths,
        annotations=test_rows,
        tta_transforms=tta_transforms_for_policy(args.image_size, tta_policy, tta_views),
        batch_size=batch_size,
        num_workers=args.num_workers,
        device=device,
        bbox_context=args.bbox_context,
        bbox_contexts=bbox_contexts,
        mask_dir_root=args.mask_dir_root,
        mask_background_attenuation=args.mask_background_attenuation,
        mask_background_mode=args.mask_background_mode,
        class_bias_offsets=class_bias_offsets,
        prediction_configs=prediction_configs,
        tta_policy=tta_policy,
        tta_views=tta_views,
        precision=args.precision,
        progress_label="Final ensemble",
    )
    averaged_logits = member_logits.mean(dim=0)
    selected = select_pseudo_labels(
        row_ids=row_ids,
        logits=averaged_logits,
        member_logits=member_logits,
        base_threshold=args.base_threshold,
        rare_threshold=args.rare_threshold,
        min_model_votes=args.min_model_votes,
        min_member_confidence=args.min_member_confidence,
        min_mean_member_confidence=args.min_mean_member_confidence,
        min_margin=args.min_margin,
        max_entropy=args.max_entropy,
        max_per_class=args.max_per_class,
    )
    log_selection_summary(selected=selected, total_rows=len(test_rows))
    rows_by_id = {str(row["row_id"]): row for row in test_rows}
    pseudo_csv_path = output_dir / "pseudo_labels.csv"
    logits_path = output_dir / "logits.pt"
    summary_path = output_dir / "summary.json"
    write_pseudo_csv(pseudo_csv_path, selected, rows_by_id)
    torch.save({"row_ids": row_ids, "logits": averaged_logits, "member_logits": member_logits}, logits_path)
    write_summary(
        summary_path,
        args=args,
        checkpoint_paths=checkpoint_paths,
        selected=selected,
        total_rows=len(test_rows),
        tta_policy=tta_policy,
        tta_views=tta_views,
        selection_report=selection_report,
    )
    LOGGER.info("Wrote pseudo labels: %s", pseudo_csv_path)
    LOGGER.info("Wrote logits: %s", logits_path)
    LOGGER.info("Wrote summary: %s", summary_path)


def configure_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


def validate_args(args: argparse.Namespace) -> None:
    if args.ensemble_size <= 0:
        raise ValueError("--ensemble-size must be positive.")
    if args.candidate_pool_size <= 0:
        raise ValueError("--candidate-pool-size must be positive.")
    if args.selection_probe_rows < 0:
        raise ValueError("--selection-probe-rows must be non-negative.")
    for name in (
        "selection_min_score",
        "selection_max_score_drop",
        "selection_diversity_weight",
        "selection_min_diversity",
        "base_threshold",
        "rare_threshold",
        "min_member_confidence",
        "min_mean_member_confidence",
        "min_margin",
    ):
        value = float(getattr(args, name))
        if value < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative.")
    if args.max_entropy is not None and args.max_entropy < 0:
        raise ValueError("--max-entropy must be non-negative.")
    if args.max_per_class is not None and args.max_per_class <= 0:
        raise ValueError("--max-per-class must be positive when provided.")
    if args.min_model_votes is not None and args.min_model_votes <= 0:
        raise ValueError("--min-model-votes must be positive when provided.")


def log_candidates(candidates: list[CheckpointCandidate], title: str) -> None:
    LOGGER.info("%s:", title)
    for index, candidate in enumerate(candidates, start=1):
        LOGGER.info(
            "  %d. %s score=%.4f source=%s checkpoint=%s",
            index,
            candidate.run_dir.name,
            candidate.score,
            candidate.score_source,
            candidate.path.name,
        )


def log_inference_plan(
    checkpoint_paths: list[Path],
    tta_policy: str,
    tta_views: int,
    bbox_contexts: tuple[float, ...],
    batch_size: int,
    selection_report: SelectionReport | None,
) -> None:
    LOGGER.info("Final ensemble checkpoints:")
    for index, path in enumerate(checkpoint_paths, start=1):
        LOGGER.info("  %d. %s", index, path)
    LOGGER.info("Final inference TTA: policy=%s views=%d", tta_policy, tta_views)
    LOGGER.info("Final inference bbox contexts: %s", ", ".join(f"{value:.3f}" for value in bbox_contexts))
    LOGGER.info("Final inference batch size: %d", batch_size)
    if selection_report is not None and selection_report.pairwise_diversity:
        LOGGER.info("Top probe diversity pairs:")
        for row in selection_report.pairwise_diversity[:5]:
            LOGGER.info(
                "  %s vs %s diversity=%.4f",
                row["left_run"],
                row["right_run"],
                row["diversity"],
            )


def log_selection_summary(selected: list[dict[str, Any]], total_rows: int) -> None:
    class_counts = dict(sorted(Counter(int(row["class_id"]) for row in selected).items()))
    selected_fraction = 100.0 * len(selected) / total_rows if total_rows else 0.0
    LOGGER.info("Selected %d/%d pseudo labels (%.1f%%)", len(selected), total_rows, selected_fraction)
    LOGGER.info("Selected class counts: %s", class_counts)
    if not selected:
        return
    confidences = sorted(float(row["confidence"]) for row in selected)
    margins = sorted(float(row["margin"]) for row in selected)
    agreements = Counter(float(row["agreement"]) for row in selected)
    LOGGER.info(
        "Confidence min/median/max: %.4f / %.4f / %.4f",
        confidences[0],
        confidences[len(confidences) // 2],
        confidences[-1],
    )
    LOGGER.info(
        "Margin min/median/max: %.4f / %.4f / %.4f",
        margins[0],
        margins[len(margins) // 2],
        margins[-1],
    )
    LOGGER.info("Agreement distribution: %s", dict(sorted(agreements.items())))


def resolve_pseudo_label_checkpoints(
    args: argparse.Namespace,
    test_rows: list[dict[str, Any]],
    device: torch.device,
) -> tuple[list[Path], SelectionReport | None]:
    if args.checkpoint:
        LOGGER.info("Using %d manually supplied checkpoint(s); automatic selection disabled.", len(args.checkpoint))
        return args.checkpoint, None
    candidates = discover_pseudo_label_candidates(
        args.checkpoint_root,
        pool_size=max(args.candidate_pool_size, args.ensemble_size),
    )
    if not candidates:
        return [], None
    LOGGER.info("Discovered %d candidate run checkpoint(s) under %s", len(candidates), args.checkpoint_root)
    log_candidates(candidates, title="Candidate pool")
    if args.selection_mode == "score" or args.selection_probe_rows <= 0:
        selected = candidates[: args.ensemble_size]
        LOGGER.info("Selected checkpoints by validation score only.")
        log_candidates(selected, title="Selected checkpoints")
        return [candidate.path for candidate in selected], score_only_selection_report(candidates, selected)
    selected, report = select_diverse_checkpoints(
        candidates=candidates,
        args=args,
        test_rows=test_rows,
        device=device,
    )
    return [candidate.path for candidate in selected], report


def discover_pseudo_label_candidates(root: Path, pool_size: int) -> list[CheckpointCandidate]:
    if pool_size <= 0:
        raise ValueError("candidate-pool-size must be positive.")
    candidates: list[CheckpointCandidate] = []
    for run_dir in sorted(path for path in root.glob("*") if path.is_dir()):
        candidate = best_candidate_for_run(run_dir)
        if candidate is not None:
            candidates.append(candidate)
    candidates.sort(key=lambda candidate: (candidate.score, str(candidate.path)), reverse=True)
    return candidates[:pool_size]


def best_candidate_for_run(run_dir: Path) -> CheckpointCandidate | None:
    summary_path = run_dir / "run_summary.json"
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            summary = {}
        checkpoint_path = Path(str(summary.get("checkpoint_path", ""))).expanduser()
        if not checkpoint_path.is_absolute():
            checkpoint_path = (run_dir / checkpoint_path).resolve()
        score = score_from_metrics(summary.get("metrics", {}))
        if checkpoint_path.is_file() and score is not None:
            return CheckpointCandidate(checkpoint_path, score, "run_summary", run_dir)

    scored_checkpoints = []
    for checkpoint_path in sorted((run_dir / "checkpoints").glob("best*.ckpt")):
        score = score_from_checkpoint_name(checkpoint_path)
        if score is not None:
            scored_checkpoints.append((score, checkpoint_path))
    if scored_checkpoints:
        score, checkpoint_path = max(scored_checkpoints, key=lambda item: (item[0], str(item[1])))
        return CheckpointCandidate(checkpoint_path, score, "checkpoint_filename", run_dir)

    metrics_path = run_dir / "metrics_summary.json"
    score = score_from_json_file(metrics_path)
    checkpoint_paths = sorted((run_dir / "checkpoints").glob("best*.ckpt"))
    if score is not None and checkpoint_paths:
        return CheckpointCandidate(checkpoint_paths[0], score, "metrics_summary", run_dir)
    return None


def score_from_json_file(path: Path) -> float | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return score_from_metrics(payload)


def score_from_metrics(metrics: dict[str, Any]) -> float | None:
    for key in ("tta_macro_f1", "macro_f1"):
        value = metrics.get(key)
        if value is not None:
            return float(value)
    target_score = metrics.get("domain_macro_f1", {}).get("target")
    return float(target_score) if target_score is not None else None


def score_from_checkpoint_name(path: Path) -> float | None:
    match = CHECKPOINT_SCORE_PATTERN.search(path.name)
    return float(match.group(1)) if match else None


def score_only_selection_report(
    candidates: list[CheckpointCandidate],
    selected: list[CheckpointCandidate],
) -> SelectionReport:
    return SelectionReport(
        candidates=[candidate_record(candidate) for candidate in candidates],
        selected=[candidate_record(candidate) for candidate in selected],
        pairwise_diversity=[],
        probe_rows=0,
    )


def select_diverse_checkpoints(
    candidates: list[CheckpointCandidate],
    args: argparse.Namespace,
    test_rows: list[dict[str, Any]],
    device: torch.device,
) -> tuple[list[CheckpointCandidate], SelectionReport]:
    best_score = candidates[0].score
    score_floor = max(float(args.selection_min_score), best_score - float(args.selection_max_score_drop))
    eligible = [candidate for candidate in candidates if candidate.score >= score_floor]
    if len(eligible) < args.ensemble_size:
        eligible = candidates[: max(args.ensemble_size, len(eligible))]
    LOGGER.info(
        "Shortlisted %d/%d candidate(s) for diversity probe: score_floor=%.4f best_score=%.4f",
        len(eligible),
        len(candidates),
        score_floor,
        best_score,
    )
    probe_rows = deterministic_probe_rows(test_rows, limit=args.selection_probe_rows)
    LOGGER.info("Running diversity probe on %d row(s), base view, one bbox context.", len(probe_rows))
    row_ids, member_logits = predict_candidate_probe_logits(
        candidates=eligible,
        annotations=probe_rows,
        args=args,
        num_workers=args.num_workers,
        device=device,
    )
    probabilities = torch.softmax(member_logits, dim=2)
    diversity = pairwise_probe_diversity(probabilities)
    selected_indices = greedy_select_candidates(
        candidates=eligible,
        diversity=diversity,
        ensemble_size=args.ensemble_size,
        diversity_weight=float(args.selection_diversity_weight),
        min_diversity=float(args.selection_min_diversity),
    )
    selected = [eligible[index] for index in selected_indices]
    LOGGER.info("Selected diverse ensemble:")
    for rank, index in enumerate(selected_indices, start=1):
        candidate = eligible[index]
        nearest = min((float(diversity[index, other].item()) for other in selected_indices if other != index), default=0.0)
        LOGGER.info(
            "  %d. %s score=%.4f source=%s nearest_diversity=%.4f",
            rank,
            candidate.run_dir.name,
            candidate.score,
            candidate.score_source,
            nearest,
        )
    return selected, SelectionReport(
        candidates=[candidate_record(candidate) for candidate in candidates],
        selected=[
            {
                **candidate_record(candidate),
                "selection_rank": rank + 1,
            }
            for rank, candidate in enumerate(selected)
        ],
        pairwise_diversity=diversity_records(eligible, diversity),
        probe_rows=len(row_ids),
    )


def predict_candidate_probe_logits(
    candidates: list[CheckpointCandidate],
    annotations: list[dict[str, Any]],
    args: argparse.Namespace,
    num_workers: int,
    device: torch.device,
) -> tuple[list[str], torch.Tensor]:
    all_logits = []
    expected_row_ids: list[str] | None = None
    class_bias_offsets = resolve_class_bias_offsets(args.class_bias, [candidate.path for candidate in candidates])
    progress = tqdm(candidates, desc="Probe checkpoints", unit="ckpt", dynamic_ncols=True)
    for model_index, candidate in enumerate(progress):
        progress.set_postfix_str(f"{candidate.run_dir.name} f1={candidate.score:.4f}")
        LOGGER.info("Probe %s: %s", candidate.run_dir.name, candidate.path)
        model = load_submission_model(candidate.path, device)
        prediction_config = prediction_config_for_checkpoint(
            checkpoint_path=candidate.path,
            fallback_image_size=args.image_size,
            fallback_bbox_context=args.bbox_context,
            fallback_bbox_contexts=(args.bbox_context,),
            fallback_mask_dir_root=args.mask_dir_root,
            fallback_mask_background_attenuation=args.mask_background_attenuation,
            fallback_mask_background_mode=args.mask_background_mode,
        )
        row_ids, logits = predict_logits(
            models=[model],
            annotations=annotations,
            tta_transforms=tta_transforms_for_policy(prediction_config.image_size, "base", 1),
            batch_size=resolve_batch_size(args.batch_size, [model], [prediction_config]),
            num_workers=num_workers,
            device=device,
            bbox_context=args.bbox_context,
            bbox_contexts=(args.bbox_context,),
            mask_dir_root=args.mask_dir_root,
            mask_background_attenuation=args.mask_background_attenuation,
            mask_background_mode=args.mask_background_mode,
            class_bias_offsets=[class_bias_offsets[model_index]] if class_bias_offsets else None,
            prediction_configs=[prediction_config],
            tta_policy="base",
            tta_views=1,
            precision=args.precision,
        )
        if expected_row_ids is None:
            expected_row_ids = row_ids
        elif expected_row_ids != row_ids:
            raise RuntimeError(f"Probe row order changed for checkpoint: {candidate.path}")
        all_logits.append(logits.cpu())
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
    if expected_row_ids is None:
        raise RuntimeError("No probe predictions were produced.")
    return expected_row_ids, torch.stack(all_logits, dim=0)


def deterministic_probe_rows(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0 or len(rows) <= limit:
        return rows
    step = len(rows) / float(limit)
    return [rows[min(int(index * step), len(rows) - 1)] for index in range(limit)]


def pairwise_probe_diversity(probabilities: torch.Tensor) -> torch.Tensor:
    num_models, _, num_classes = probabilities.shape
    diversity = probabilities.new_zeros((num_models, num_models))
    log_classes = math.log(float(num_classes))
    predictions = probabilities.argmax(dim=2)
    for i in range(num_models):
        for j in range(i + 1, num_models):
            disagreement = predictions[i].ne(predictions[j]).float().mean()
            mixture = 0.5 * (probabilities[i] + probabilities[j])
            kl_i = kl_divergence(probabilities[i], mixture).mean()
            kl_j = kl_divergence(probabilities[j], mixture).mean()
            js = 0.5 * (kl_i + kl_j)
            value = (0.7 * disagreement) + (0.3 * (js / log_classes).clamp(min=0.0, max=1.0))
            diversity[i, j] = value
            diversity[j, i] = value
    return diversity.cpu()


def kl_divergence(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = left.clamp_min(1e-12)
    right = right.clamp_min(1e-12)
    return (left * (left.log() - right.log())).sum(dim=1)


def greedy_select_candidates(
    candidates: list[CheckpointCandidate],
    diversity: torch.Tensor,
    ensemble_size: int,
    diversity_weight: float,
    min_diversity: float,
) -> list[int]:
    if ensemble_size <= 0:
        raise ValueError("ensemble-size must be positive.")
    selected = [0]
    best_score = candidates[0].score
    worst_score = min(candidate.score for candidate in candidates)
    score_range = max(best_score - worst_score, 1e-6)
    while len(selected) < min(ensemble_size, len(candidates)):
        best_index = None
        best_value = -float("inf")
        for index, candidate in enumerate(candidates):
            if index in selected:
                continue
            min_distance = min(float(diversity[index, selected_index].item()) for selected_index in selected)
            if min_distance < min_diversity:
                continue
            quality = (candidate.score - worst_score) / score_range
            value = quality + (diversity_weight * min_distance)
            if value > best_value:
                best_value = value
                best_index = index
        if best_index is None:
            remaining = [index for index in range(len(candidates)) if index not in selected]
            if not remaining:
                break
            best_index = max(remaining, key=lambda index: candidates[index].score)
        selected.append(best_index)
    return selected


def candidate_record(candidate: CheckpointCandidate) -> dict[str, Any]:
    return {
        "run": candidate.run_dir.name,
        "checkpoint": str(candidate.path),
        "score": candidate.score,
        "score_source": candidate.score_source,
    }


def diversity_records(candidates: list[CheckpointCandidate], diversity: torch.Tensor) -> list[dict[str, Any]]:
    records = []
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            records.append(
                {
                    "left_run": candidates[i].run_dir.name,
                    "right_run": candidates[j].run_dir.name,
                    "diversity": float(diversity[i, j].item()),
                }
            )
    return sorted(records, key=lambda row: row["diversity"], reverse=True)


def predict_member_logits(
    models: list[Any],
    checkpoint_paths: list[Path],
    annotations: list[dict[str, Any]],
    tta_transforms: list[Any],
    batch_size: int,
    num_workers: int,
    device: torch.device,
    bbox_context: float,
    bbox_contexts: tuple[float, ...],
    mask_dir_root: Path | None,
    mask_background_attenuation: float,
    mask_background_mode: str,
    class_bias_offsets: list[list[float] | None] | None,
    prediction_configs: list[Any] | None,
    tta_policy: str,
    tta_views: int,
    precision: str,
    progress_label: str = "Checkpoints",
) -> tuple[list[str], torch.Tensor]:
    all_logits = []
    expected_row_ids: list[str] | None = None
    progress_items = list(enumerate(zip(models, checkpoint_paths, strict=True)))
    progress = tqdm(progress_items, desc=progress_label, unit="ckpt", dynamic_ncols=True)
    for model_index, (model, checkpoint_path) in progress:
        progress.set_postfix_str(checkpoint_path.parent.parent.name)
        LOGGER.info("Predicting checkpoint %d/%d: %s", model_index + 1, len(models), checkpoint_path)
        row_ids, logits = predict_logits(
            models=[model],
            annotations=annotations,
            tta_transforms=tta_transforms,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            bbox_context=bbox_context,
            bbox_contexts=bbox_contexts,
            mask_dir_root=mask_dir_root,
            mask_background_attenuation=mask_background_attenuation,
            mask_background_mode=mask_background_mode,
            class_bias_offsets=[class_bias_offsets[model_index]] if class_bias_offsets else None,
            prediction_configs=[prediction_configs[model_index]] if prediction_configs else None,
            tta_policy=tta_policy,
            tta_views=tta_views,
            precision=precision,
        )
        if expected_row_ids is None:
            expected_row_ids = row_ids
        elif expected_row_ids != row_ids:
            raise RuntimeError(f"Prediction row order changed for checkpoint: {checkpoint_path}")
        all_logits.append(logits.cpu())
    if expected_row_ids is None:
        raise RuntimeError("No predictions were produced.")
    return expected_row_ids, torch.stack(all_logits, dim=0)


def select_pseudo_labels(
    row_ids: list[str],
    logits: torch.Tensor,
    base_threshold: float,
    rare_threshold: float,
    max_per_class: int | None,
    member_logits: torch.Tensor | None = None,
    min_model_votes: int | None = None,
    min_member_confidence: float = 0.0,
    min_mean_member_confidence: float = 0.0,
    min_margin: float = 0.0,
    max_entropy: float | None = None,
) -> list[dict[str, Any]]:
    probabilities = torch.softmax(logits, dim=1)
    confidences, class_ids = probabilities.max(dim=1)
    top2 = probabilities.topk(k=min(2, probabilities.shape[1]), dim=1).values
    margins = top2[:, 0] - top2[:, 1] if top2.shape[1] > 1 else torch.ones_like(confidences)
    entropy = -(probabilities.clamp_min(1e-12) * probabilities.clamp_min(1e-12).log()).sum(dim=1)
    member_probabilities = torch.softmax(member_logits, dim=2) if member_logits is not None else None
    if member_probabilities is not None:
        member_classes = member_probabilities.argmax(dim=2)
        required_votes = min_model_votes or min(2, member_probabilities.shape[0])
        required_votes = max(1, min(int(required_votes), member_probabilities.shape[0]))
    else:
        member_classes = None
        required_votes = 1
    candidates_by_class: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for index, (row_id, class_id, confidence) in enumerate(
        zip(row_ids, class_ids.tolist(), confidences.tolist(), strict=True)
    ):
        class_id = int(class_id)
        threshold = rare_threshold if int(class_id) in RARE_CLASSES else base_threshold
        if float(confidence) < threshold or float(margins[index].item()) < min_margin:
            continue
        if max_entropy is not None and float(entropy[index].item()) > max_entropy:
            continue
        member_vote_count = 1
        member_min_confidence = float(confidence)
        member_mean_confidence = float(confidence)
        if member_classes is not None and member_probabilities is not None:
            votes = member_classes[:, index].eq(class_id)
            member_vote_count = int(votes.sum().item())
            if member_vote_count < required_votes:
                continue
            chosen_confidences = member_probabilities[:, index, class_id]
            member_min_confidence = float(chosen_confidences.min().item())
            member_mean_confidence = float(chosen_confidences.mean().item())
            if member_min_confidence < min_member_confidence:
                continue
            if member_mean_confidence < min_mean_member_confidence:
                continue
        candidates_by_class[class_id].append(
            {
                "row_id": row_id,
                "class_id": class_id,
                "confidence": float(confidence),
                "margin": float(margins[index].item()),
                "entropy": float(entropy[index].item()),
                "model_votes": member_vote_count,
                "model_count": int(member_logits.shape[0]) if member_logits is not None else 1,
                "agreement": member_vote_count / float(member_logits.shape[0]) if member_logits is not None else 1.0,
                "min_member_confidence": member_min_confidence,
                "mean_member_confidence": member_mean_confidence,
            }
        )
    cap = max_per_class
    if cap is None and candidates_by_class:
        cap = sorted(len(rows) for rows in candidates_by_class.values())[len(candidates_by_class) // 2]
    selected = []
    for class_id, rows in sorted(candidates_by_class.items()):
        rows = sorted(rows, key=lambda row: (-row["confidence"], row["row_id"]))
        selected.extend(rows[:cap] if cap is not None else rows)
    return sorted(selected, key=lambda row: row["row_id"])


def write_pseudo_csv(
    path: Path,
    selected: list[dict[str, Any]],
    rows_by_id: dict[str, dict[str, Any]],
) -> None:
    fieldnames = [
        "row_id",
        "image_id",
        "width",
        "height",
        "bbox",
        "class_id",
        "confidence",
        "margin",
        "entropy",
        "model_votes",
        "model_count",
        "agreement",
        "min_member_confidence",
        "mean_member_confidence",
        "source_split",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for item in selected:
            row = rows_by_id[item["row_id"]]
            writer.writerow(
                {
                    "row_id": row["row_id"],
                    "image_id": row["image_id"],
                    "width": row["width"],
                    "height": row["height"],
                    "bbox": row["bbox"],
                    "class_id": item["class_id"],
                    "confidence": item["confidence"],
                    "margin": item.get("margin"),
                    "entropy": item.get("entropy"),
                    "model_votes": item.get("model_votes"),
                    "model_count": item.get("model_count"),
                    "agreement": item.get("agreement"),
                    "min_member_confidence": item.get("min_member_confidence"),
                    "mean_member_confidence": item.get("mean_member_confidence"),
                    "source_split": "test",
                }
            )


def write_summary(
    path: Path,
    args: argparse.Namespace,
    checkpoint_paths: list[Path],
    selected: list[dict[str, Any]],
    total_rows: int,
    tta_policy: str,
    tta_views: int,
    selection_report: SelectionReport | None = None,
) -> None:
    confidences = sorted(float(row["confidence"]) for row in selected)
    margins = sorted(float(row["margin"]) for row in selected)
    mean_member_confidences = sorted(float(row["mean_member_confidence"]) for row in selected)
    summary = {
        "total_test_rows": total_rows,
        "selected_rows": len(selected),
        "checkpoints": [str(path) for path in checkpoint_paths],
        "tta_policy": tta_policy,
        "tta_views": tta_views,
        "class_counts": dict(sorted(Counter(int(row["class_id"]) for row in selected).items())),
        "confidence_min": confidences[0] if confidences else None,
        "confidence_median": confidences[len(confidences) // 2] if confidences else None,
        "confidence_max": confidences[-1] if confidences else None,
        "margin_min": margins[0] if margins else None,
        "margin_median": margins[len(margins) // 2] if margins else None,
        "mean_member_confidence_min": mean_member_confidences[0] if mean_member_confidences else None,
        "mean_member_confidence_median": mean_member_confidences[len(mean_member_confidences) // 2]
        if mean_member_confidences
        else None,
        "agreement_counts": dict(sorted(Counter(float(row["agreement"]) for row in selected).items())),
        "checkpoint_selection": {
            "candidates": selection_report.candidates,
            "selected": selection_report.selected,
            "pairwise_diversity": selection_report.pairwise_diversity,
            "probe_rows": selection_report.probe_rows,
        }
        if selection_report is not None
        else None,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
