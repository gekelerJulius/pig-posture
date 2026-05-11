"""Create a Kaggle submission from a compact PigPose checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pigpose.config import DATA_DIR, OUTPUTS_DIR, timestamped_run_dir  # noqa: E402
from pigpose.data.loaders import load_split  # noqa: E402
from pigpose.data.single_bbox_dataset import SingleBboxDataset  # noqa: E402
from pigpose.data.transforms import build_tta_transforms  # noqa: E402
from pigpose.models.dino_crop_classifier import Dinov3HubCropEncoder, SimpleDinoCropClassifier, TimmDinoCropEncoder  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        action="append",
        default=None,
        help="Checkpoint to include in the ensemble. Repeat to pass multiple checkpoints.",
    )
    parser.add_argument(
        "--ensemble-size",
        type=int,
        default=2,
        help="Number of best run-summary checkpoints to auto-ensemble when --checkpoint is omitted.",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=OUTPUTS_DIR / "train",
        help="Directory to search for run_summary checkpoints when --checkpoint is omitted.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Submission CSV path. Defaults to outputs/submissions/<timestamp>/T2_ensemble_submission.csv.",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--bbox-context", type=float, default=0.15)
    parser.add_argument(
        "--bbox-contexts",
        default="0.05,0.10,0.15",
        help="Comma-separated bbox context values to ensemble. Empty string uses --bbox-context only.",
    )
    parser.add_argument("--mask-dir-root", type=Path, default=DATA_DIR)
    parser.add_argument("--mask-background-attenuation", type=float, default=0.4)
    parser.add_argument("--mask-background-mode", choices=["none", "attenuate", "hard", "soft"], default="attenuate")
    parser.add_argument(
        "--preprocessing",
        choices=["auto", "args"],
        default="auto",
        help="Use each checkpoint's run_summary validation preprocessing, or the explicit CLI image/mask args for every model.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Prediction batch size. Defaults to 64 for train DINO checkpoints and 128 otherwise.",
    )
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--precision",
        choices=["16-mixed", "bf16-mixed", "32-true"],
        default="16-mixed",
        help="CUDA inference precision. CPU always uses 32-bit.",
    )
    parser.add_argument(
        "--tta-views",
        type=int,
        choices=[1, 2, 4],
        default=None,
        help="Number of TTA views. Defaults to 1 for train DINO checkpoints and 2 otherwise.",
    )
    parser.add_argument("--tta-policy", choices=["auto", "base", "base_flip", "base_zoom", "all"], default="base_flip")
    parser.add_argument(
        "--class-bias",
        choices=["auto", "none"],
        default="auto",
        help="Apply validation-calibrated class-bias offsets saved in run_summary.json when available.",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    torch.set_float32_matmul_precision("medium")
    checkpoint_paths = args.checkpoint or discover_best_checkpoints(args.checkpoint_root, limit=args.ensemble_size)
    if not checkpoint_paths:
        raise RuntimeError("No checkpoints were provided or discovered.")
    bbox_contexts = parse_bbox_contexts(args.bbox_contexts, fallback=args.bbox_context)
    device = resolve_device(args.device)
    models = [load_submission_model(path, device) for path in checkpoint_paths]
    print("Using checkpoints:")
    for path in checkpoint_paths:
        print(f"  {path}")
    tta_policy = resolve_tta_policy(args.tta_policy, checkpoint_paths)
    class_bias_offsets = resolve_class_bias_offsets(args.class_bias, checkpoint_paths)
    prediction_configs = resolve_prediction_configs(args, checkpoint_paths)
    batch_size = resolve_batch_size(args.batch_size, models, prediction_configs)
    tta_views = resolve_tta_views(args.tta_views, models, prediction_configs)
    print(f"Using TTA policy: {tta_policy} ({tta_views} view{'s' if tta_views != 1 else ''})")
    print(f"Using bbox contexts: {', '.join(f'{value:.2f}' for value in bbox_contexts)}")
    print(f"Using batch size: {batch_size}")
    annotations = [{**row, "domain": "target"} for row in load_split("test").annotations]
    row_ids, logits = predict_logits(
        models=models,
        annotations=annotations,
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
    )
    rows = rows_from_logits(row_ids, logits)
    default_output = timestamped_run_dir(OUTPUTS_DIR / "submissions") / "ensemble_submission.csv"
    output_path = t2_submission_path(args.output or default_output)
    write_submission(output_path, rows)
    print(f"Wrote {len(rows)} predictions to {output_path}")


def discover_best_checkpoints(outputs_dir: Path, limit: int = 3) -> list[Path]:
    if limit <= 0:
        raise ValueError("ensemble-size must be positive.")

    candidates: list[tuple[float, Path]] = []
    seen: set[tuple[str, float]] = set()
    for summary_path in sorted(outputs_dir.glob("**/run_summary.json")):
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        checkpoint_path = Path(str(summary.get("checkpoint_path", ""))).expanduser()
        if not checkpoint_path.is_absolute():
            checkpoint_path = (summary_path.parent / checkpoint_path).resolve()
        if not checkpoint_path.is_file():
            continue
        metrics = summary.get("metrics", {})
        score = metrics.get("tta_macro_f1")
        if score is None:
            score = metrics.get("macro_f1")
        if score is None:
            score = metrics.get("domain_macro_f1", {}).get("target")
        if score is None:
            continue
        score_value = float(score)
        dedupe_key = (str(checkpoint_path.resolve()), round(score_value, 6))
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        candidates.append((score_value, checkpoint_path))

    candidates.sort(key=lambda item: item[0], reverse=True)
    return [path for _, path in candidates[:limit]]


def predict_submission_rows(
    models: list[nn.Module],
    annotations: list[dict[str, Any]],
    tta_transforms: list[Any],
    batch_size: int,
    num_workers: int,
    device: torch.device,
    bbox_context: float,
    bbox_contexts: tuple[float, ...] | None = None,
    mask_dir_root: Path | None = DATA_DIR,
    mask_background_attenuation: float = 0.4,
    mask_background_mode: str = "attenuate",
    class_bias_offsets: list[list[float] | None] | list[float] | None = None,
) -> list[dict[str, int | str]]:
    row_ids, logits = predict_logits(
        models=models,
        annotations=annotations,
        tta_transforms=tta_transforms,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        bbox_context=bbox_context,
        bbox_contexts=bbox_contexts or (bbox_context,),
        mask_dir_root=mask_dir_root,
        mask_background_attenuation=mask_background_attenuation,
        mask_background_mode=mask_background_mode,
        class_bias_offsets=class_bias_offsets,
        precision="16-mixed",
    )
    return rows_from_logits(row_ids, logits)


TTA_POLICIES = {
    "base": [0],
    "base_flip": [0, 1],
    "base_zoom": [0, 2],
    "all": [0, 1, 2, 3],
}


def restore_horizontally_flipped_logits(logits: torch.Tensor) -> torch.Tensor:
    """Map predictions from a flipped image back to the original label space."""

    restored = logits.clone()
    restored[:, 0] = logits[:, 1]
    restored[:, 1] = logits[:, 0]
    return restored


@dataclass(frozen=True)
class PredictionConfig:
    image_size: int
    bbox_context: float
    bbox_contexts: tuple[float, ...] | None
    mask_dir_root: Path | None
    mask_background_attenuation: float
    mask_background_mode: str
    simple_crop: bool = False


def load_submission_model(checkpoint_path: Path, device: torch.device) -> nn.Module:
    if not is_simple_dino_checkpoint(checkpoint_path):
        raise ValueError(f"Unsupported checkpoint format after train.py removal: {checkpoint_path}")
    return load_simple_dino_checkpoint(checkpoint_path, device)


def is_simple_dino_checkpoint(checkpoint_path: Path) -> bool:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hparams = checkpoint.get("hyper_parameters", {})
    if "feature_dim" not in hparams or "context_image_size" in hparams:
        return False
    state_dict = checkpoint.get("state_dict", {})
    return any(str(key).startswith("backbone.") for key in state_dict)


def load_simple_dino_checkpoint(checkpoint_path: Path, device: torch.device) -> SimpleDinoCropClassifier:
    summary = run_summary_for_checkpoint(checkpoint_path)
    summary_args = summary.get("args", {}) if summary is not None else {}
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hparams = checkpoint.get("hyper_parameters", {})
    state_dict = checkpoint.get("state_dict", {})
    summary_args = simple_dino_backbone_args_for_checkpoint(summary_args, hparams, state_dict)
    backbone = build_simple_backbone(summary_args)
    classifier_head_layers = infer_simple_dino_classifier_head_layers(state_dict)
    feature_dim = int(hparams.get("feature_dim", getattr(backbone, "output_size")))
    hidden_dim = int(hparams.get("hidden_dim", infer_simple_dino_hidden_dim(state_dict, fallback=512)))
    num_classes = int(hparams.get("num_classes", infer_simple_dino_num_classes(state_dict, fallback=5)))
    model = SimpleDinoCropClassifier(
        backbone=backbone,
        feature_dim=feature_dim,
        num_classes=num_classes,
        learning_rate=float(hparams.get("learning_rate", 2e-4)),
        weight_decay=float(hparams.get("weight_decay", 0.01)),
        label_smoothing=float(hparams.get("label_smoothing", 0.05)),
        dropout=float(hparams.get("dropout", 0.2)),
        hidden_dim=hidden_dim,
        freeze_backbone=bool(hparams.get("freeze_backbone", True)),
        backbone_lr=hparams.get("backbone_lr"),
        lr_scheduler_monitor=hparams.get("lr_scheduler_monitor", "val_macro_f1"),
        lr_scheduler_type=str(hparams.get("lr_scheduler_type", "plateau")),
        plateau_factor=float(hparams.get("plateau_factor", 0.5)),
        plateau_patience=int(hparams.get("plateau_patience", 125)),
        plateau_threshold=float(hparams.get("plateau_threshold", 0.002)),
        plateau_min_lr_factor=float(hparams.get("plateau_min_lr_factor", 0.05)),
        step_lr_every_epochs=int(hparams.get("step_lr_every_epochs", 100)),
        step_lr_gamma=float(hparams.get("step_lr_gamma", 0.5)),
        orientation_aux=False,
        orientation_aux_weight=float(hparams.get("orientation_aux_weight", 0.3)),
        hard_sample_tracking=False,
        classifier_head_layers=classifier_head_layers,
    )
    filtered_state_dict = {
        key: value
        for key, value in state_dict.items()
        if not str(key).startswith("orientation_aux_classifier.")
    }
    model.load_state_dict(filtered_state_dict, strict=False)
    model.hparams.orientation_aux = False
    model.hparams.hard_sample_tracking = False
    return model.to(device).eval()


def simple_dino_backbone_args_for_checkpoint(
    summary_args: dict[str, Any],
    hparams: dict[str, Any],
    state_dict: dict[str, torch.Tensor],
) -> dict[str, Any]:
    args = dict(summary_args)
    if "model_name" not in args and "model_name" in hparams:
        args["model_name"] = hparams["model_name"]
    if "backbone_source" not in args and any(str(key).startswith("backbone.model.") for key in state_dict):
        args["backbone_source"] = "timm"
    if "lora" not in args and simple_dino_state_has_lora(state_dict):
        args["lora"] = True
    if "feature_pooling" not in args and "feature_dim" in hparams:
        feature_dim = int(hparams["feature_dim"])
        if feature_dim >= 2048 or _summary_bool(args.get("lora"), default=False):
            args["feature_pooling"] = "cls_mean"
    return args


def simple_dino_state_has_lora(state_dict: dict[str, torch.Tensor]) -> bool:
    return any(".lora_A." in str(key) or ".lora_B." in str(key) for key in state_dict)


def infer_simple_dino_classifier_head_layers(state_dict: dict[str, torch.Tensor]) -> int:
    if "classifier.7.weight" in state_dict:
        return 3
    if "classifier.4.weight" in state_dict:
        return 2
    return 3


def infer_simple_dino_hidden_dim(state_dict: dict[str, torch.Tensor], fallback: int) -> int:
    weight = state_dict.get("classifier.1.weight")
    if isinstance(weight, torch.Tensor) and weight.ndim == 2:
        return int(weight.shape[0])
    return fallback


def infer_simple_dino_num_classes(state_dict: dict[str, torch.Tensor], fallback: int) -> int:
    for key in ("classifier.7.weight", "classifier.4.weight"):
        weight = state_dict.get(key)
        if isinstance(weight, torch.Tensor) and weight.ndim == 2:
            return int(weight.shape[0])
    return fallback


def build_simple_backbone(summary_args: dict[str, Any]) -> TimmDinoCropEncoder | Dinov3HubCropEncoder:
    backbone_source = str(summary_args.get("backbone_source", "timm"))
    feature_pooling = str(summary_args.get("feature_pooling", "auto"))
    lora = _summary_bool(summary_args.get("lora"), default=False)
    if feature_pooling == "auto":
        feature_pooling = "cls_mean" if lora else "pooled"
    if backbone_source == "timm":
        return TimmDinoCropEncoder(
            model_name=str(summary_args.get("model_name", "vit_large_patch16_dinov3")),
            pretrained=False,
            feature_pooling=feature_pooling,
            lora=lora,
            lora_r=int(summary_args.get("lora_r", 8)),
            lora_alpha=int(summary_args.get("lora_alpha", 16)),
            lora_dropout=float(summary_args.get("lora_dropout", 0.05)),
            lora_target_modules=_parse_target_modules(str(summary_args.get("lora_target_modules", "qkv,proj"))),
        )
    return Dinov3HubCropEncoder(
        repo=str(summary_args.get("dinov3_repo", "facebookresearch/dinov3")),
        weights=str(summary_args.get("dinov3_weights")),
        model_name=str(summary_args.get("torchhub_model_name", "dinov3_vitl16")),
    )


def is_simple_dino_summary(summary_args: dict[str, Any]) -> bool:
    return "backbone_source" in summary_args or "lora" in summary_args or "feature_pooling" in summary_args


def is_simple_crop_model(model: nn.Module, config: PredictionConfig | None) -> bool:
    return isinstance(model, SimpleDinoCropClassifier)


def _summary_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _parse_target_modules(value: str) -> tuple[str, ...]:
    modules = tuple(part.strip() for part in value.split(",") if part.strip())
    if not modules:
        raise ValueError("LoRA target modules cannot be empty.")
    return modules


def tta_transforms_for_policy(image_size: int, policy: str, tta_views: int = 4) -> list[Any]:
    views = max(tta_views, 4 if policy in {"base_zoom", "all"} else 2)
    transforms = build_tta_transforms(image_size, views=views)
    indices = TTA_POLICIES.get(policy)
    if indices is None:
        raise ValueError(f"Unsupported TTA policy: {policy}")
    return [transforms[index] for index in indices if index < len(transforms)]


def resolve_tta_policy(requested_policy: str, checkpoint_paths: list[Path]) -> str:
    if requested_policy != "auto":
        return requested_policy
    for checkpoint_path in checkpoint_paths:
        summary_path = checkpoint_path.parent.parent / "run_summary.json"
        if not summary_path.is_file():
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        policy = summary.get("selected_tta_policy") or summary.get("metrics", {}).get("selected_tta_policy")
        if policy in TTA_POLICIES:
            return str(policy)
    return "all"


def resolve_class_bias_offsets(mode: str, checkpoint_paths: list[Path]) -> list[list[float] | None] | None:
    if mode == "none":
        return None
    offsets = [class_bias_offsets_for_checkpoint(path) for path in checkpoint_paths]
    return offsets if any(offset is not None for offset in offsets) else None


def class_bias_offsets_for_checkpoint(checkpoint_path: Path) -> list[float] | None:
    summary_path = checkpoint_path.parent.parent / "run_summary.json"
    if not summary_path.is_file():
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    offsets = summary.get("metrics", {}).get("class_bias_offsets")
    if not isinstance(offsets, list) or not offsets:
        return None
    return [float(value) for value in offsets]


def resolve_batch_size(
    requested_batch_size: int | None,
    models: list[nn.Module],
    prediction_configs: list[PredictionConfig] | None,
) -> int:
    if requested_batch_size is not None:
        return requested_batch_size
    if any(is_simple_crop_model(model, config) for model, config in zip(models, prediction_configs or [], strict=False)):
        return 32
    if any(isinstance(model, SimpleDinoCropClassifier) for model in models):
        return 32
    return 32


def resolve_tta_views(
    requested_tta_views: int | None,
    models: list[nn.Module],
    prediction_configs: list[PredictionConfig] | None,
) -> int:
    if requested_tta_views is not None:
        return requested_tta_views
    if any(is_simple_crop_model(model, config) for model, config in zip(models, prediction_configs or [], strict=False)):
        return 2
    if any(isinstance(model, SimpleDinoCropClassifier) for model in models):
        return 2
    return 2


def resolve_prediction_configs(args: argparse.Namespace, checkpoint_paths: list[Path]) -> list[PredictionConfig] | None:
    if args.preprocessing == "args":
        return None
    bbox_contexts = parse_bbox_contexts(args.bbox_contexts, fallback=args.bbox_context)
    return [
        prediction_config_for_checkpoint(
            checkpoint_path=path,
            fallback_image_size=args.image_size,
            fallback_bbox_context=args.bbox_context,
            fallback_bbox_contexts=bbox_contexts,
            fallback_mask_dir_root=args.mask_dir_root,
            fallback_mask_background_attenuation=args.mask_background_attenuation,
            fallback_mask_background_mode=args.mask_background_mode,
        )
        for path in checkpoint_paths
    ]


def prediction_config_for_checkpoint(
    checkpoint_path: Path,
    fallback_image_size: int,
    fallback_bbox_context: float,
    fallback_mask_dir_root: Path | None,
    fallback_mask_background_attenuation: float,
    fallback_mask_background_mode: str,
    fallback_bbox_contexts: tuple[float, ...] | None = None,
) -> PredictionConfig:
    summary = run_summary_for_checkpoint(checkpoint_path)
    summary_args = summary.get("args", {}) if summary is not None else {}
    simple_crop = is_simple_dino_summary(summary_args)
    image_size = int(summary_args.get("image_size", fallback_image_size))
    bbox_context = float(summary_args.get("bbox_context", fallback_bbox_context))
    bbox_contexts = fallback_bbox_contexts
    mask_dir_root_value = summary_args.get("mask_dir_root")
    mask_dir_root = Path(mask_dir_root_value) if mask_dir_root_value else fallback_mask_dir_root
    has_summary_mask_config = any(
        key in summary_args
        for key in (
            "mask_dir_root",
            "mask_background_attenuation",
            "mask_background_mode",
            "val_mask_background_attenuation",
            "val_mask_background_mode",
        )
    )
    if simple_crop and not has_summary_mask_config:
        mask_dir_root = None
        mask_background_attenuation = 0.0
        mask_background_mode = "none"
    else:
        mask_background_attenuation = _validation_mask_background_attenuation(
            summary_args,
            fallback=fallback_mask_background_attenuation,
        )
        mask_background_mode = _validation_mask_background_mode(
            summary_args,
            fallback=fallback_mask_background_mode,
        )
    return PredictionConfig(
        image_size=image_size,
        bbox_context=bbox_context,
        bbox_contexts=bbox_contexts,
        mask_dir_root=mask_dir_root,
        mask_background_attenuation=mask_background_attenuation,
        mask_background_mode=mask_background_mode,
        simple_crop=simple_crop,
    )


def run_summary_for_checkpoint(checkpoint_path: Path) -> dict[str, Any] | None:
    summary_path = checkpoint_path.parent.parent / "run_summary.json"
    if not summary_path.is_file():
        return None
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _validation_mask_background_attenuation(summary_args: dict[str, Any], fallback: float) -> float:
    value = summary_args.get("val_mask_background_attenuation")
    if value is None:
        value = summary_args.get("mask_background_attenuation", fallback)
    return float(value)


def _validation_mask_background_mode(summary_args: dict[str, Any], fallback: str) -> str:
    value = summary_args.get("val_mask_background_mode")
    if value is None:
        value = summary_args.get("mask_background_mode", fallback)
    value = str(value)
    if value == "random":
        return "soft"
    if value == "random_attenuate":
        return "attenuate"
    return value


def predict_logits(
    models: list[nn.Module],
    annotations: list[dict[str, Any]],
    tta_transforms: list[Any],
    batch_size: int,
    num_workers: int,
    device: torch.device,
    bbox_context: float,
    bbox_contexts: tuple[float, ...] | None = None,
    mask_dir_root: Path | None = DATA_DIR,
    mask_background_attenuation: float = 0.4,
    mask_background_mode: str = "attenuate",
    class_bias_offsets: list[list[float] | None] | list[float] | None = None,
    prediction_configs: list[PredictionConfig] | None = None,
    tta_policy: str = "base_flip",
    tta_views: int = 2,
    precision: str = "16-mixed",
) -> tuple[list[str], torch.Tensor]:
    if not models:
        raise ValueError("At least one model is required.")
    if prediction_configs is not None and len(prediction_configs) != len(models):
        raise ValueError("prediction_configs must contain one entry per model.")
    summed_logits: torch.Tensor | None = None
    row_ids: list[str] | None = None
    fallback_contexts = bbox_contexts or (bbox_context,)
    config_values = prediction_configs or [None for _ in models]
    can_reuse_loaders = len({_loader_reuse_key(config, tta_policy, tta_views, fallback_contexts) for config in config_values}) == 1
    if not can_reuse_loaders:
        transform_counts = [
            len(_transforms_for_model_config(config, tta_policy, tta_views, tta_transforms))
            * len(_bbox_contexts_for_model_config(config, fallback_contexts))
            for config in config_values
        ]
    else:
        transform_counts = [
            len(_transforms_for_model_config(config_values[0], tta_policy, tta_views, tta_transforms))
            * len(_bbox_contexts_for_model_config(config_values[0], fallback_contexts))
        ]
    prediction_count = 0
    total_batches = sum(transform_counts) * _num_batches(len(annotations), batch_size)
    with torch.inference_mode(), tqdm(total=total_batches, desc="Predicting", unit="batch") as progress:
        if can_reuse_loaders:
            config = config_values[0]
            model_transforms = _transforms_for_model_config(config, tta_policy, tta_views, tta_transforms)
            model_bbox_contexts = _bbox_contexts_for_model_config(config, fallback_contexts)
            for transform in model_transforms:
                for model_bbox_context in model_bbox_contexts:
                    loader = _build_loader_for_config(
                        annotations=annotations,
                        transform=transform,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        device=device,
                        bbox_context=model_bbox_context,
                        mask_dir_root=mask_dir_root,
                        mask_background_attenuation=mask_background_attenuation,
                        mask_background_mode=mask_background_mode,
                        config=config,
                    )
                    view_row_ids: list[str] = []
                    logits_by_model = [[] for _ in models]
                    for batch in loader:
                        view_row_ids.extend(batch["row_id"])
                        for model_index, model in enumerate(models):
                            logits = _predict_batch_logits(
                                model=model,
                                batch=batch,
                                device=device,
                                precision=precision,
                            )
                            if getattr(transform, "horizontally_flipped", False):
                                logits = restore_horizontally_flipped_logits(logits)
                            logits = apply_class_bias(logits, _bias_for_model(class_bias_offsets, model_index))
                            logits_by_model[model_index].append(logits.cpu())
                        progress.update()
                    if row_ids is None:
                        row_ids = view_row_ids
                    elif row_ids != view_row_ids:
                        raise RuntimeError("Prediction loaders produced inconsistent row order.")
                    for model_chunks in logits_by_model:
                        logits = torch.cat(model_chunks)
                        summed_logits = logits if summed_logits is None else summed_logits + logits
                        prediction_count += 1
        else:
            for model_index, model in enumerate(models):
                model_bias = _bias_for_model(class_bias_offsets, model_index)
                config = config_values[model_index]
                model_transforms = _transforms_for_model_config(config, tta_policy, tta_views, tta_transforms)
                model_bbox_contexts = _bbox_contexts_for_model_config(config, fallback_contexts)
                for transform in model_transforms:
                    for model_bbox_context in model_bbox_contexts:
                        loader = _build_loader_for_config(
                            annotations=annotations,
                            transform=transform,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            device=device,
                            bbox_context=model_bbox_context,
                            mask_dir_root=mask_dir_root,
                            mask_background_attenuation=mask_background_attenuation,
                            mask_background_mode=mask_background_mode,
                            config=config,
                        )
                        logits_chunks = []
                        view_row_ids = []
                        for batch in loader:
                            logits = _predict_batch_logits(
                                model=model,
                                batch=batch,
                                device=device,
                                precision=precision,
                            )
                            if getattr(transform, "horizontally_flipped", False):
                                logits = restore_horizontally_flipped_logits(logits)
                            logits = apply_class_bias(logits, model_bias)
                            logits_chunks.append(logits.cpu())
                            view_row_ids.extend(batch["row_id"])
                            progress.update()
                        logits = torch.cat(logits_chunks)
                        if row_ids is None:
                            row_ids = view_row_ids
                        elif row_ids != view_row_ids:
                            raise RuntimeError("Prediction loaders produced inconsistent row order.")
                        summed_logits = logits if summed_logits is None else summed_logits + logits
                        prediction_count += 1
    if summed_logits is None or row_ids is None:
        raise RuntimeError("No predictions were produced.")
    return row_ids, summed_logits / max(1, prediction_count)


def _transforms_for_model_config(
    config: PredictionConfig | None,
    tta_policy: str,
    tta_views: int,
    fallback_transforms: list[Any],
) -> list[Any]:
    if config is None:
        return fallback_transforms
    return tta_transforms_for_policy(config.image_size, tta_policy, tta_views)


def _bbox_contexts_for_model_config(
    config: PredictionConfig | None,
    fallback_contexts: tuple[float, ...],
) -> tuple[float, ...]:
    if config is None:
        return fallback_contexts
    return config.bbox_contexts or (config.bbox_context,)


def _loader_reuse_key(
    config: PredictionConfig | None,
    tta_policy: str,
    tta_views: int,
    fallback_contexts: tuple[float, ...],
) -> tuple[Any, ...]:
    contexts = _bbox_contexts_for_model_config(config, fallback_contexts)
    if config is None:
        return ("args", tta_policy, tta_views, contexts)
    return (
        config.image_size,
        tta_policy,
        tta_views,
        contexts,
        str(config.mask_dir_root),
        config.mask_background_attenuation,
        config.mask_background_mode,
        config.simple_crop,
    )


def _build_loader_for_config(
    annotations: list[dict[str, Any]],
    transform: Any,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    bbox_context: float,
    mask_dir_root: Path | None,
    mask_background_attenuation: float,
    mask_background_mode: str,
    config: PredictionConfig | None,
) -> DataLoader:
    return _build_loader(
        annotations=annotations,
        crop_transform=transform,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        bbox_context=bbox_context,
        mask_dir_root=mask_dir_root if config is None else config.mask_dir_root,
        mask_background_attenuation=mask_background_attenuation
        if config is None
        else config.mask_background_attenuation,
        mask_background_mode=mask_background_mode if config is None else config.mask_background_mode,
    )


def _predict_batch_logits(
    model: nn.Module,
    batch: dict[str, Any],
    device: torch.device,
    precision: str,
) -> torch.Tensor:
    autocast_dtype = torch.bfloat16 if precision == "bf16-mixed" else torch.float16
    autocast_enabled = device.type == "cuda" and precision != "32-true"
    with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_enabled):
        logits, _ = model(batch["image"].to(device, non_blocking=True))
    return logits


def parse_bbox_contexts(value: str | None, fallback: float) -> tuple[float, ...]:
    if value is None or value.strip() == "":
        return (float(fallback),)
    contexts = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not contexts:
        return (float(fallback),)
    if any(context < 0 for context in contexts):
        raise ValueError("--bbox-contexts values must be non-negative.")
    return contexts


def apply_class_bias(logits: torch.Tensor, class_bias: list[float] | None) -> torch.Tensor:
    if not class_bias:
        return logits
    bias = torch.as_tensor(class_bias, dtype=logits.dtype, device=logits.device)
    return logits + bias.view(1, -1)


def _bias_for_model(
    class_bias_offsets: list[list[float] | None] | list[float] | None,
    model_index: int,
) -> list[float] | None:
    if class_bias_offsets is None or not class_bias_offsets:
        return None
    if all(isinstance(value, (int, float)) for value in class_bias_offsets):
        return [float(value) for value in class_bias_offsets]  # type: ignore[arg-type]
    if model_index >= len(class_bias_offsets):  # type: ignore[arg-type]
        return None
    values = class_bias_offsets[model_index]  # type: ignore[index]
    return [float(value) for value in values] if values else None


def rows_from_logits(row_ids: list[str], logits: torch.Tensor) -> list[dict[str, int | str]]:
    return [
        {"row_id": row_id, "class_id": int(class_id)}
        for row_id, class_id in zip(row_ids, logits.argmax(dim=1).tolist(), strict=True)
    ]


def _build_loader(
    annotations: list[dict[str, Any]],
    crop_transform: Any,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    bbox_context: float,
    mask_dir_root: Path | None,
    mask_background_attenuation: float,
    mask_background_mode: str,
) -> DataLoader:
    return DataLoader(
        SingleBboxDataset(
            annotations,
            transform=crop_transform,
            bbox_context=bbox_context,
            mask_dir_root=mask_dir_root,
            mask_background_attenuation=mask_background_attenuation,
            mask_background_mode=mask_background_mode,
            overlap_reference_rows=annotations,
            include_mask_metadata=False,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda" and num_workers > 0,
        persistent_workers=num_workers > 0,
    )


def write_submission(output_path: Path, rows: list[dict[str, int | str]]) -> None:
    output_path = t2_submission_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["row_id", "class_id"])
        writer.writeheader()
        writer.writerows(rows)


def t2_submission_path(path: Path) -> Path:
    if path.name.startswith("T2_"):
        return path
    return path.with_name(f"T2_{path.name}")


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but CUDA is not available.")
    return torch.device(device)


def _num_batches(num_items: int, batch_size: int) -> int:
    return (num_items + batch_size - 1) // batch_size


if __name__ == "__main__":
    main()
