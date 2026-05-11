from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from create_submission import (  # noqa: E402
    _parse_target_modules,
    _summary_bool,
    infer_simple_dino_classifier_head_layers,
    load_simple_dino_checkpoint,
    is_simple_dino_summary,
    parse_args,
    parse_bbox_contexts,
    prediction_config_for_checkpoint,
    resolve_tta_views,
    simple_dino_backbone_args_for_checkpoint,
)


def test_prediction_config_for_train_checkpoint_uses_crop_only_preprocessing(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "checkpoints" / "best.ckpt"
    checkpoint_path.parent.mkdir()
    checkpoint_path.write_bytes(b"checkpoint")
    summary_path = tmp_path / "run_summary.json"
    summary_path.write_text(
        """
{
  "args": {
    "backbone_source": "timm",
    "lora": true,
    "image_size": 384,
    "bbox_context": 0.1
  }
}
""".strip(),
        encoding="utf-8",
    )

    config = prediction_config_for_checkpoint(
        checkpoint_path=checkpoint_path,
        fallback_image_size=224,
        fallback_bbox_context=0.15,
        fallback_mask_dir_root=Path("Data"),
        fallback_mask_background_attenuation=0.75,
        fallback_mask_background_mode="soft",
    )

    assert config.image_size == 384
    assert config.bbox_context == 0.1
    assert config.bbox_contexts is None
    assert config.mask_dir_root is None
    assert config.mask_background_attenuation == 0.0
    assert config.mask_background_mode == "none"
    assert config.simple_crop


def test_prediction_config_for_train_checkpoint_uses_validation_mask_preprocessing(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "checkpoints" / "best.ckpt"
    checkpoint_path.parent.mkdir()
    checkpoint_path.write_bytes(b"checkpoint")
    (tmp_path / "run_summary.json").write_text(
        f"""
{{
  "args": {{
    "backbone_source": "timm",
    "image_size": 384,
    "bbox_context": 0.1,
    "mask_dir_root": "{tmp_path / "masks"}",
    "mask_background_attenuation": 0.4,
    "mask_background_mode": "attenuate",
    "val_mask_background_attenuation": null,
    "val_mask_background_mode": null
  }}
}}
""".strip(),
        encoding="utf-8",
    )

    config = prediction_config_for_checkpoint(
        checkpoint_path=checkpoint_path,
        fallback_image_size=224,
        fallback_bbox_context=0.15,
        fallback_mask_dir_root=Path("Data"),
        fallback_mask_background_attenuation=0.4,
        fallback_mask_background_mode="attenuate",
    )

    assert config.mask_dir_root == tmp_path / "masks"
    assert config.mask_background_attenuation == 0.4
    assert config.mask_background_mode == "attenuate"
    assert config.simple_crop


def test_simple_dino_summary_detection_uses_train_args() -> None:
    assert is_simple_dino_summary({"backbone_source": "timm"})
    assert is_simple_dino_summary({"lora": True})
    assert not is_simple_dino_summary({"model_arch": "convnext_tiny"})


def test_summary_bool_parses_json_and_string_values() -> None:
    assert _summary_bool(True)
    assert _summary_bool("true")
    assert _summary_bool("1")
    assert not _summary_bool("false")
    assert not _summary_bool(None)


def test_parse_target_modules_strips_empty_parts() -> None:
    assert _parse_target_modules("qkv, proj,,fc1") == ("qkv", "proj", "fc1")


def test_submission_defaults_target_two_train_folds_with_context_tta() -> None:
    args = parse_args([])

    assert args.ensemble_size == 2
    assert args.checkpoint_root == PROJECT_ROOT / "outputs" / "train"
    assert args.mask_background_attenuation == 0.4
    assert args.mask_background_mode == "attenuate"
    assert parse_bbox_contexts(args.bbox_contexts, fallback=args.bbox_context) == (0.05, 0.10, 0.15)


def test_simple_dino_defaults_to_flip_tta_views() -> None:
    assert resolve_tta_views(None, [], []) == 2


def test_parse_bbox_contexts_uses_fallback_for_empty_string() -> None:
    assert parse_bbox_contexts("", fallback=0.15) == (0.15,)


def test_simple_dino_checkpoint_loader_accepts_old_head_and_skips_aux(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class DummyBackbone(torch.nn.Module):
        output_size = 4

        def forward(self, images):
            return images

    checkpoint_path = tmp_path / "checkpoints" / "best.ckpt"
    checkpoint_path.parent.mkdir()
    torch.save(
        {
            "hyper_parameters": {
                "feature_dim": 4,
                "hidden_dim": 8,
                "num_classes": 5,
                "orientation_aux": True,
            },
            "state_dict": {
                "classifier.0.weight": torch.ones(4),
                "classifier.0.bias": torch.zeros(4),
                "classifier.1.weight": torch.ones(8, 4),
                "classifier.1.bias": torch.zeros(8),
                "classifier.4.weight": torch.ones(5, 8),
                "classifier.4.bias": torch.zeros(5),
                "orientation_aux_classifier.0.weight": torch.ones(4),
                "orientation_aux_classifier.0.bias": torch.zeros(4),
            },
        },
        checkpoint_path,
    )
    monkeypatch.setattr("create_submission.build_simple_backbone", lambda _summary_args: DummyBackbone())

    model = load_simple_dino_checkpoint(checkpoint_path, torch.device("cpu"))

    assert infer_simple_dino_classifier_head_layers(torch.load(checkpoint_path)["state_dict"]) == 2
    assert model.classifier[-1].weight.shape == (5, 8)
    assert not hasattr(model, "orientation_aux_classifier")
    assert model.hparams.orientation_aux is False


def test_simple_dino_loader_infers_lora_cls_mean_backbone_without_run_summary() -> None:
    args = simple_dino_backbone_args_for_checkpoint(
        summary_args={},
        hparams={"feature_dim": 2048},
        state_dict={
            "backbone.model.base_model.model.blocks.0.attn.qkv.lora_A.default.weight": torch.ones(8, 4),
        },
    )

    assert args["backbone_source"] == "timm"
    assert args["lora"] is True
    assert args["feature_pooling"] == "cls_mean"
