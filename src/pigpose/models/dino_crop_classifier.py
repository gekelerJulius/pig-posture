"""Crop-only DINO posture classifier."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import lightning as L
from peft import LoraConfig, get_peft_model
import timm
import torch
from torch import nn
from torch.nn import functional as F

from pigpose.data.transforms import IMAGENET_MEAN, IMAGENET_STD
from pigpose.training.metrics import accuracy, balanced_accuracy_from_predictions, macro_f1_from_predictions


POSTURE_CLASS_NAMES = ("lateral_left", "lateral_right", "sitting", "standing", "sternal")
KEY_TENSORBOARD_METRICS = {
    "val_macro_f1": "key/01_val_macro_f1",
    "val_acc": "key/02_val_acc",
    "val_balanced_acc": "key/03_val_balanced_acc",
    "train_target_acc": "key/04_train_target_acc",
    "train_source_acc": "key/05_train_source_acc",
}


class TimmDinoCropEncoder(nn.Module):
    """timm-hosted DINOv3 feature extractor returning pooled embeddings."""

    def __init__(
        self,
        model_name: str = "vit_large_patch16_dinov3",
        pretrained: bool = True,
        feature_pooling: str = "pooled",
        lora: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: tuple[str, ...] = ("qkv", "proj"),
    ) -> None:
        super().__init__()
        if feature_pooling not in {"pooled", "cls_mean"}:
            raise ValueError("feature_pooling must be one of: pooled, cls_mean.")
        self.model_name = normalize_timm_model_name(model_name)
        self.feature_pooling = feature_pooling
        model_kwargs: dict[str, Any] = {"pretrained": pretrained, "num_classes": 0}
        if feature_pooling == "cls_mean":
            model_kwargs["global_pool"] = ""
        self.model = timm.create_model(self.model_name, **model_kwargs)
        self.output_size = int(getattr(self.model, "num_features", 0))
        if self.output_size <= 0:
            raise ValueError(f"Could not infer timm feature dim for model_name={model_name!r}.")
        if feature_pooling == "cls_mean":
            self.output_size *= 2
        if lora:
            freeze_module(self.model)
            resolved_targets = resolve_lora_linear_target_modules(self.model, lora_target_modules)
            self.model = get_peft_model(
                self.model,
                LoraConfig(
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    bias="none",
                    target_modules=list(resolved_targets),
                ),
            )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.feature_pooling == "pooled":
            return self.model(images)
        output = self.model.forward_features(images)
        return timm_cls_mean_patch_features(output)


class Dinov3HubCropEncoder(nn.Module):
    """DINOv3 backbone wrapper returning CLS plus mean patch features."""

    def __init__(
        self,
        repo: str | Path,
        weights: str | Path,
        model_name: str = "dinov3_vitl16",
    ) -> None:
        super().__init__()
        if not str(weights):
            raise ValueError("DINOv3 hub weights must be provided with --dinov3-weights.")
        repo_value = Path(repo).expanduser() if not _looks_like_github_repo(str(repo)) else str(repo)
        source = "local" if isinstance(repo_value, Path) else "github"
        self.model = torch.hub.load(
            str(repo_value),
            model_name,
            source=source,
            weights=str(Path(weights).expanduser()) if not _looks_like_url(str(weights)) else str(weights),
        )
        self.embedding_size = _dinov3_embedding_size(model_name)
        self.output_size = self.embedding_size * 2

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        output = self.model.forward_features(images) if hasattr(self.model, "forward_features") else self.model(images)
        return dinov3_cls_mean_patch_features(output)


class SimpleDinoCropClassifier(L.LightningModule):
    """Frozen DINO crop classifier with a small trainable MLP head."""

    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int,
        num_classes: int = 5,
        learning_rate: float = 2e-4,
        weight_decay: float = 0.01,
        label_smoothing: float = 0.05,
        dropout: float = 0.2,
        hidden_dim: int = 256,
        freeze_backbone: bool = True,
        backbone_lr: float | None = None,
        lr_scheduler_monitor: str | None = "val_macro_f1",
        lr_scheduler_type: str = "plateau",
        plateau_factor: float = 0.5,
        plateau_patience: int = 125,
        plateau_threshold: float = 0.002,
        plateau_min_lr_factor: float = 0.05,
        step_lr_every_epochs: int = 100,
        step_lr_gamma: float = 0.5,
        orientation_aux: bool = False,
        orientation_aux_weight: float = 0.3,
        hard_sample_tracking: bool = False,
        classifier_head_layers: int = 2,
    ) -> None:
        super().__init__()
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive.")
        if num_classes <= 1:
            raise ValueError("num_classes must be greater than one.")
        if not 0 <= label_smoothing < 1:
            raise ValueError("label_smoothing must be in [0, 1).")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1).")
        if lr_scheduler_type not in {"none", "plateau", "step"}:
            raise ValueError("lr_scheduler_type must be one of: none, plateau, step.")
        if step_lr_every_epochs <= 0:
            raise ValueError("step_lr_every_epochs must be positive.")
        if not 0 < step_lr_gamma < 1:
            raise ValueError("step_lr_gamma must be in (0, 1).")
        if orientation_aux_weight < 0:
            raise ValueError("orientation_aux_weight must be non-negative.")
        if classifier_head_layers not in {2, 3}:
            raise ValueError("classifier_head_layers must be one of: 2, 3.")
        self.save_hyperparameters(ignore=["backbone"])
        self.backbone = backbone

        if classifier_head_layers == 2:
            self.classifier = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            self.classifier = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, num_classes),
            )
        if orientation_aux:
            self.orientation_aux_classifier = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 3),
            )
        if freeze_backbone:
            freeze_module(self.backbone)

        self._validation_outputs: list[dict[str, torch.Tensor]] = []
        self._logged_sample_image_splits: set[str] = set()
        self._hard_sample_observations: list[dict[str, Any]] = []

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if bool(self.hparams.freeze_backbone):
            self.backbone.eval()
            with torch.no_grad():
                features = self.backbone(images)
        else:
            features = self.backbone(images)
        features = features.float()
        return self.classifier(features), features

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        self._log_sample_images_once(batch, split="train", batch_idx=batch_idx)
        logits, features = self(batch["image"])
        labels = batch["label"].long()
        per_sample_loss = self._per_sample_training_loss(batch, logits, labels)
        sample_weights = batch.get("sample_loss_weight")
        if sample_weights is None:
            sample_weights = torch.ones_like(per_sample_loss)
        else:
            sample_weights = sample_weights.to(per_sample_loss.device, dtype=per_sample_loss.dtype)
        if "is_pseudo" in batch:
            pseudo_scale = float(getattr(self, "pseudo_loss_scale", 1.0))
            is_pseudo = batch["is_pseudo"].to(sample_weights.device, dtype=torch.bool)
            sample_weights = torch.where(is_pseudo, sample_weights * pseudo_scale, sample_weights)
        main_loss = weighted_mean(per_sample_loss, sample_weights)
        self._record_hard_sample_observations(batch, logits, labels)
        loss = self._loss_with_orientation_aux(main_loss, features, labels, split="train", sample_weights=sample_weights)
        self.log("train/loss", loss, on_epoch=True, prog_bar=True, batch_size=labels.size(0))
        self.log("train/supervised_loss", main_loss, on_epoch=True, batch_size=labels.size(0))
        self.log("train/acc", accuracy(logits, labels), on_epoch=True, prog_bar=True, batch_size=labels.size(0))
        self._log_bucket_training_accuracy(batch, logits, labels)
        self._log_pseudo_training_metrics(batch, logits, labels, per_sample_loss, sample_weights)
        self._log_bucket_class_exposure(batch, labels)
        return loss

    def _per_sample_training_loss(
        self,
        batch: dict[str, Any],
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        hard_loss = F.cross_entropy(
            logits,
            labels,
            label_smoothing=float(self.hparams.label_smoothing),
            reduction="none",
        )
        if "pseudo_soft_target" not in batch or "is_pseudo" not in batch:
            return hard_loss
        is_pseudo = batch["is_pseudo"].to(logits.device, dtype=torch.bool)
        has_soft_target = batch.get("pseudo_has_soft_target")
        if has_soft_target is None:
            has_soft_target_mask = is_pseudo
        else:
            has_soft_target_mask = has_soft_target.to(logits.device, dtype=torch.bool)
        soft_mask = is_pseudo & has_soft_target_mask
        if not soft_mask.any():
            return hard_loss
        soft_targets = batch["pseudo_soft_target"].to(logits.device, dtype=logits.dtype)
        soft_targets = soft_targets / soft_targets.sum(dim=1, keepdim=True).clamp_min(1e-8)
        soft_loss = -(soft_targets * F.log_softmax(logits, dim=1)).sum(dim=1)
        return torch.where(soft_mask, soft_loss, hard_loss)

    def pop_hard_sample_observations(self) -> list[dict[str, Any]]:
        observations = self._hard_sample_observations
        self._hard_sample_observations = []
        return observations

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        self._log_sample_images_once(batch, split="val", batch_idx=batch_idx)
        logits, features = self(batch["image"])
        labels = batch["label"].long()
        loss = F.cross_entropy(logits, labels)
        self._log_orientation_aux_metrics(features, labels, split="val")
        preds = logits.argmax(dim=1)
        val_acc = accuracy(logits, labels)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=labels.size(0))
        self.log("val/acc", val_acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=labels.size(0))
        self.log(
            KEY_TENSORBOARD_METRICS["val_acc"],
            val_acc,
            on_step=False,
            on_epoch=True,
            batch_size=labels.size(0),
        )
        self._validation_outputs.append({"preds": preds.detach().cpu(), "labels": labels.detach().cpu()})

    def on_validation_epoch_start(self) -> None:
        self._validation_outputs = []

    def on_validation_epoch_end(self) -> None:
        if not self._validation_outputs:
            return
        preds = torch.cat([item["preds"] for item in self._validation_outputs])
        labels = torch.cat([item["labels"] for item in self._validation_outputs])
        macro_f1 = macro_f1_from_predictions(preds, labels, int(self.hparams.num_classes))
        balanced_acc = balanced_accuracy_from_predictions(preds, labels, int(self.hparams.num_classes))
        self.log("val/macro_f1", macro_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_macro_f1", macro_f1, on_step=False, on_epoch=True)
        self.log("val/balanced_acc", balanced_acc, on_step=False, on_epoch=True)
        self.log(KEY_TENSORBOARD_METRICS["val_macro_f1"], macro_f1, on_step=False, on_epoch=True)
        self.log(KEY_TENSORBOARD_METRICS["val_balanced_acc"], balanced_acc, on_step=False, on_epoch=True)
        self._log_validation_per_class_metrics(preds, labels)
        self._validation_outputs = []

    def configure_optimizers(self):
        if self.hparams.backbone_lr is not None:
            optimizer = self._split_lr_optimizer()
            return self._optimizer_config(optimizer)
        decay_params = []
        no_decay_params = []
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if parameter.ndim == 1 or name.endswith(".bias"):
                no_decay_params.append(parameter)
            else:
                decay_params.append(parameter)
        groups: list[dict[str, Any]] = []
        if decay_params:
            groups.append({"params": decay_params, "weight_decay": float(self.hparams.weight_decay)})
        if no_decay_params:
            groups.append({"params": no_decay_params, "weight_decay": 0.0})
        optimizer = torch.optim.AdamW(groups, lr=float(self.hparams.learning_rate))
        return self._optimizer_config(optimizer)

    def _split_lr_optimizer(self) -> torch.optim.Optimizer:
        head_params = [parameter for parameter in self.classifier.parameters() if parameter.requires_grad]
        if bool(getattr(self.hparams, "orientation_aux", False)):
            head_params.extend(
                parameter for parameter in self.orientation_aux_classifier.parameters() if parameter.requires_grad
            )
        backbone_params = [parameter for parameter in self.backbone.parameters() if parameter.requires_grad]
        groups: list[dict[str, Any]] = []
        if head_params:
            groups.append(
                {
                    "params": head_params,
                    "lr": float(self.hparams.learning_rate),
                    "weight_decay": float(self.hparams.weight_decay),
                }
            )
        if backbone_params:
            groups.append(
                {
                    "params": backbone_params,
                    "lr": float(self.hparams.backbone_lr),
                    "weight_decay": float(self.hparams.weight_decay),
                }
            )
        if not groups:
            raise RuntimeError("No trainable parameters found for optimizer.")
        return torch.optim.AdamW(groups)

    def _optimizer_config(self, optimizer: torch.optim.Optimizer) -> dict[str, Any]:
        scheduler_type = str(self.hparams.lr_scheduler_type)
        if scheduler_type == "none":
            return {"optimizer": optimizer}
        if scheduler_type == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=int(self.hparams.step_lr_every_epochs),
                gamma=float(self.hparams.step_lr_gamma),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                },
            }
        if self.hparams.lr_scheduler_monitor is None:
            raise RuntimeError("Plateau scheduler requires lr_scheduler_monitor.")
        min_lrs = [float(group["lr"]) * float(self.hparams.plateau_min_lr_factor) for group in optimizer.param_groups]
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=float(self.hparams.plateau_factor),
            patience=int(self.hparams.plateau_patience),
            threshold=float(self.hparams.plateau_threshold),
            threshold_mode="abs",
            min_lr=min_lrs,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": str(self.hparams.lr_scheduler_monitor),
                "interval": "epoch",
            },
        }

    def _log_sample_images_once(self, batch: dict[str, Any], split: str, batch_idx: int, max_images: int = 8) -> None:
        if batch_idx != 0 or split in self._logged_sample_image_splits or "image" not in batch:
            return
        logger = getattr(self, "logger", None)
        experiment = getattr(logger, "experiment", None)
        add_images = getattr(experiment, "add_images", None)
        if add_images is None:
            return
        images = _denormalize_images(batch["image"][:max_images])
        add_images(f"samples/{split}_images", images, global_step=getattr(self, "global_step", 0))
        self._logged_sample_image_splits.add(split)

    def _loss_with_orientation_aux(
        self,
        main_loss: torch.Tensor,
        features: torch.Tensor,
        labels: torch.Tensor,
        split: str,
        sample_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not bool(getattr(self.hparams, "orientation_aux", False)):
            return main_loss
        aux_loss = self._log_orientation_aux_metrics(features, labels, split=split, sample_weights=sample_weights)
        return main_loss + float(self.hparams.orientation_aux_weight) * aux_loss

    def _log_orientation_aux_metrics(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        split: str,
        sample_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not bool(getattr(self.hparams, "orientation_aux", False)):
            return features.new_zeros(())
        aux_logits = self.orientation_aux_classifier(features)
        aux_labels = orientation_aux_labels(labels)
        aux_losses = F.cross_entropy(
            aux_logits,
            aux_labels,
            label_smoothing=float(self.hparams.label_smoothing),
            reduction="none",
        )
        aux_loss = weighted_mean(aux_losses, sample_weights)
        self.log(
            f"{split}/orientation_aux_loss",
            aux_loss,
            on_epoch=True,
            batch_size=labels.size(0),
        )
        self.log(
            f"{split}/orientation_aux_acc",
            accuracy(aux_logits, aux_labels),
            on_epoch=True,
            batch_size=labels.size(0),
        )
        return aux_loss

    def _log_pseudo_training_metrics(
        self,
        batch: dict[str, Any],
        logits: torch.Tensor,
        labels: torch.Tensor,
        per_sample_loss: torch.Tensor,
        sample_weights: torch.Tensor,
    ) -> None:
        if "is_pseudo" not in batch:
            return
        is_pseudo = batch["is_pseudo"].to(logits.device, dtype=torch.bool)
        pseudo_count = int(is_pseudo.sum().item())
        if pseudo_count == 0:
            self.log("train/pseudo_fraction", logits.new_zeros(()), on_epoch=True, batch_size=labels.size(0))
            return
        labeled_mask = ~is_pseudo
        pseudo_loss = weighted_mean(per_sample_loss[is_pseudo], sample_weights[is_pseudo])
        pseudo_acc = accuracy(logits[is_pseudo], labels[is_pseudo])
        pseudo_confidence = batch["pseudo_confidence"].to(logits.device, dtype=logits.dtype)[is_pseudo]
        pseudo_weights = sample_weights[is_pseudo]
        self.log("train/pseudo_rows", logits.new_tensor(float(pseudo_count)), on_epoch=True, batch_size=labels.size(0))
        self.log(
            "train/pseudo_fraction",
            logits.new_tensor(float(pseudo_count) / labels.size(0)),
            on_epoch=True,
            batch_size=labels.size(0),
        )
        self.log("train/pseudo_loss", pseudo_loss, on_epoch=True, batch_size=pseudo_count)
        self.log("train/pseudo_acc", pseudo_acc, on_epoch=True, batch_size=pseudo_count)
        self.log("train/pseudo_confidence_mean", pseudo_confidence.mean(), on_epoch=True, batch_size=pseudo_count)
        self.log("train/pseudo_loss_scale", logits.new_tensor(float(getattr(self, "pseudo_loss_scale", 1.0))), on_epoch=True, batch_size=pseudo_count)
        self.log(
            "train/effective_pseudo_weight_mean",
            pseudo_weights.mean(),
            on_epoch=True,
            batch_size=pseudo_count,
        )
        self.log("train/effective_pseudo_weight_min", pseudo_weights.min(), on_epoch=True, batch_size=pseudo_count)
        self.log("train/effective_pseudo_weight_max", pseudo_weights.max(), on_epoch=True, batch_size=pseudo_count)

    def _log_bucket_training_accuracy(
        self,
        batch: dict[str, Any],
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        bucket_masks = self._sample_bucket_masks(batch, logits.device)
        metric_specs = (
            ("source", "train/source_acc", KEY_TENSORBOARD_METRICS["train_source_acc"]),
            ("target", "train/target_acc", KEY_TENSORBOARD_METRICS["train_target_acc"]),
        )
        for bucket_name, metric_name, key_metric_name in metric_specs:
            mask = bucket_masks.get(bucket_name)
            if mask is None:
                continue
            count = int(mask.sum().item())
            if count == 0:
                continue
            bucket_acc = accuracy(logits[mask], labels[mask])
            self.log(metric_name, bucket_acc, on_epoch=True, batch_size=count)
            self.log(key_metric_name, bucket_acc, on_step=False, on_epoch=True, batch_size=count)

    def _log_validation_per_class_metrics(self, preds: torch.Tensor, labels: torch.Tensor) -> None:
        num_classes = int(self.hparams.num_classes)
        for class_id in range(num_classes):
            class_name = posture_class_metric_name(class_id)
            label_mask = labels.eq(class_id)
            pred_mask = preds.eq(class_id)
            true_positives = (label_mask & pred_mask).sum().to(dtype=torch.float32)
            predicted_count = pred_mask.sum().to(dtype=torch.float32)
            label_count = label_mask.sum().to(dtype=torch.float32)
            precision = safe_divide(true_positives, predicted_count)
            recall = safe_divide(true_positives, label_count)
            f1 = safe_divide(2 * precision * recall, precision + recall)
            prefix = f"val_class/{class_id:02d}_{class_name}"
            self.log(f"{prefix}_f1", f1, on_step=False, on_epoch=True)
            self.log(f"{prefix}_recall", recall, on_step=False, on_epoch=True)
            self.log(f"{prefix}_precision", precision, on_step=False, on_epoch=True)

    def _sample_bucket_masks(self, batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
        raw_buckets = batch.get("sample_bucket")
        if raw_buckets is None:
            raw_buckets = batch.get("domain")
        if raw_buckets is None:
            return {}
        if isinstance(raw_buckets, torch.Tensor):
            return {}
        buckets = [str(bucket) for bucket in raw_buckets]
        return {
            bucket_name: torch.tensor(
                [bucket == bucket_name for bucket in buckets],
                device=device,
                dtype=torch.bool,
            )
            for bucket_name in ("source", "target", "pseudo")
        }
        for key, metric_name in (
            ("pseudo_margin", "train/pseudo_margin_mean"),
            ("pseudo_entropy", "train/pseudo_entropy_mean"),
            ("pseudo_agreement", "train/pseudo_agreement_mean"),
            ("pseudo_quality_weight", "train/pseudo_quality_weight_mean"),
        ):
            if key in batch:
                values = batch[key].to(logits.device, dtype=logits.dtype)[is_pseudo]
                self.log(metric_name, values.mean(), on_epoch=True, batch_size=pseudo_count)
        if "pseudo_has_soft_target" in batch:
            soft_fraction = batch["pseudo_has_soft_target"].to(logits.device, dtype=logits.dtype)[is_pseudo].mean()
            self.log("train/pseudo_soft_target_fraction", soft_fraction, on_epoch=True, batch_size=pseudo_count)
        self._log_pseudo_confidence_buckets(pseudo_confidence)
        if labeled_mask.any():
            labeled_loss = weighted_mean(per_sample_loss[labeled_mask], sample_weights[labeled_mask])
            self.log("train/labeled_loss", labeled_loss, on_epoch=True, batch_size=int(labeled_mask.sum().item()))

    def _log_pseudo_confidence_buckets(self, pseudo_confidence: torch.Tensor) -> None:
        buckets = (
            ("085_090", pseudo_confidence.lt(0.90)),
            ("090_095", pseudo_confidence.ge(0.90) & pseudo_confidence.lt(0.95)),
            ("095_plus", pseudo_confidence.ge(0.95)),
        )
        for name, mask in buckets:
            self.log(
                f"train/pseudo_conf_bucket_{name}",
                mask.to(dtype=pseudo_confidence.dtype).mean(),
                on_epoch=True,
                batch_size=int(pseudo_confidence.numel()),
            )

    def _log_bucket_class_exposure(self, batch: dict[str, Any], labels: torch.Tensor) -> None:
        buckets = batch.get("sample_bucket")
        if buckets is None:
            return
        labels_cpu = labels.detach().cpu()
        for bucket in ("source", "target", "pseudo"):
            bucket_mask = torch.tensor([str(value) == bucket for value in buckets], dtype=torch.bool)
            bucket_count = int(bucket_mask.sum().item())
            self.log(
                f"sampler/{bucket}_fraction",
                labels.new_tensor(float(bucket_count) / max(1, labels.numel()), dtype=torch.float32),
                on_epoch=True,
                batch_size=labels.size(0),
            )
            if bucket_count == 0:
                continue
            bucket_labels = labels_cpu[bucket_mask]
            for class_id in range(int(self.hparams.num_classes)):
                class_fraction = bucket_labels.eq(class_id).float().mean()
                self.log(
                    f"sampler/{bucket}_class_{class_id}",
                    class_fraction.to(labels.device),
                    on_epoch=True,
                    batch_size=bucket_count,
                )

    def _record_hard_sample_observations(
        self,
        batch: dict[str, Any],
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        if not bool(getattr(self.hparams, "hard_sample_tracking", False)):
            return
        row_ids = batch.get("row_id")
        domains = batch.get("domain")
        if row_ids is None or domains is None:
            return
        is_pseudo = batch.get("is_pseudo")
        scores = hard_sample_scores(logits.detach(), labels.detach())
        for index, row_id in enumerate(row_ids):
            if str(domains[index]) != "target":
                continue
            if is_pseudo is not None and bool(is_pseudo[index]):
                continue
            self._hard_sample_observations.append(
                {
                    "row_id": str(row_id),
                    "hardness": float(scores["hardness"][index].item()),
                    "true_probability": float(scores["true_probability"][index].item()),
                    "margin": float(scores["margin"][index].item()),
                    "pred_class_id": int(scores["pred_class_id"][index].item()),
                    "correct": bool(scores["correct"][index].item()),
                }
            )


def orientation_aux_labels(labels: torch.Tensor) -> torch.Tensor:
    """Map posture labels to other/sternal/lateral auxiliary orientation labels."""

    aux_labels = torch.zeros_like(labels, dtype=torch.long)
    aux_labels[labels == 4] = 1
    aux_labels[(labels == 0) | (labels == 1)] = 2
    return aux_labels


def weighted_mean(losses: torch.Tensor, weights: torch.Tensor | None) -> torch.Tensor:
    if weights is None:
        return losses.mean()
    weights = weights.to(device=losses.device, dtype=losses.dtype)
    return (losses * weights).sum() / weights.sum().clamp_min(1e-8)


def safe_divide(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    return torch.where(denominator > 0, numerator / denominator.clamp_min(1e-8), numerator.new_zeros(()))


def posture_class_metric_name(class_id: int) -> str:
    if 0 <= class_id < len(POSTURE_CLASS_NAMES):
        return POSTURE_CLASS_NAMES[class_id]
    return f"class_{class_id}"


def hard_sample_scores(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return detached per-sample hardness signals for dynamic hard sampling."""

    probabilities = logits.float().softmax(dim=1)
    labels = labels.long()
    true_probability = probabilities.gather(1, labels.view(-1, 1)).squeeze(1)
    other_probabilities = probabilities.clone()
    other_probabilities.scatter_(1, labels.view(-1, 1), -1.0)
    other_probability = other_probabilities.max(dim=1).values
    pred_class_id = probabilities.argmax(dim=1)
    margin = true_probability - other_probability
    ce = -true_probability.clamp_min(1e-8).log()
    boundary_penalty = ((0.20 - margin) / 0.20).clamp(min=0.0, max=2.0)
    hardness = ce + (0.5 * boundary_penalty)
    return {
        "hardness": hardness.detach(),
        "true_probability": true_probability.detach(),
        "margin": margin.detach(),
        "pred_class_id": pred_class_id.detach(),
        "correct": pred_class_id.eq(labels).detach(),
    }


def dinov3_cls_mean_patch_features(output: Any) -> torch.Tensor:
    """Return concatenated CLS and mean patch features from DINO-style output."""

    if isinstance(output, dict):
        if "x_norm_clstoken" not in output or "x_norm_patchtokens" not in output:
            raise RuntimeError(f"Unsupported DINO feature keys: {sorted(output)}")
        cls_features = output["x_norm_clstoken"]
        patch_features = output["x_norm_patchtokens"].mean(dim=1)
        return torch.cat([cls_features, patch_features], dim=1)
    if isinstance(output, torch.Tensor):
        return output
    raise RuntimeError(f"Unsupported DINO output type: {type(output).__name__}")


def timm_cls_mean_patch_features(output: Any) -> torch.Tensor:
    """Return concatenated CLS and mean patch features from timm ViT output."""

    if isinstance(output, dict):
        output = output.get("x", output.get("features"))
        if output is None:
            raise RuntimeError("Unsupported timm feature dict; expected key 'x' or 'features'.")
    if not isinstance(output, torch.Tensor):
        raise RuntimeError(f"Unsupported timm feature output type: {type(output).__name__}")
    if output.ndim == 3:
        cls_features = output[:, 0]
        patch_features = output[:, 1:].mean(dim=1)
        return torch.cat([cls_features, patch_features], dim=1)
    if output.ndim == 2:
        return torch.cat([output, output], dim=1)
    raise RuntimeError(f"Unsupported timm feature tensor shape: {tuple(output.shape)}")


def freeze_module(module: nn.Module) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad = False


def resolve_lora_linear_target_modules(model: nn.Module, target_modules: tuple[str, ...]) -> tuple[str, ...]:
    requested = set(target_modules)
    matches = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if name.rsplit(".", 1)[-1] in requested:
            matches.append(name)
    if not matches:
        raise ValueError(f"No linear LoRA target modules matched: {', '.join(target_modules)}")
    return tuple(matches)


def _denormalize_images(images: torch.Tensor) -> torch.Tensor:
    images = images.detach().float().cpu()
    mean = images.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (images * std + mean).clamp(0.0, 1.0)


def normalize_timm_model_name(model_name: str) -> str:
    """Map HF timm collection names to installed timm registry names when needed."""

    if model_name.startswith("hf-hub:"):
        return model_name
    available = set(timm.list_models())
    if model_name in available:
        return model_name
    if model_name.endswith(".lvd1689m"):
        base_name = model_name.removesuffix(".lvd1689m")
        if base_name in available:
            return base_name
    return model_name


def _dinov3_embedding_size(model_name: str) -> int:
    if "vitl16" in model_name:
        return 1024
    if "vitb16" in model_name:
        return 768
    if "vits16" in model_name:
        return 384
    if "vith16" in model_name or "vit7b16" in model_name:
        return 1280
    raise ValueError(f"Unknown DINOv3 ViT embedding size for model_name={model_name!r}; pass a supported ViT model.")


def _looks_like_github_repo(value: str) -> bool:
    return "/" in value and not Path(value).expanduser().exists() and not value.startswith((".", "~", "/"))


def _looks_like_url(value: str) -> bool:
    return value.startswith(("http://", "https://"))
