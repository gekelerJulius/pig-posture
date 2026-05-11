"""Transforms for cropped RGB pig posture images."""

from __future__ import annotations

from dataclasses import dataclass
import random

import torch
from torchvision import transforms
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class TtaTransform:
    transform: transforms.Compose
    horizontally_flipped: bool = False

    def __call__(self, image):
        return self.transform(image)


@dataclass(frozen=True)
class RandomGamma:
    gamma_range: tuple[float, float] = (0.90, 1.12)

    def __call__(self, image):
        gamma = random.uniform(*self.gamma_range)
        return F.adjust_gamma(image, gamma=gamma)


@dataclass(frozen=True)
class RandomGaussianNoise:
    sigma_range: tuple[float, float] = (0.002, 0.015)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        sigma = random.uniform(*self.sigma_range)
        return (tensor + torch.randn_like(tensor) * sigma).clamp(0.0, 1.0)


@dataclass(frozen=True)
class RandomOneOf:
    transforms_: tuple[object, ...]

    def __call__(self, image):
        if not self.transforms_:
            return image
        transform = random.choice(self.transforms_)
        return transform(image)


def build_train_transforms(image_size: int = 224) -> transforms.Compose:
    """Build moderate training transforms for low-data domain adaptation."""

    return transforms.Compose(
        [
            transforms.Resize(
                (image_size + 40, image_size + 40),
                interpolation=InterpolationMode.BILINEAR,
            ),
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.90, 1.0),
                ratio=(0.94, 1.06),
                interpolation=InterpolationMode.BILINEAR,
            ),
            transforms.RandomApply(
                [
                    transforms.RandomAffine(
                        degrees=8,
                        translate=(0.035, 0.035),
                        scale=(0.94, 1.06),
                        shear=(-2, 2),
                        interpolation=InterpolationMode.BILINEAR,
                        fill=0,
                    )
                ],
                p=0.40,
            ),
            transforms.RandomApply(
                [
                    transforms.RandomPerspective(
                        distortion_scale=0.08,
                        p=1.0,
                        interpolation=InterpolationMode.BILINEAR,
                        fill=0,
                    )
                ],
                p=0.02,
            ),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.20,
                        contrast=0.20,
                        saturation=0.10,
                        hue=0.015,
                    )
                ],
                p=0.45,
            ),
            transforms.RandomApply([RandomGamma()], p=0.15),
            transforms.RandomApply(
                [
                    RandomOneOf(
                        (
                            transforms.RandomResizedCrop(
                                image_size,
                                scale=(0.78, 0.96),
                                ratio=(0.88, 1.14),
                                interpolation=InterpolationMode.BILINEAR,
                            ),
                            transforms.RandomAffine(
                                degrees=14,
                                translate=(0.06, 0.06),
                                scale=(0.88, 1.12),
                                shear=(-4, 4),
                                interpolation=InterpolationMode.BILINEAR,
                                fill=0,
                            ),
                            transforms.ColorJitter(
                                brightness=0.35,
                                contrast=0.35,
                                saturation=0.20,
                                hue=0.025,
                            ),
                            transforms.RandomPerspective(
                                distortion_scale=0.08,
                                p=1.0,
                                interpolation=InterpolationMode.BILINEAR,
                                fill=0,
                            ),
                        )
                    )
                ],
                p=0.08,
            ),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8))],
                p=0.05,
            ),
            transforms.ToTensor(),
            transforms.RandomApply([RandomGaussianNoise()], p=0.06),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.RandomErasing(
                p=0.04,
                scale=(0.005, 0.035),
                ratio=(0.4, 2.5),
                value="random",
            ),
        ]
    )


def build_target_orb_train_transforms(image_size: int = 224) -> transforms.Compose:
    """Build stronger Orbbec-target transforms for the weakest target camera subgroup."""

    return transforms.Compose(
        [
            transforms.Resize(
                (image_size + 32, image_size + 32),
                interpolation=InterpolationMode.BILINEAR,
            ),
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.88, 1.0),
                ratio=(0.92, 1.10),
                interpolation=InterpolationMode.BILINEAR,
            ),
            transforms.RandomApply(
                [
                    transforms.RandomAffine(
                        degrees=7,
                        translate=(0.04, 0.04),
                        scale=(0.93, 1.07),
                        shear=(-3, 3),
                        interpolation=InterpolationMode.BILINEAR,
                        fill=0,
                    )
                ],
                p=0.35,
            ),
            transforms.RandomApply(
                [
                    transforms.RandomPerspective(
                        distortion_scale=0.08,
                        p=1.0,
                        interpolation=InterpolationMode.BILINEAR,
                        fill=0,
                    )
                ],
                p=0.03,
            ),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.24,
                        contrast=0.24,
                        saturation=0.18,
                        hue=0.025,
                    )
                ],
                p=0.40,
            ),
            transforms.RandomApply([RandomGamma((0.88, 1.16))], p=0.18),
            transforms.RandomAutocontrast(p=0.08),
            transforms.RandomEqualize(p=0.04),
            transforms.RandomApply(
                [
                    RandomOneOf(
                        (
                            transforms.RandomResizedCrop(
                                image_size,
                                scale=(0.76, 0.96),
                                ratio=(0.86, 1.16),
                                interpolation=InterpolationMode.BILINEAR,
                            ),
                            transforms.RandomAffine(
                                degrees=15,
                                translate=(0.07, 0.07),
                                scale=(0.87, 1.13),
                                shear=(-5, 5),
                                interpolation=InterpolationMode.BILINEAR,
                                fill=0,
                            ),
                            transforms.ColorJitter(
                                brightness=0.38,
                                contrast=0.38,
                                saturation=0.22,
                                hue=0.03,
                            ),
                            transforms.RandomPerspective(
                                distortion_scale=0.10,
                                p=1.0,
                                interpolation=InterpolationMode.BILINEAR,
                                fill=0,
                            ),
                        )
                    )
                ],
                p=0.10,
            ),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))],
                p=0.08,
            ),
            transforms.ToTensor(),
            transforms.RandomApply([RandomGaussianNoise((0.003, 0.018))], p=0.10),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.RandomErasing(
                p=0.06,
                scale=(0.01, 0.06),
                ratio=(0.3, 3.3),
                value="random",
            ),
        ]
    )


def build_eval_transforms(image_size: int = 224) -> transforms.Compose:
    """Build deterministic validation/test transforms."""

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_tta_transforms(image_size: int = 224, views: int = 4) -> list[TtaTransform]:
    """Build deterministic classification TTA transforms."""

    if views not in {1, 2, 4}:
        raise ValueError("views must be one of: 1, 2, 4.")

    base = [
        TtaTransform(
            transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ]
            ),
            horizontally_flipped=False,
        )
    ]
    if views >= 2:
        base.append(
            TtaTransform(
                transforms.Compose(
                    [
                        transforms.RandomHorizontalFlip(p=1.0),
                        transforms.Resize((image_size, image_size)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                    ]
                ),
                horizontally_flipped=True,
            )
        )
    if views >= 4:
        zoom_size = image_size + max(16, image_size // 8)
        base.extend(
            [
                TtaTransform(
                    transforms.Compose(
                        [
                            transforms.Resize((zoom_size, zoom_size)),
                            transforms.CenterCrop(image_size),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                        ]
                    ),
                    horizontally_flipped=False,
                ),
                TtaTransform(
                    transforms.Compose(
                        [
                            transforms.RandomHorizontalFlip(p=1.0),
                            transforms.Resize((zoom_size, zoom_size)),
                            transforms.CenterCrop(image_size),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                        ]
                    ),
                    horizontally_flipped=True,
                ),
            ]
        )
    return base
