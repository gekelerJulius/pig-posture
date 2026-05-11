from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pigpose.data.single_bbox_dataset import (
    BLACK_RGB,
    IMAGENET_NEUTRAL_RGB,
    SingleBboxDataset,
    build_mask_paths_by_row_id,
    resolve_mask_background_mode,
)


def test_masked_crop_blends_background_toward_neutral(tmp_path: Path) -> None:
    image_path = tmp_path / "train2_images" / "image_a.jpg"
    mask_path = tmp_path / "train2_masks" / "image_a_mask_01.png"
    write_rgb_image(image_path, color=(200, 100, 20), size=(6, 6))
    write_mask(mask_path, true_region=(slice(2, 4), slice(2, 4)), size=(6, 6))
    row = make_row(image_path=image_path, instance_index="0000", bbox=[1.0, 1.0, 4.0, 4.0])

    item = SingleBboxDataset(
        [row],
        mask_dir_root=tmp_path,
        mask_background_attenuation=0.75,
        mask_background_mode="attenuate",
    )[0]

    pixels = np.asarray(item["image"])
    neutral = np.asarray(IMAGENET_NEUTRAL_RGB)
    expected_background = np.rint((0.25 * np.asarray([200, 100, 20])) + (0.75 * neutral)).astype(np.uint8)
    assert pixels[0, 0].tolist() == expected_background.tolist()
    assert pixels[1, 1].tolist() == [200, 100, 20]


def test_mask_lookup_uses_original_instance_index_not_subset_order(tmp_path: Path) -> None:
    image_path = tmp_path / "train2_images" / "image_a.jpg"
    row = make_row(image_path=image_path, instance_index="0002", bbox=[0.0, 0.0, 4.0, 4.0])

    paths = build_mask_paths_by_row_id([row], mask_dir=None, mask_dir_root=tmp_path)

    assert paths[row["row_id"]].name == "image_a_mask_03.png"


def test_masked_crop_missing_mask_fails_with_path(tmp_path: Path) -> None:
    image_path = tmp_path / "train2_images" / "image_a.jpg"
    write_rgb_image(image_path, color=(200, 100, 20), size=(6, 6))
    row = make_row(image_path=image_path, instance_index="0000", bbox=[1.0, 1.0, 4.0, 4.0])
    dataset = SingleBboxDataset(
        [row],
        mask_dir_root=tmp_path,
        mask_background_attenuation=0.75,
        mask_background_mode="attenuate",
    )

    with pytest.raises(FileNotFoundError, match="image_a_mask_01.png"):
        dataset[0]


def test_masked_crop_keeps_masks_in_cpu_pil_cache(tmp_path: Path) -> None:
    image_path = tmp_path / "train2_images" / "image_a.jpg"
    mask_path = tmp_path / "train2_masks" / "image_a_mask_01.png"
    write_rgb_image(image_path, color=(200, 100, 20), size=(6, 6))
    write_mask(mask_path, true_region=(slice(2, 4), slice(2, 4)), size=(6, 6))
    row = make_row(image_path=image_path, instance_index="0000", bbox=[1.0, 1.0, 4.0, 4.0])
    dataset = SingleBboxDataset(
        [row],
        mask_dir_root=tmp_path,
        mask_background_attenuation=0.75,
        mask_background_mode="attenuate",
        mask_cache_size=2,
    )

    dataset[0]
    dataset[0]

    assert list(dataset._mask_cache) == [mask_path]
    cached_mask = dataset._mask_cache[mask_path]
    assert isinstance(cached_mask, Image.Image)
    assert cached_mask.mode == "L"


def test_masked_crop_is_flipped_after_attenuation(tmp_path: Path) -> None:
    image_path = tmp_path / "train2_images" / "image_a.jpg"
    mask_path = tmp_path / "train2_masks" / "image_a_mask_01.png"
    image_path.parent.mkdir(parents=True)
    pixels = np.zeros((4, 4, 3), dtype=np.uint8)
    pixels[:, :2] = [200, 100, 20]
    pixels[:, 2:] = [20, 100, 200]
    Image.fromarray(pixels, mode="RGB").save(image_path, format="PNG")
    write_mask(mask_path, true_region=(slice(0, 4), slice(0, 2)), size=(4, 4))
    row = make_row(image_path=image_path, instance_index="0000", bbox=[0.0, 0.0, 4.0, 4.0], class_id=0)

    item = SingleBboxDataset(
        [row],
        horizontal_flip_probability=1.0,
        mask_dir_root=tmp_path,
        mask_background_attenuation=0.75,
        mask_background_mode="attenuate",
    )[0]

    flipped = np.asarray(item["image"])
    assert flipped[0, 2].tolist() == [200, 100, 20]
    assert item["label"] == 1


def test_masked_crop_hard_mode_replaces_background_with_neutral(tmp_path: Path) -> None:
    image_path = tmp_path / "train2_images" / "image_a.jpg"
    mask_path = tmp_path / "train2_masks" / "image_a_mask_01.png"
    write_rgb_image(image_path, color=(200, 100, 20), size=(6, 6))
    write_mask(mask_path, true_region=(slice(2, 4), slice(2, 4)), size=(6, 6))
    row = make_row(image_path=image_path, instance_index="0000", bbox=[1.0, 1.0, 4.0, 4.0])

    item = SingleBboxDataset(
        [row],
        mask_dir_root=tmp_path,
        mask_background_mode="hard",
    )[0]

    pixels = np.asarray(item["image"])
    assert pixels[0, 0].tolist() == list(BLACK_RGB)
    assert pixels[1, 1].tolist() == [200, 100, 20]


def test_random_mask_mode_samples_context_preserving_modes(monkeypatch: pytest.MonkeyPatch) -> None:
    sampled_modes = []

    def fake_choice(modes):
        sampled_modes.extend(modes)
        return modes[0]

    monkeypatch.setattr("pigpose.data.single_bbox_dataset.random.choice", fake_choice)

    assert resolve_mask_background_mode("random") == "none"
    assert sampled_modes == ["none", "attenuate", "soft"]


def test_random_attenuate_mask_mode_is_half_original_half_attenuated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("pigpose.data.single_bbox_dataset.random.random", lambda: 0.49)
    assert resolve_mask_background_mode("random_attenuate") == "attenuate"

    monkeypatch.setattr("pigpose.data.single_bbox_dataset.random.random", lambda: 0.50)
    assert resolve_mask_background_mode("random_attenuate") == "none"


def test_foreground_view_uses_same_crop_and_hard_mask(tmp_path: Path) -> None:
    image_path = tmp_path / "train2_images" / "image_a.jpg"
    mask_path = tmp_path / "train2_masks" / "image_a_mask_01.png"
    write_rgb_image(image_path, color=(200, 100, 20), size=(6, 6))
    write_mask(mask_path, true_region=(slice(2, 4), slice(2, 4)), size=(6, 6))
    row = make_row(image_path=image_path, instance_index="0000", bbox=[1.0, 1.0, 4.0, 4.0])

    item = SingleBboxDataset(
        [row],
        mask_dir_root=tmp_path,
        mask_background_mode="none",
        include_foreground_view=True,
        foreground_mask_mode="hard",
    )[0]

    image = np.asarray(item["image"])
    foreground = np.asarray(item["foreground_image"])
    assert image.shape == foreground.shape
    assert image[0, 0].tolist() == [200, 100, 20]
    assert foreground[0, 0].tolist() == list(BLACK_RGB)
    assert foreground[1, 1].tolist() == [200, 100, 20]


def test_mask_geometry_metadata_is_appended(tmp_path: Path) -> None:
    image_path = tmp_path / "train2_images" / "image_a.jpg"
    mask_path = tmp_path / "train2_masks" / "image_a_mask_01.png"
    write_rgb_image(image_path, color=(200, 100, 20), size=(6, 6))
    write_mask(mask_path, true_region=(slice(2, 4), slice(2, 4)), size=(6, 6))
    row = make_row(image_path=image_path, instance_index="0000", bbox=[1.0, 1.0, 4.0, 4.0])

    item = SingleBboxDataset(
        [row],
        mask_dir_root=tmp_path,
        metadata_vocab={"camera_id": {"unknown": 0}},
    )[0]

    metadata = item["metadata_continuous"].tolist()
    assert len(metadata) == 13
    assert metadata[6] == pytest.approx(4.0 / 36.0)
    assert metadata[8] == pytest.approx(1.0)


def write_rgb_image(path: Path, color: tuple[int, int, int], size: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=color).save(path, format="PNG")


def write_mask(path: Path, true_region: tuple[slice, slice], size: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.zeros(size, dtype=np.uint8)
    mask[true_region] = 255
    Image.fromarray(mask, mode="L").save(path)


def make_row(
    *,
    image_path: Path,
    instance_index: str,
    bbox: list[float],
    class_id: int = 3,
) -> dict:
    return {
        "split": "train2",
        "row_id": f"train_image_a_{instance_index}",
        "instance_index": instance_index,
        "image_id": image_path.name,
        "image_path": image_path,
        "width": 6,
        "height": 6,
        "bbox": bbox,
        "class_id": class_id,
    }
