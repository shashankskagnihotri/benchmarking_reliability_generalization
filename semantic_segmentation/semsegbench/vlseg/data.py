"""Validation-split readers for ADE20K, Cityscapes, and ACDC.

ADE20K annotations use ``reduce_zero_label`` exactly as
``mmseg.datasets.transforms.loading.LoadAnnotations``: label 0 becomes ignore
255, and the remaining ids shift down by one. Cityscapes and ACDC already
store trainIds, with 255 ignored. ACDC uses the official validation images,
which are the annotated evaluation split. The mmseg config directory name
``test`` is not the official unlabeled test split.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .geometry import resize_image

ADE_ROOT = Path(os.environ.get("VLSEG_ADE_ROOT", "/ceph/sagnihot/datasets/ade/ADEChallengeData2016"))
CITYSCAPES_ROOT = Path(os.environ.get("VLSEG_CITYSCAPES_ROOT", "/ceph/sagnihot/datasets/cityscapes"))
ACDC_ROOT = Path(os.environ.get("VLSEG_ACDC_ROOT", "/ceph/sagnihot/datasets/ACDC"))

EXPECTED_COUNTS = {
    "ade20k": 2000,
    "cityscapes": 500,
    "acdc_fog": 100,
    "acdc_night": 106,
    "acdc_rain": 100,
    "acdc_snow": 100,
    "acdc": 406,
}
IGNORE_INDEX = 255


@dataclass(frozen=True)
class Sample:
    image_id: str
    image_path: str
    label_path: str
    dataset: str
    condition: str


def _walk_files(root: Path, suffix: str) -> dict[str, Path]:
    found = {}
    for path in root.rglob(f"*{suffix}"):
        if not path.is_file():
            continue
        key = path.name[: -len(suffix)]
        if key in found:
            raise RuntimeError(f"duplicate id {key} under {root}")
        found[key] = path
    return found


def reduce_zero_label(label: np.ndarray) -> np.ndarray:
    label = label.astype(np.int64, copy=True)
    label[label == 0] = 255
    label = label - 1
    label[label == 254] = 255
    return label


def list_samples(dataset: str, condition: str = "") -> list[Sample]:
    if dataset == "ade20k":
        images = _walk_files(ADE_ROOT / "images" / "validation", ".jpg")
        labels = _walk_files(ADE_ROOT / "annotations" / "validation", ".png")
        condition = ""
    elif dataset == "cityscapes":
        images = _walk_files(CITYSCAPES_ROOT / "leftImg8bit" / "val", "_leftImg8bit.png")
        labels = _walk_files(CITYSCAPES_ROOT / "gtFine" / "val", "_gtFine_labelTrainIds.png")
        condition = ""
    elif dataset == "acdc":
        conditions = (condition,) if condition else ("fog", "night", "rain", "snow")
        images = {}
        labels = {}
        for name in conditions:
            image_part = _walk_files(ACDC_ROOT / "rgb_anon" / name / "val", "_rgb_anon.png")
            label_part = _walk_files(ACDC_ROOT / "gt" / name / "val", "_gt_labelTrainIds.png")
            overlap = set(image_part) & set(images)
            if overlap:
                raise RuntimeError(f"ACDC image id repeats across conditions: {sorted(overlap)[:5]}")
            images.update(image_part)
            labels.update(label_part)
    else:
        raise ValueError(dataset)
    missing = sorted(set(images) - set(labels))
    extra = sorted(set(labels) - set(images))
    if missing or extra:
        raise RuntimeError(f"unpaired {dataset} {condition}: missing {missing[:5]} extra {extra[:5]}")
    samples = [
        Sample(image_id, str(images[image_id]), str(labels[image_id]), dataset, condition if dataset == "acdc" and condition else _acdc_condition(images[image_id]) if dataset == "acdc" else "")
        for image_id in sorted(images)
    ]
    key = dataset if not condition else f"acdc_{condition}"
    expected = EXPECTED_COUNTS[key if dataset == "acdc" else dataset]
    if len(samples) != expected:
        raise RuntimeError(f"{key} has {len(samples)} pairs, expected {expected}")
    ids = [sample.image_id for sample in samples]
    paths = [sample.image_path for sample in samples]
    if len(set(ids)) != len(ids) or len(set(paths)) != len(paths):
        raise RuntimeError(f"duplicate ids or paths in {key}")
    if dataset == "acdc" and not condition:
        seen = {sample.condition for sample in samples}
        if seen != {"fog", "night", "rain", "snow"}:
            raise RuntimeError(f"aggregate conditions are {seen}")
        for name, count in (("fog", 100), ("night", 106), ("rain", 100), ("snow", 100)):
            got = sum(sample.condition == name for sample in samples)
            if got != count:
                raise RuntimeError(f"aggregate {name} count {got}")
    return samples


def _acdc_condition(path: str) -> str:
    parts = Path(path).parts
    for name in ("fog", "night", "rain", "snow"):
        if name in parts:
            return name
    raise RuntimeError(f"ACDC path has no condition: {path}")


def load_sample(sample: Sample) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return original RGB uint8, benchmark-resized RGB uint8, and label ids."""
    image = np.asarray(Image.open(sample.image_path).convert("RGB"), dtype=np.uint8)
    label = np.asarray(Image.open(sample.label_path), dtype=np.uint8)
    if label.ndim != 2:
        raise RuntimeError(f"label {sample.label_path} has shape {label.shape}")
    if sample.dataset == "ade20k":
        label = reduce_zero_label(label)
    else:
        label = label.astype(np.int64)
    if image.shape[:2] != label.shape:
        raise RuntimeError(f"image/label size mismatch for {sample.image_id}")
    resized = resize_image(image, sample.dataset)
    return image, resized, label
