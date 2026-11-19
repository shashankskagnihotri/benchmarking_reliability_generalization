"""Spatial transforms matching the SemSegBench test pipelines.

ADE20K uses ``Resize, scale=(2048, 512), keep_ratio=True``.
Cityscapes and ACDC use ``Resize, scale=(2048, 1024), keep_ratio=True``.
The scale rule is mmcv's keep-ratio rule: the image is scaled by
``min(max(scale) / long_edge, min(scale) / short_edge)``.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

BENCHMARK_SCALE = {
    "ade20k": (2048, 512),
    "cityscapes": (2048, 1024),
    "acdc": (2048, 1024),
}


def keep_ratio_size(width: int, height: int, scale: tuple[int, int]) -> tuple[int, int]:
    max_long = max(scale)
    max_short = min(scale)
    factor = min(max_long / max(width, height), max_short / min(width, height))
    return int(width * float(factor) + 0.5), int(height * float(factor) + 0.5)


def resize_image(image: np.ndarray, dataset: str) -> np.ndarray:
    """Resize an HWC uint8 image with bilinear interpolation."""
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected HWC uint8 RGB, got shape {image.shape} dtype {image.dtype}")
    height, width = image.shape[:2]
    new_w, new_h = keep_ratio_size(width, height, BENCHMARK_SCALE[dataset])
    if (new_w, new_h) == (width, height):
        return image
    resized = Image.fromarray(image).resize((new_w, new_h), Image.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)
