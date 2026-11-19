"""Severity-3 corruptions in the same place as SegDataPreProcessor.

``SegDataPreProcessor.forward`` calls ``imagecorruptions.corrupt`` on the
loaded array before the BGR-to-RGB channel swap and before mean/std
normalization. OpenCV loading in that pipeline is BGR. This helper applies
the same order to an RGB uint8 image: RGB to BGR, corrupt, BGR to RGB.

The corruption is not applied to labels. Each image uses a deterministic
NumPy seed derived from its id, so a resumed run reproduces the same
corruption instead of depending on how many images were already processed.
"""

from __future__ import annotations

import hashlib

import numpy as np

CORRUPTIONS = (
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
)
SEVERITY = 3


def corruption_seed(image_id: str) -> int:
    digest = hashlib.sha256(f"corruption:{image_id}".encode()).digest()
    return int.from_bytes(digest[:4], "little")


def apply_corruption(image_rgb: np.ndarray, name: str, image_id: str, severity: int = SEVERITY) -> np.ndarray:
    if name not in CORRUPTIONS:
        raise ValueError(f"unknown corruption {name}")
    if severity != SEVERITY:
        raise ValueError(f"this extension evaluates severity {SEVERITY}, got {severity}")
    if image_rgb.dtype != np.uint8 or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("corruption expects HWC uint8 RGB before normalization")
    from imagecorruptions import corrupt

    state = np.random.get_state()
    np.random.seed(corruption_seed(image_id))
    try:
        bgr = image_rgb[:, :, ::-1]
        corrupted = corrupt(bgr, corruption_name=name, severity=severity)
    finally:
        np.random.set_state(state)
    corrupted = np.asarray(corrupted)
    if corrupted.dtype != np.uint8:
        corrupted = np.clip(corrupted, 0, 255).astype(np.uint8)
    if corrupted.ndim != 3 or corrupted.shape[2] != 3:
        raise ValueError(f"corrupt returned shape {corrupted.shape}")
    return corrupted[:, :, ::-1].copy()
