"""IoU accumulation matching mmseg.evaluation.metrics.iou_metric.IoUMetric.

Reported aAcc, mIoU, and mAcc are percentages. mIoU and mAcc are the mean
over classes whose union or ground-truth area is nonzero, then rounded to
two decimals, which is what IoUMetric writes.
"""

from __future__ import annotations

import numpy as np


def intersect_and_union(pred: np.ndarray, label: np.ndarray, num_classes: int, ignore_index: int = 255):
    """Return per-class intersection, union, prediction area, and label area."""
    pred = np.asarray(pred)
    label = np.asarray(label)
    if pred.shape != label.shape:
        raise ValueError(f"prediction shape {pred.shape} does not match label shape {label.shape}")
    mask = label != ignore_index
    pred = pred[mask].astype(np.int64, copy=False)
    label = label[mask].astype(np.int64, copy=False)
    valid = (pred >= 0) & (pred < num_classes)
    pred = pred[valid]
    label = label[valid]
    intersect = pred[pred == label]
    area_intersect = np.bincount(intersect, minlength=num_classes).astype(np.float64)
    area_pred = np.bincount(pred, minlength=num_classes).astype(np.float64)
    area_label = np.bincount(label, minlength=num_classes).astype(np.float64)
    area_union = area_pred + area_label - area_intersect
    return area_intersect, area_union, area_pred, area_label


def empty_counts(num_classes: int):
    zero = np.zeros(num_classes, dtype=np.float64)
    return zero.copy(), zero.copy(), zero.copy(), zero.copy()


def add_counts(total, part):
    return tuple(a + b for a, b in zip(total, part))


def metrics_from_counts(area_intersect, area_union, area_pred, area_label):
    """Return unrounded proportions and the IoUMetric percentage summary."""
    del area_pred
    label_sum = float(area_label.sum())
    if label_sum <= 0:
        raise ValueError("no valid ground-truth pixels")
    aacc = float(area_intersect.sum() / label_sum)
    iou = np.divide(area_intersect, area_union, out=np.full_like(area_intersect, np.nan), where=area_union > 0)
    acc = np.divide(area_intersect, area_label, out=np.full_like(area_intersect, np.nan), where=area_label > 0)
    summary = {
        "aAcc": float(np.round(aacc * 100, 2)),
        "mIoU": float(np.round(np.nanmean(iou) * 100, 2)),
        "mAcc": float(np.round(np.nanmean(acc) * 100, 2)),
    }
    return {
        "summary": summary,
        "aAcc_proportion": aacc,
        "iou": iou,
        "acc": acc,
        "area_intersect": area_intersect,
        "area_union": area_union,
        "area_label": area_label,
    }


def counts_close(left, right, atol=1e-6):
    return all(np.allclose(a, b, atol=atol, rtol=0) for a, b in zip(left, right))
