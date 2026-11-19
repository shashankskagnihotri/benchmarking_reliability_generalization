"""Pooled ACDC mIoU is not the mean of the four condition mIoUs."""

from __future__ import annotations

import numpy as np

from semsegbench.vlseg.metrics import add_counts, empty_counts, intersect_and_union, metrics_from_counts


def _case():
    # Condition A is one correct class-0 pixel. Condition B is 100 wrong class-0 pixels.
    # Their mIoUs are 100 and 0. The pooled IoU is 1/101.
    pred_a = np.zeros((1, 1), dtype=np.int64)
    label_a = np.zeros((1, 1), dtype=np.int64)
    pred_b = np.zeros((10, 10), dtype=np.int64)
    label_b = np.ones((10, 10), dtype=np.int64)
    counts_a = intersect_and_union(pred_a, label_a, num_classes=2)
    counts_b = intersect_and_union(pred_b, label_b, num_classes=2)
    pooled = add_counts(empty_counts(2), counts_a)
    pooled = add_counts(pooled, counts_b)
    metric_a = metrics_from_counts(*counts_a)["summary"]["mIoU"]
    metric_b = metrics_from_counts(*counts_b)["summary"]["mIoU"]
    metric_pool = metrics_from_counts(*pooled)["summary"]["mIoU"]
    arithmetic = (metric_a + metric_b) / 2
    if metric_pool == arithmetic:
        raise AssertionError(f"pooled {metric_pool} unexpectedly equals the condition mean {arithmetic}")
    iou = pooled[0] / pooled[1]
    if not np.isclose(iou[0], 1 / 101):
        raise AssertionError(f"class-0 pooled IoU {iou[0]} is not 1/101")
    # Every condition is included once: the pooled label area is the sum of the parts.
    if pooled[3][0] != counts_a[3][0] + counts_b[3][0]:
        raise AssertionError("a condition was dropped or repeated")
    if pooled[3][1] != counts_a[3][1] + counts_b[3][1]:
        raise AssertionError("a condition was dropped or repeated")
    rebuilt = add_counts(counts_a, counts_b)
    if not all(np.array_equal(a, b) for a, b in zip(pooled, rebuilt)):
        raise AssertionError("summing condition counts does not rebuild the pool")


if __name__ == "__main__":
    _case()
    print("acdc aggregate regression passed")
