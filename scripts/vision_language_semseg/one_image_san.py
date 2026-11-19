"""One ADE20K image through the official SAN checkpoint.

Checks class count, finite scores, and a nonzero image gradient. This is
the GPU check before the 2,000-image clean evaluation.
"""

import sys

import torch

from semsegbench.vlseg.adapters.san import SANSegmentor
from semsegbench.vlseg.data import list_samples, load_sample


def main() -> None:
    sample = list_samples("ade20k")[0]
    _original, resized, label = load_sample(sample)
    image = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float()
    print(
        f"image {sample.image_id} resized {tuple(resized.shape)} "
        f"dtype {image.dtype} min {float(image.min())} max {float(image.max())} "
        f"label {tuple(label.shape)} {int(label.min())} {int(label.max())}",
        flush=True,
    )
    segmentor = SANSegmentor({"dataset": "ade20k", "model_key": "san"}, "cuda")
    segmentor.freeze()
    image = image.cuda().requires_grad_(True)
    scores = segmentor.scores(image)
    finite = bool(torch.isfinite(scores).all())
    print(
        f"scores {tuple(scores.shape)} min {float(scores.min())} max {float(scores.max())} finite {finite}",
        flush=True,
    )
    if scores.shape != (1, 150, resized.shape[0], resized.shape[1]) or not finite:
        raise SystemExit("SAN scores are not a finite 150-class map at the image size")
    scores.sum().backward()
    grad = image.grad
    total = float(grad.abs().sum())
    print(f"grad_abs_sum {total} finite {bool(torch.isfinite(grad).all())}", flush=True)
    if total == 0 or not torch.isfinite(grad).all():
        raise SystemExit("SAN image gradient is zero or non-finite")
    print(f"one_image_ok {sample.image_id}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
