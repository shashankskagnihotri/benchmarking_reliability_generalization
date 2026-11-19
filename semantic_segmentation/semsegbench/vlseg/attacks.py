"""Untargeted PGD, SegPGD, and CosPGD in the SemSegBench image space.

The projection and initialization are ``cospgd.functions`` (``step_inf``,
``step_l2``, ``init_linf``, ``init_l2``). The loss weighting is the weighting
in ``EncoderDecoder.predict`` and ``EncoderDecoder.segpgd_scale`` /
``cospgd_scale``: plain PGD averages per-pixel cross-entropy, SegPGD applies
the iteration-dependent pixel weights, and CosPGD multiplies that
cross-entropy by the detached cosine weight.

Images are float tensors in ``[0, 255]``. The configuration values are the
benchmark's ``epsilon=8, alpha=0.01`` for Linf and ``epsilon=64, alpha=0.1``
for L2. In this ``[0, 255]`` space the effective step is ``alpha * 255``
(2.55 and 25.5). The L2 radius is the norm of the whole per-image ``CHW``
perturbation, which is what ``Attack.lp_normalize(..., dim=(-1,-2,-3))``
implements. The scale is taken from this contract, not from ``tensor.max()``.
"""

from __future__ import annotations

import hashlib

import torch
import torch.nn.functional as F

ATTACKS = ("PGD", "SegPGD", "CosPGD")
NORMS = ("Linf", "L2")
ITERATIONS = 20

# Configuration values as stored by semsegbench/evals.py.
CONFIG_BUDGET = {
    "Linf": {"epsilon": 8.0, "alpha": 0.01},
    "L2": {"epsilon": 64.0, "alpha": 0.1},
}


def pixel_budget(norm: str) -> tuple[float, float]:
    """Return (epsilon, alpha) in [0, 255] pixel units."""
    cfg = CONFIG_BUDGET[norm]
    epsilon = cfg["epsilon"]
    alpha = cfg["alpha"] * 255.0
    return epsilon, alpha


def attack_seed(experiment_id: str, image_id: str) -> int:
    digest = hashlib.sha256(f"attack:{experiment_id}:{image_id}".encode()).digest()
    return int.from_bytes(digest[:4], "little")


def _cospgd():
    # The benchmark calls cospgd.functions. This is that Attack instance.
    from .cospgd_functions import functions as attack
    return attack


def segpgd_scale(predictions, labels, loss, iteration, iterations, targeted=False):
    """Same weighting as EncoderDecoder.segpgd_scale."""
    lambda_t = iteration / (2 * iterations)
    output_idx = torch.argmax(predictions, dim=1)
    correct = output_idx == labels
    if targeted:
        weighted = torch.where(correct, lambda_t * loss, (1 - lambda_t) * loss)
    else:
        weighted = torch.where(correct, (1 - lambda_t) * loss, lambda_t * loss)
    return weighted.sum() / (predictions.shape[-2] * predictions.shape[-1])


def cospgd_weight(predictions, labels, num_classes, targeted=False):
    """Detached cosine weight from EncoderDecoder.cospgd_scale."""
    transformed = F.one_hot(
        torch.clamp(labels, 0, num_classes - 1),
        num_classes=num_classes,
    ).permute(0, 3, 1, 2).float()
    cossim = F.cosine_similarity(F.softmax(predictions, dim=1), transformed, dim=1)
    if targeted:
        cossim = 1 - cossim
    return cossim.detach()


def _scores_at_label(scores, label):
    if scores.shape[-2:] != label.shape[-2:]:
        scores = F.interpolate(scores, size=label.shape[-2:], mode="bilinear", align_corners=False)
    return scores


def attack_image(score_fn, image, label, attack_name, norm, experiment_id, image_id):
    """Return the attacked ``[0, 255]`` image and the final scores.

    ``score_fn`` maps a differentiable ``[1,3,H,W]`` float image in
    ``[0, 255]`` RGB to class scores ``[1,C,H,W]``. Model weights must not
    require gradients. ``label`` is ``[1,H,W]`` with ignore index 255.
    """
    if attack_name not in ATTACKS:
        raise ValueError(attack_name)
    if norm not in NORMS:
        raise ValueError(norm)
    if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 3:
        raise ValueError(f"expected one RGB image, got {tuple(image.shape)}")
    attack = _cospgd()
    epsilon, alpha = pixel_budget(norm)
    device = image.device
    torch.manual_seed(attack_seed(experiment_id, image_id))
    orig = image.detach()
    if norm == "Linf":
        adv = attack.init_linf(orig, epsilon, clamp_min=0, clamp_max=255)
    else:
        adv = attack.init_l2(orig, epsilon, clamp_min=0, clamp_max=255)
    adv = adv.detach()
    for iteration in range(ITERATIONS):
        adv.requires_grad_(True)
        scores = score_fn(adv)
        if not scores.requires_grad:
            raise RuntimeError("class scores are not connected to the image")
        aligned = _scores_at_label(scores, label)
        loss = F.cross_entropy(aligned, label.long(), ignore_index=255, reduction="none")
        if attack_name == "CosPGD":
            weight = cospgd_weight(aligned, label, aligned.shape[1], targeted=False)
            loss = weight * loss
            objective = loss.mean()
        elif attack_name == "SegPGD":
            objective = segpgd_scale(
                aligned, label, loss, iteration=iteration, iterations=ITERATIONS, targeted=False
            )
        else:
            objective = loss.mean()
        if not torch.isfinite(objective):
            raise RuntimeError(f"non-finite attack loss at iteration {iteration}")
        objective.backward()
        grad = adv.grad
        if grad is None or not torch.isfinite(grad).all():
            raise RuntimeError(f"missing or non-finite image gradient at iteration {iteration}")
        if norm == "Linf":
            adv = attack.step_inf(
                adv, epsilon, data_grad=grad, orig_image=orig, alpha=alpha,
                targeted=False, clamp_min=0, clamp_max=255,
            )
        else:
            adv = attack.step_l2(
                adv, epsilon, data_grad=grad, orig_image=orig, alpha=alpha,
                targeted=False, clamp_min=0, clamp_max=255,
            )
        adv = adv.detach()
    _assert_constraints(orig, adv, norm, epsilon)
    with torch.no_grad():
        final_scores = score_fn(adv)
    return adv.detach(), final_scores.detach()


def _assert_constraints(orig, adv, norm, epsilon, tol=1e-3):
    if not torch.isfinite(adv).all():
        raise RuntimeError("adversarial image is not finite")
    if float(adv.min()) < -tol or float(adv.max()) > 255 + tol:
        raise RuntimeError(f"adversarial image left [0, 255]: {float(adv.min())}, {float(adv.max())}")
    delta = (adv - orig).detach()
    if norm == "Linf":
        linf = float(delta.abs().max())
        if linf > epsilon + tol:
            raise RuntimeError(f"Linf perturbation {linf} exceeds {epsilon}")
    else:
        l2 = float(torch.linalg.vector_norm(delta.reshape(delta.shape[0], -1), ord=2, dim=1).max())
        if l2 > epsilon + tol:
            raise RuntimeError(f"L2 perturbation {l2} exceeds {epsilon}")


def verify_one_projected_step():
    """Check one Linf and one L2 step on a differentiable dummy score."""
    attack = _cospgd()
    torch.manual_seed(0)
    image = torch.full((1, 3, 8, 8), 128.0)
    label = torch.zeros((1, 8, 8), dtype=torch.long)
    label[:, :, 4:] = 1

    def score_fn(adv):
        # Class 0 responds to darker pixels, class 1 to brighter pixels.
        gray = adv.mean(dim=1, keepdim=True)
        return torch.cat([10 - gray, gray], dim=1)

    for norm in NORMS:
        epsilon, alpha = pixel_budget(norm)
        orig = image.detach()
        if norm == "Linf":
            adv = attack.init_linf(orig, epsilon, clamp_min=0, clamp_max=255)
        else:
            adv = attack.init_l2(orig, epsilon, clamp_min=0, clamp_max=255)
        adv = adv.detach().requires_grad_(True)
        aligned = score_fn(adv)
        loss = F.cross_entropy(aligned, label, reduction="none")
        if norm == "Linf":
            objective = loss.mean()
        else:
            objective = loss.mean()
        objective.backward()
        grad = adv.grad
        if grad is None or float(grad.abs().sum()) == 0 or not torch.isfinite(grad).all():
            raise RuntimeError(f"{norm} dummy gradient is missing, zero, or non-finite")
        if norm == "Linf":
            stepped = attack.step_inf(
                adv, epsilon, data_grad=grad, orig_image=orig, alpha=alpha,
                targeted=False, clamp_min=0, clamp_max=255,
            )
        else:
            stepped = attack.step_l2(
                adv, epsilon, data_grad=grad, orig_image=orig, alpha=alpha,
                targeted=False, clamp_min=0, clamp_max=255,
            )
        _assert_constraints(orig, stepped, norm, epsilon)
    return True
