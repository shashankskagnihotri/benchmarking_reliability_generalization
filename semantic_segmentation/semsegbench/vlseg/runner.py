"""Run one manifest entry to completion, with exact-image resumption."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .attacks import ITERATIONS, attack_image
from .corruptions import apply_corruption
from .data import IGNORE_INDEX, list_samples, load_sample
from .manifest import RESULTS_DIR, experiment_for
from .metrics import add_counts, empty_counts, intersect_and_union, metrics_from_counts
from .prompts import class_names

RUNS = RESULTS_DIR / "runs"


def git_revision() -> str:
    repo = Path(__file__).resolve().parents[3]
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _atomic_write(path: Path, payload: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _load_state(path: Path, num_classes: int):
    if not path.exists():
        return {}, empty_counts(num_classes)
    state = np.load(path, allow_pickle=False)
    ids = [str(image_id) for image_id in state["image_ids"].tolist()]
    counts = tuple(state[name] for name in ("intersect", "union", "pred", "label"))
    per_image = {image_id: index for index, image_id in enumerate(ids)}
    if len(per_image) != len(ids):
        raise RuntimeError(f"duplicate image ids in {path}")
    return {"ids": ids, "index": per_image}, counts


def _save_state(path: Path, image_ids, counts):
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez(
        temporary,
        image_ids=np.asarray(image_ids),
        intersect=counts[0],
        union=counts[1],
        pred=counts[2],
        label=counts[3],
    )
    os.replace(temporary, path)


def predict_map(scores: torch.Tensor, label_shape: tuple[int, int]) -> np.ndarray:
    if scores.ndim != 4 or scores.shape[0] != 1:
        raise RuntimeError(f"scores shape {tuple(scores.shape)}")
    if tuple(scores.shape[-2:]) != label_shape:
        scores = F.interpolate(scores.float(), size=label_shape, mode="bilinear", align_corners=False)
    return scores.argmax(dim=1)[0].detach().cpu().numpy().astype(np.int64)


def run_experiment(group: str, index: int, device: str = "cuda"):
    spec = experiment_for(group, index)
    from .adapters import load_segmentor

    experiment_id = spec["experiment_id"]
    run_dir = RUNS / experiment_id
    run_dir.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            {
                "experiment_id": experiment_id,
                "model": spec["model"],
                "checkpoint": spec["checkpoint"],
                "dataset": spec["dataset"],
                "setting": spec["setting"],
                "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
                "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", ""),
                "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "hostname": os.uname().nodename,
            }
        ),
        flush=True,
    )
    condition = spec["condition"] if spec["condition"] not in {"", "aggregate"} else ""
    samples = list_samples(spec["dataset"], condition)
    num_classes = int(spec["num_classes"])
    if len(class_names(spec["dataset"])) != num_classes:
        raise RuntimeError("class count does not match the manifest")
    segmentor = load_segmentor(spec, device)
    segmentor.freeze()
    state_path = run_dir / "counts.npz"
    done, counts = _load_state(state_path, num_classes)
    image_ids = list(done.get("ids", []))
    seen = set(image_ids)
    ranges = []
    for sample in samples:
        if sample.image_id in seen:
            continue
        _original, image, label = load_sample(sample)
        if spec["evaluation_type"] == "corruption":
            image = apply_corruption(image, spec["corruption"], sample.image_id)
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(device)
        if not ranges:
            ranges.append(_range_record("benchmark_rgb", tensor))
        if spec["evaluation_type"] == "adversarial":
            tensor, scores = attack_image(
                segmentor.scores,
                tensor,
                torch.from_numpy(label).unsqueeze(0).to(device),
                spec["attack"],
                spec["norm"],
                experiment_id,
                sample.image_id,
            )
        else:
            with torch.no_grad():
                scores = segmentor.scores(tensor)
        prediction = predict_map(scores, label.shape)
        part = intersect_and_union(prediction, label, num_classes, IGNORE_INDEX)
        counts = add_counts(counts, part)
        image_ids.append(sample.image_id)
        seen.add(sample.image_id)
        _save_state(state_path, image_ids, counts)
    if len(image_ids) != len(samples) or len(set(image_ids)) != len(samples):
        raise RuntimeError(f"{experiment_id} coverage {len(set(image_ids))} of {len(samples)}")
    metrics = metrics_from_counts(*counts)
    result = {
        "experiment_id": experiment_id,
        "model": spec["model"],
        "variant": spec["variant"],
        "backbone": spec["backbone"],
        "checkpoint": spec["checkpoint"],
        "dataset": spec["dataset"],
        "split": spec["split"],
        "evaluation_type": spec["evaluation_type"],
        "setting": spec["setting"],
        "attack": spec["attack"],
        "corruption": spec["corruption"],
        "severity": spec["severity"],
        "norm": spec["norm"],
        "epsilon": spec["epsilon"],
        "alpha": spec["alpha"],
        "iterations": spec["iterations"] if spec["evaluation_type"] == "adversarial" else "",
        "image_count": len(samples),
        "evaluated_class_count": num_classes,
        "mIoU": metrics["summary"]["mIoU"],
        "mAcc": metrics["summary"]["mAcc"],
        "aAcc": metrics["summary"]["aAcc"],
        "completion_status": "complete",
        "output_location": str(run_dir),
        "implementation_revision": git_revision(),
        "attack_iterations_configured": ITERATIONS if spec["evaluation_type"] == "adversarial" else "",
    }
    if spec["setting"] == "aggregate":
        _verify_aggregate(spec, counts, metrics)
    _atomic_write(run_dir / "metrics.json", json.dumps(result, indent=2).encode())
    _atomic_write(run_dir / "ranges.json", json.dumps(ranges, indent=2).encode())
    print(json.dumps({"experiment_id": experiment_id, "status": "complete", "mIoU": result["mIoU"]}), flush=True)
    return result


def _range_record(stage, tensor):
    return {
        "stage": stage,
        "dtype": str(tensor.dtype),
        "min": float(tensor.min()),
        "max": float(tensor.max()),
        "shape": list(tensor.shape),
        "channel_order": "RGB",
    }


def _verify_aggregate(spec, counts, metrics):
    """Recompute the pooled counts from the four condition runs when they exist."""
    parts = []
    for condition in ("fog", "night", "rain", "snow"):
        path = RUNS / f"{spec['model_key']}__acdc__{condition}" / "counts.npz"
        if not path.exists():
            return
        state = np.load(path)
        parts.append(tuple(state[name] for name in ("intersect", "union", "pred", "label")))
    rebuilt = empty_counts(int(spec["num_classes"]))
    for part in parts:
        rebuilt = add_counts(rebuilt, part)
    if not all(np.array_equal(a, b) for a, b in zip(counts, rebuilt)):
        # Condition jobs and the aggregate job both count pixels. They must agree.
        # A mismatch means an image was repeated, omitted, or scored with a different map.
        difference = [float(np.max(np.abs(a - b))) for a, b in zip(counts, rebuilt)]
        raise RuntimeError(f"ACDC aggregate does not match summed condition counts: {difference}")
    rebuilt_metrics = metrics_from_counts(*rebuilt)
    if rebuilt_metrics["summary"] != metrics["summary"]:
        raise RuntimeError("ACDC aggregate summary disagrees with the summed counts")
