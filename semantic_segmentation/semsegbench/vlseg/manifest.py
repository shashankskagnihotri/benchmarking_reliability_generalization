"""The 258-experiment matrix for the RGB-text extension."""

from __future__ import annotations

import csv
from pathlib import Path

from .corruptions import CORRUPTIONS, SEVERITY
from .attacks import CONFIG_BUDGET, ITERATIONS

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = REPO_ROOT / "results" / "vision_language_semseg"

MODELS = (
    {
        "model_key": "san",
        "model": "SAN",
        "variant": "san-vit-l14_coco-stuff164k-640x640",
        "backbone": "CLIP ViT-L/14",
        "training_dataset": "COCO-Stuff 164k",
        "pretraining": "CLIP ViT-L/14@336px",
        "trained_on_ade_cityscapes_acdc": "no",
        "checkpoint": "san-vit-l14_20230907-a11e098f.pth",
        "checkpoint_url": "https://download.openmmlab.com/mmsegmentation/v0.5/san/san-vit-l14_20230907-a11e098f.pth",
        "config": "semantic_segmentation/mmsegmentation/configs/san/san-vit-l14_coco-stuff164k-640x640.py",
        "source_repo": "https://github.com/open-mmlab/mmsegmentation",
        "source_revision": "config and SideAdapterCLIPHead bundled in this repository",
        "prompt_protocol": "mmseg CLIPTextEncoder templates=vild, cat_bg=True, background channel dropped at prediction",
    },
    {
        "model_key": "catseg",
        "model": "CAT-Seg",
        "variant": "CAT-Seg (L)",
        "backbone": "CLIP ViT-L/14@336px",
        "training_dataset": "COCO-Stuff",
        "pretraining": "CLIP ViT-L/14@336px",
        "trained_on_ade_cityscapes_acdc": "no",
        "checkpoint": "catseg_l_model_large.pth",
        "checkpoint_url": "https://huggingface.co/spaces/hamacojr/CAT-Seg-weights/resolve/main/model_large.pth",
        "config": "configs/vitl_336.yaml",
        "source_repo": "https://github.com/cvlab-kaist/CAT-Seg",
        "source_revision": "6d3a188af95165147fe2f34a8237fa7d2633e784",
        "prompt_protocol": "PROMPT_ENSEMBLE_TYPE=single: 'A photo of a {} in the scene'",
    },
    {
        "model_key": "sed",
        "model": "SED",
        "variant": "SED (L)",
        "backbone": "ConvNeXt-L",
        "training_dataset": "COCO-Stuff",
        "pretraining": "OpenCLIP ConvNeXt-L laion2b_s29b_b131k_ft_soup",
        "trained_on_ade_cityscapes_acdc": "no",
        "checkpoint": "sed_l.pth",
        "checkpoint_url": "https://drive.google.com/file/d/1zAXE0QXy47n0cVn7j_2cSR85eqxdDGg8/view",
        "config": "configs/convnextL_768.yaml with TEST.FAST_INFERENCE=False",
        "source_repo": "https://github.com/xb534/SED",
        "source_revision": "db20d93c0ec8b0ce7fae267a10f826dfc7da15f5",
        "prompt_protocol": "PROMPT_ENSEMBLE_TYPE=imagenet; FAST_INFERENCE false so this is SED (L), not SED-fast (L)",
    },
    {
        "model_key": "fcclip",
        "model": "FC-CLIP",
        "variant": "FC-CLIP (ResNet50x16)",
        "backbone": "CLIP ResNet50x16",
        "training_dataset": "COCO panoptic",
        "pretraining": "CLIP ResNet50x16",
        "trained_on_ade_cityscapes_acdc": "no",
        "checkpoint": "fcclip_rn50x16.pth (head fcclip_cocopan_r50x16; CLIP ResNet50x16 from the eval config)",
        "checkpoint_url": "https://drive.google.com/file/d/1rJu2UwwtI0VLDl9CyB8Mkgw9qHlnX70W/view",
        "config": "configs/coco/panoptic-segmentation/fcclip/fcclip_convnext_large_eval_ade20k_r50x16.yaml",
        "source_repo": "https://github.com/bytedance/fc-clip",
        "source_revision": "2b0bbe213070d44da9182530fa2e826fef03f974",
        "prompt_protocol": "official FC-CLIP text templates from the ResNet50x16 evaluation config",
    },
    {
        "model_key": "maftp",
        "model": "MAFT+",
        "variant": "MAFTP-Large semantic",
        "backbone": "CLIP ConvNeXt-L",
        "training_dataset": "COCO-Stuff semantic",
        "pretraining": "CLIP ConvNeXt-L",
        "trained_on_ade_cityscapes_acdc": "no",
        "checkpoint": "maftp_l.pth",
        "checkpoint_url": "https://drive.google.com/file/d/1EQo5guVuKkSSZj4bv0FQN_4X9h_Rwfe5/view",
        "config": "configs/semantic/eval.yaml",
        "source_repo": "https://github.com/jiaosiyu1999/MAFT-Plus",
        "source_revision": "fd12806df651d309883229de9503e40533f92689",
        "prompt_protocol": "VILD_PROMPT in maft/utils/text_templetes.py; semantic checkpoint, not maftp_l_pano.pth",
    },
    {
        "model_key": "sam3",
        "model": "SAM 3",
        "variant": "SAM 3 image text",
        "backbone": "SAM 3",
        "training_dataset": "SA-Co",
        "pretraining": "SAM 3",
        "trained_on_ade_cityscapes_acdc": "no",
        "checkpoint": "facebook/sam3",
        "checkpoint_url": "https://huggingface.co/facebook/sam3",
        "config": "sam3.model_builder.build_sam3_image_model",
        "source_repo": "https://github.com/facebookresearch/sam3",
        "source_revision": "31cb7e624e4fe95301b1be1d4e809111593d2e45",
        "prompt_protocol": "one text prompt per class, the stripped class name, with no learned prompt",
    },
)

MODEL_BY_KEY = {row["model_key"]: row for row in MODELS}


def _row(model, dataset, split, evaluation_type, name, norm, index, group):
    budget = CONFIG_BUDGET.get(norm, {})
    condition = ""
    if dataset == "acdc":
        condition = name
        setting = name
    elif evaluation_type == "clean":
        setting = "clean"
    elif evaluation_type == "adversarial":
        setting = f"{name}_{norm}"
    else:
        setting = name
    experiment_id = f"{model['model_key']}__{dataset}__{setting}"
    return {
        "experiment_id": experiment_id,
        "array_group": group,
        "array_index": index,
        "model": model["model"],
        "model_key": model["model_key"],
        "variant": model["variant"],
        "backbone": model["backbone"],
        "checkpoint": model["checkpoint"],
        "dataset": dataset,
        "split": split,
        "condition": condition,
        "evaluation_type": evaluation_type,
        "setting": setting,
        "attack": name if evaluation_type == "adversarial" else "",
        "corruption": name if evaluation_type == "corruption" else "",
        "severity": SEVERITY if evaluation_type == "corruption" else "",
        "norm": norm,
        "epsilon": budget.get("epsilon", ""),
        "alpha": budget.get("alpha", ""),
        "iterations": ITERATIONS if evaluation_type == "adversarial" else "",
        "num_classes": 150 if dataset == "ade20k" else 19,
    }


def build_manifest() -> list[dict]:
    rows = []
    clean_i = adv_i = corr_i = acdc_i = 0
    for model in MODELS:
        rows.append(_row(model, "ade20k", "validation", "clean", "clean", "", clean_i, "clean"))
        clean_i += 1
        rows.append(_row(model, "cityscapes", "validation", "clean", "clean", "", clean_i, "clean"))
        clean_i += 1
        for norm in ("Linf", "L2"):
            for attack in ("PGD", "SegPGD", "CosPGD"):
                rows.append(_row(model, "ade20k", "validation", "adversarial", attack, norm, adv_i, "adversarial"))
                adv_i += 1
        for dataset in ("ade20k", "cityscapes"):
            split = "validation"
            for corruption in CORRUPTIONS:
                rows.append(_row(model, dataset, split, "corruption", corruption, "", corr_i, "corruptions"))
                corr_i += 1
        for condition in ("fog", "night", "rain", "snow", "aggregate"):
            rows.append(_row(model, "acdc", "validation", "acdc", condition, "", acdc_i, "acdc"))
            acdc_i += 1
    _assert_manifest(rows)
    return rows


def _assert_manifest(rows):
    ids = [row["experiment_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate experiment ids")
    counts = {}
    for row in rows:
        counts[row["evaluation_type"]] = counts.get(row["evaluation_type"], 0) + 1
    expected = {"clean": 12, "adversarial": 36, "corruption": 180, "acdc": 30}
    if counts != expected or len(rows) != 258:
        raise RuntimeError(f"manifest counts {counts} total {len(rows)}")
    if len({row["checkpoint"] for row in rows}) != 6:
        raise RuntimeError("expected 6 checkpoints")
    acdc_conditions = [row for row in rows if row["evaluation_type"] == "acdc" and row["setting"] != "aggregate"]
    acdc_agg = [row for row in rows if row["setting"] == "aggregate"]
    if len(acdc_conditions) != 24 or len(acdc_agg) != 6:
        raise RuntimeError("ACDC condition/aggregate split is wrong")
    for group, count in (("clean", 12), ("adversarial", 36), ("corruptions", 180), ("acdc", 30)):
        indices = [int(row["array_index"]) for row in rows if row["array_group"] == group]
        if indices != list(range(count)):
            raise RuntimeError(f"{group} array indices are not 0..{count - 1}")


def write_manifest(path: Path | None = None) -> list[dict]:
    rows = build_manifest()
    path = path or (RESULTS_DIR / "experiment_manifest.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def load_manifest(path: Path | None = None) -> list[dict]:
    path = path or (RESULTS_DIR / "experiment_manifest.csv")
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    _assert_manifest(rows)
    return rows


def experiment_for(group: str, index: int) -> dict:
    rows = load_manifest()
    matches = [row for row in rows if row["array_group"] == group and int(row["array_index"]) == index]
    if len(matches) != 1:
        raise RuntimeError(f"{group}[{index}] matched {len(matches)} rows")
    return matches[0]


def write_model_manifest(path: Path | None = None):
    path = path or (RESULTS_DIR / "model_manifest.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MODELS[0].keys()))
        writer.writeheader()
        writer.writerows(MODELS)
