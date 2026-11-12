#!/usr/bin/env python3
"""Reproduce ReM, GAM, and the model tables from stored benchmark results.

Reads semantic_segmentation/semsegbench/Config-Paths.csv. Does not run
models, attacks, or training. Rewrites the CSV tables under results/.
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SS = ROOT / "semantic_segmentation"
RESULT_CSV = SS / "semsegbench" / "Config-Paths.csv"
OUT = ROOT / "results"
FIG = OUT / "figures"
GH = "https://github.com/shashankskagnihotri/benchmarking_reliability_generalization/blob/main"

CORRUPTIONS = [
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
]
ATTACKS = ["pgd", "segpgd", "cospgd"]
NORMS = ["linf", "l2"]
ACDC = ["fog", "snow", "rain", "night"]
# 76 models x these 27 slots = 2,052, the count stated in the paper.
PAPER_SETTINGS = (
    ["clean"]
    + [f"2d_{name}" for name in CORRUPTIONS]
    + [f"{attack}_{norm}" for attack in ATTACKS for norm in NORMS]
    + ["acdc"]
    + [f"acdc_{name}" for name in ACDC]
)
EXPECTED_CLASSES = {"ade20k": "150", "cityscapes": "19", "voc": "21"}

# 61 COCO detectors x these 67 columns = 4,087, the count stated in the paper.
# Summaries (gam, rem_20, mean_2d, mean_3d) are not extra evaluations.
DET_COCO_SRC = "results/supplementary/detection/method_with_time_stamp.csv"
DET_VOC_SRC = "results/supplementary/detection/pascal_results.csv"
DET_ATTACKS = ["fgsm"] + [f"pgd_{i}" for i in range(1, 21)] + [f"bim_{i}" for i in range(1, 21)]
DET_CORRUPTIONS = [
    "gaussian_noise_2d", "shot_noise_2d", "impulse_noise_2d", "defocus_blur_2d",
    "glass_blur_2d", "motion_blur_2d", "zoom_blur_2d", "snow_2d", "frost_2d",
    "fog_2d", "brightness_2d", "contrast_2d", "elastic_transform_2d",
    "pixelate_2d", "jpeg_compression_2d", "color_quant_3d", "far_focus_3d",
    "fog_3d", "iso_noise_3d", "low_light_3d", "near_focus_3d",
    "xy_motion_blur_3d", "z_motion_blur_3d", "h265_crf_3d", "h265_abr_3d",
]
DET_GAM = [name for name in DET_CORRUPTIONS if not name.startswith("h265")]
DET_COCO_SETTINGS = ["iid"] + DET_ATTACKS + DET_CORRUPTIONS


def num(value: str):
    value = (value or "").strip()
    if value in ("", "#N/A", "nan", "NaN"):
        return None
    return float(value)


def load_models():
    with RESULT_CSV.open(newline="") as handle:
        rows = list(csv.reader(handle))
    models = []
    for index in range(1, len(rows[0])):
        record = {row[0]: row[index] for row in rows}
        record["model_id"] = (
            f"{record['architecture']}_{record['backbone']}_{record['dataset']}_{record['crop_size']}"
        )
        record["column"] = index
        models.append(record)
    return models


def backbone_pattern(backbone: str, architecture: str) -> re.Pattern:
    backbone = backbone.lower()
    if backbone.startswith("resnet"):
        token = "r" + backbone[len("resnet") :]
        return re.compile(rf"(^|[_\-]){token}([_\-.]|$)")
    if backbone.startswith("internimage-"):
        token = "internimage_" + backbone.split("-", 1)[1]
        return re.compile(re.escape(token))
    if backbone == "beit-b":
        return re.compile(r"beit[-_](base|b)([_\-.]|$)")
    if backbone == "beit-l":
        return re.compile(r"beit[-_](large|l)([_\-.]|$)")
    if backbone == "custom" and architecture == "unet":
        return re.compile(r"unet-s5-d16_fcn")
    if backbone.startswith("mit-b"):
        return re.compile(re.escape(backbone) + r"(?!\d)")
    if backbone.startswith("swin-"):
        return re.compile(re.escape(backbone) + r"(?![a-z0-9])")
    return re.compile(re.escape(backbone))


def crop_in_name(name: str, crop: str) -> bool:
    if crop in name:
        return True
    # InternImage ADE20K configs shorten 512x512 / 640x640 / 896x896 to one side.
    if crop in ("512x512", "640x640", "896x896"):
        side = crop.split("x")[0]
        if crop == "512x512" and "512x1024" in name:
            return False
        return re.search(rf"(^|[_\-]){side}([_\-.]|$)", name) is not None
    return False


def dataset_in_name(name: str, dataset: str) -> bool:
    if dataset == "voc":
        return "voc" in name
    return dataset in name


def iter_configs():
    roots = [SS / "mmsegmentation" / "configs", SS / "configs"]
    for root in roots:
        for path in root.rglob("*.py"):
            rel = str(path.relative_to(SS))
            if "acdc" in rel or path.name.endswith("_at.py"):
                continue
            if "/_base_/" in rel.replace("\\", "/"):
                continue
            yield path


def candidate_configs(model):
    pattern = backbone_pattern(model["backbone"], model["architecture"])
    found = []
    for path in iter_configs():
        name = path.name
        rel = str(path.relative_to(SS)).replace("\\", "/")
        arch_dir = f"configs/{model['architecture']}/"
        beit_dir = model["architecture"] == "upernet" and model["backbone"].startswith("beit")
        if arch_dir not in rel and not (beit_dir and "configs/beit/" in rel):
            continue
        if "night-driving" in name or "dark-zurich" in name:
            continue
        if model["architecture"] == "upernet" and "upernet" not in name and "beit" not in name:
            continue
        if not pattern.search(name):
            continue
        if not dataset_in_name(name, model["dataset"]):
            continue
        if not crop_in_name(name, model["crop_size"]):
            continue
        found.append(path)
    return found


def parse_readmes():
    """Map config basename to published mIoU and checkpoint URL."""
    catalog = {}
    for readme in (SS / "mmsegmentation" / "configs").glob("*/README.md"):
        lines = readme.read_text(errors="replace").splitlines()
        header = None
        for line in lines:
            if not line.startswith("|"):
                header = None
                continue
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            lowered = [cell.lower() for cell in cells]
            if "config" in lowered and any(cell == "miou" for cell in lowered):
                header = lowered
                continue
            if header is None or "---" in line:
                continue
            if len(cells) < len(header):
                continue
            joined = " ".join(cells)
            match = re.search(r"([A-Za-z0-9_.+-]+\.py)", joined)
            if not match:
                continue
            basename = match.group(1)
            try:
                miou = float(cells[header.index("miou")])
            except ValueError:
                continue
            urls = re.findall(r"https://[^|\s)]+?\.pth", joined)
            catalog[basename] = {
                "miou": miou,
                "checkpoint": urls[0] if urls else "",
                "readme": str(readme.relative_to(ROOT)),
            }
    return catalog


def base_files(path: Path, text: str):
    block = re.search(r"_base_\s*=\s*\[(.*?)\]", text, re.S)
    if block:
        items = re.findall(r"['\"]([^'\"]+)['\"]", block.group(1))
    else:
        single = re.search(r"_base_\s*=\s*['\"]([^'\"]+)['\"]", text)
        items = [single.group(1)] if single else []
    resolved = []
    for item in items:
        candidate = (path.parent / item).resolve()
        if candidate.is_file():
            resolved.append(candidate)
    return resolved


def optimizer_fields(text: str):
    index = text.find("optimizer")
    if index < 0:
        return "", "", ""
    window = text[index : index + 450]
    kind = re.search(r"type=['\"](\w+)['\"]", window)
    lr = re.search(r"\blr=([0-9.eE+-]+)", window)
    extra = []
    for key in ("momentum", "weight_decay", "betas"):
        found = re.search(rf"{key}=(\([^)]*\)|[0-9.eE+-]+)", window)
        if found:
            extra.append(f"{key}={found.group(1)}")
    return (
        kind.group(1) if kind else "",
        lr.group(1) if lr else "",
        ", ".join(extra),
    )


def config_chain(path: Path, seen=None):
    seen = seen or set()
    resolved = path.resolve()
    if resolved in seen or not resolved.is_file():
        return []
    seen.add(resolved)
    text = resolved.read_text(errors="replace")
    chain = [(resolved, text)]
    for base in base_files(resolved, text):
        chain.extend(config_chain(base, seen))
    return chain


def train_fields(path: Path):
    chain = config_chain(path)
    text = chain[0][1] if chain else path.read_text(errors="replace")
    bases = [item[0] for item in chain[1:]]
    kind = lr = extra = ""
    for _, body in chain:
        if "AdamW" not in body and "SGD" not in body:
            continue
        found_kind, found_lr, found_extra = optimizer_fields(body)
        if found_kind in ("AdamW", "SGD", "Adam"):
            kind, lr, extra = found_kind, found_lr, found_extra
            break
    iterations = ""
    for _, body in chain:
        found_iters = re.search(r"max_iters=(\d+)", body)
        if found_iters:
            iterations = found_iters.group(1)
            break
    if not iterations:
        short = re.search(r"-(\d+)k_", path.name)
        if short:
            iterations = str(int(short.group(1)) * 1000)

    aug_file = ""
    aug = ""
    for base, body in chain:
        pipeline = re.search(
            r"train_pipeline\s*=\s*\[(.*?)\]\s*(?:test_pipeline|img_ratios|train_dataloader)",
            body,
            re.S,
        )
        if pipeline:
            aug_file = str(base.relative_to(ROOT))
            aug = ", ".join(re.findall(r"(?<![\w])type=['\"](\w+)['\"]", pipeline.group(1)))
            break
    urls = re.findall(r"https://[^'\"\s]+", text)
    classes = re.findall(r"num_classes(?:=|\s*=\s*)(\d+)", text)
    return {
        "optimizer": kind,
        "lr": lr,
        "optimizer_extra": extra,
        "iterations": iterations,
        "augmentation": aug,
        "augmentation_file": aug_file,
        "pretrain_urls": " ".join(dict.fromkeys(urls)),
        "num_classes": " ".join(dict.fromkeys(classes)),
        "bases": [str(base.relative_to(ROOT)) for base in bases],
    }


def training_scripts():
    """Basename -> uncommented train scripts that name that config."""
    found = {}
    for script in sorted((SS / "scripts").glob("train_*.sh")):
        for line in script.read_text(errors="replace").splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            for name in re.findall(r"[\w./+-]+\.py", stripped):
                found.setdefault(Path(name).name, []).append(str(script.relative_to(ROOT)))
    return found


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def gh(path: str) -> str:
    return f"{GH}/{path}"


def _plain_recipe(path: Path) -> bool:
    name = path.name
    return not any(token in name for token in ("-amp-", "rsb", "mg124"))


def choose_config(model, catalog, scripts):
    candidates = candidate_configs(model)
    recorded = num(model["mIoU_clean"])
    scored = []
    for path in candidates:
        published = catalog.get(path.name)
        distance = None
        if published and recorded is not None:
            distance = abs(published["miou"] - recorded)
        scored.append((path, published, distance))
    with_distance = [item for item in scored if item[2] is not None]
    with_distance.sort(key=lambda item: (item[2], 0 if _plain_recipe(item[0]) else 1))

    path = None
    published = None
    origin = ""
    note = ""
    if with_distance and with_distance[0][2] <= 0.2:
        near = [item for item in with_distance if item[2] <= with_distance[0][2] + 0.05]
        plains = [item for item in near if _plain_recipe(item[0])]
        best = plains[0] if plains else with_distance[0]
        rest = [item[2] for item in with_distance if item[0] != best[0]]
        second = min(rest) if rest else 999
        if best[2] + 0.05 < second or best[2] <= 0.05:
            path, published = best[0], best[1]
            origin = "publicly released"
            if best[2] > 0.05:
                note = f"Benchmark clean mIoU is {recorded}; the published row for this config reports {published['miou']}."

    if path is None:
        scripted = [item for item in scored if scripts.get(item[0].name)]
        if len(scripted) == 1:
            path, published = scripted[0][0], scripted[0][1]
            origin = "trained by authors"
            if published and recorded is not None and abs(published["miou"] - recorded) > 0.2:
                note = (
                    f"Clean mIoU {recorded} does not match the published checkpoint "
                    f"({published['miou']}) for {published and scripted[0][0].name}."
                )
        elif len(candidates) == 1:
            path = candidates[0]
            published = catalog.get(path.name)
        elif candidates:
            local = [item for item in candidates if "mmsegmentation/configs" not in rel(item)]
            path = local[0] if len(local) == 1 else None
            if path is not None:
                published = catalog.get(path.name)
                note = "Several files match this checkpoint. The local training config is listed."
        else:
            return None
    if path is None:
        return None

    fields = train_fields(path)
    script_list = scripts.get(path.name, [])
    text = path.read_text(errors="replace")
    if origin != "publicly released":
        if "InternImage" in text and "OpenGVLab" in text:
            origin = "publicly released"
            note = (
                "Vendored InternImage config. The file records the backbone pretraining URL only; "
                "the segmentation checkpoint URL is not in this repository."
            )
        elif script_list:
            origin = "trained by authors"
        elif published and len(candidates) == 1:
            origin = "publicly released"
            note = (
                f"Only config file for this checkpoint. Benchmark clean mIoU is {recorded}; "
                f"the published row reports {published['miou']}."
            )
        elif "mmsegmentation/configs" not in rel(path):
            origin = "training config present; launch not confirmed"
            note = (note + " " if note else "") + "The training config is in the repository. No uncommented training script launches it."
        else:
            origin = "unavailable"
            note = (note + " " if note else "") + "No published checkpoint match and no training script names this file."
    if origin == "publicly released" and script_list and published:
        note = (note + " " if note else "") + "A training script also names a config with this filename; the clean mIoU matches the published checkpoint."

    classes = fields["num_classes"].split()
    expected = EXPECTED_CLASSES[model["dataset"]]
    if classes and expected not in classes:
        origin = "not confirmed"
        note = (note + " " if note else "") + (
            f"This file sets num_classes={classes[0]}; {model['dataset']} has {expected} classes. "
            "It is the closest config in the repository, not a confirmed training config for the evaluated checkpoint."
        )
    checkpoint = ""
    if published and origin == "publicly released" and published.get("checkpoint"):
        checkpoint = published["checkpoint"]
    elif origin == "trained by authors":
        checkpoint = (
            "not released; place the .pth in "
            f"semantic_segmentation/checkpoints/{model['architecture']}/ "
            "(ResNet files are matched as r50, r101, or r18; other backbones are matched as written; "
            "the filename must also contain the dataset)"
        )
    else:
        checkpoint = "segmentation checkpoint URL not recorded in this repository"
    if fields["pretrain_urls"]:
        pretrain = fields["pretrain_urls"]
    else:
        pretrain = "not recorded in the config"
    return {
        "config": rel(path),
        "origin": origin,
        "checkpoint": checkpoint,
        "pretrain": pretrain,
        "iterations": fields["iterations"] or "not recorded",
        "optimizer": fields["optimizer"] or "not recorded",
        "lr": fields["lr"] or "not recorded",
        "optimizer_extra": fields["optimizer_extra"],
        "augmentation": fields["augmentation"] or "not recorded",
        "augmentation_file": fields["augmentation_file"],
        "training_scripts": ";".join(script_list),
        "note": note.strip(),
        "published_miou": "" if not published else published["miou"],
    }


def worst(values):
    present = [value for value in values if value is not None]
    if not present:
        return None, 0
    return min(present), len(present)


def fmt(value):
    if value is None:
        return ""
    return f"{value:.4f}".rstrip("0").rstrip(".")


def setting_meta(setting: str):
    if setting == "clean":
        return "clean", "", "", "", ""
    if setting.startswith("2d_"):
        return "corruption", setting[3:], "", "3", "severity 3; paper Appendix D"
    if setting.startswith("acdc_"):
        return "acdc", setting[5:], "", "", "real adverse condition"
    if setting == "acdc":
        return "acdc", "aggregate", "", "", "stored aggregate; not the mean of the four conditions"
    attack, norm = setting.rsplit("_", 1)
    if norm == "linf":
        budget = "epsilon=8/255; alpha=0.01; iterations=20"
    else:
        budget = "epsilon=64; alpha=0.1; iterations=20"
    return "attack", attack, norm, budget, "paper Appendix D; semsegbench/evals.py retrieve_existing"


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def placeholder(value: str) -> bool:
    return str(value).strip() in ("", "999", "999.0")


def detection_origin(flag: str) -> str:
    if flag == "0":
        return "supplementary pretrained=0"
    if flag == "1":
        return "supplementary pretrained=1"
    return "not recorded"


def coco_evaluations(rows):
    evaluations = []
    missing = []
    for row in rows:
        model_id = row["architecture"]
        for setting in DET_COCO_SETTINGS:
            value = row[setting].strip()
            if placeholder(value):
                missing.append({
                    "model_id": model_id,
                    "setting": setting,
                    "reason": "placeholder 999.0 in method_with_time_stamp.csv",
                })
                continue
            if setting == "iid":
                kind = "clean"
            elif setting in DET_ATTACKS:
                kind = "attack"
            else:
                kind = "corruption"
            iterations = setting.split("_", 1)[1] if setting.startswith(("pgd_", "bim_")) else ""
            evaluations.append({
                "evaluation_id": f"{model_id}__{setting}",
                "task": "object_detection",
                "dataset": row["dataset"],
                "architecture": row["method_short"],
                "backbone": row["backbone"],
                "model_id": model_id,
                "config": "",
                "checkpoint": "",
                "evaluation_type": kind,
                "attack_or_corruption": setting,
                "iterations": iterations,
                "mAP": value,
                "result_file": DET_COCO_SRC,
            })
    return evaluations, missing


def detection_summary(rows, source):
    summary = []
    mismatches = []
    for row in rows:
        clean = num(row["iid"])
        attack_vals = [float(row[name]) for name in ("pgd_20", "bim_20") if name in row and not placeholder(row[name])]
        rem = min(attack_vals) if len(attack_vals) == 2 else None
        if "rem_20" in row and not placeholder(row["rem_20"]) and rem is not None:
            if abs(float(row["rem_20"]) - rem) > 1e-6:
                mismatches.append((row["architecture"], "rem", row["rem_20"], rem))
        gam_vals = [float(row[name]) for name in DET_GAM if name in row and not placeholder(row[name])]
        gam = min(gam_vals) if len(gam_vals) == len(DET_GAM) else None
        if "gam" in row and not placeholder(row["gam"]) and gam is not None:
            if abs(float(row["gam"]) - gam) > 1e-6:
                mismatches.append((row["architecture"], "gam", row["gam"], gam))

        def drop(score):
            if clean is None or score is None or clean == 0:
                return ""
            return f"{(clean - score) / clean:.4f}"

        summary.append({
            "model_id": row["architecture"],
            "architecture": row["method_short"],
            "backbone": row["backbone"],
            "dataset": row["dataset"],
            "parameters": row["num_params"],
            "pretrained_flag": row["pretrained"],
            "origin": detection_origin(row["pretrained"]),
            "clean_mAP": row["iid"],
            "ReM": fmt(rem),
            "stored_rem_20": "" if "rem_20" not in row or placeholder(row.get("rem_20", "")) else row["rem_20"],
            "GAM": fmt(gam),
            "stored_gam": "" if "gam" not in row or placeholder(row.get("gam", "")) else row["gam"],
            "relative_drop_ReM": drop(rem),
            "relative_drop_GAM": drop(gam),
            "config": "",
            "checkpoint": "",
            "result_file": source,
        })
    return summary, mismatches


def scatter_svg(path: Path, title: str, panels):
    width, height = 980, 340
    panel_w = 300
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:sans-serif;font-size:12px;fill:#222}</style>',
        f'<text x="16" y="22" font-size="14">{title}</text>',
    ]
    colors = ["#1f4e79", "#b85c38", "#2f6f4e", "#6b4c9a", "#8a6a12", "#9c3b4a", "#3d6b8a"]
    for index, (xlabel, ylabel, points) in enumerate(panels):
        left = 40 + index * 320
        top = 48
        parts.append(f'<rect x="{left}" y="{top}" width="{panel_w}" height="250" fill="none" stroke="#ccc"/>')
        parts.append(f'<text x="{left}" y="{top + 268}">{xlabel}</text>')
        parts.append(f'<text x="{left}" y="{top - 8}">{ylabel}</text>')
        if not points:
            continue
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        if xmax == xmin:
            xmax += 1
        if ymax == ymin:
            ymax += 1

        def sx(value):
            return left + 8 + (value - xmin) / (xmax - xmin) * (panel_w - 16)

        def sy(value):
            return top + 242 - (value - ymin) / (ymax - ymin) * 232

        archs = []
        for _, _, arch in points:
            if arch not in archs:
                archs.append(arch)
        for x, y, arch in points:
            color = colors[archs.index(arch) % len(colors)]
            parts.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3.2" fill="{color}"/>')
    parts.append("</svg>")
    path.write_text("\n".join(parts))


def supported_models_md(models):
    lines = [
        "# Supported models",
        "",
        "Each row is one evaluated checkpoint. The same architecture, backbone, and dataset can appear twice when the input crop differs. `model_id` is the unique name.",
        "",
        "Checkpoint files are not in git. Public weights are downloaded with `python object_detection/download_models.py` into `checkpoints/<architecture>/`. `load_model` reads the config path in `results/model_provenance.csv` for that `model_id`. `get_checkpoint_path` looks in `checkpoints/<architecture>/`. ResNet filenames are matched as `r50`, `r101`, or `r18`. Other backbones are matched as written (`mit-b0`, `swin-b`). The filename must also contain the dataset, and the crop when more than one file matches. If no local file is found, loading falls back to the MMSegmentation config name, which downloads a published checkpoint when one is registered.",
        "",
        "Provenance, training settings, and checkpoint URLs: [docs/model_provenance.md](../docs/model_provenance.md).",
        "",
        "| model_id | architecture | backbone | dataset | crop |",
        "| --- | --- | --- | --- | --- |",
    ]
    for model in models:
        lines.append(
            f"| `{model['model_id']}` | {model['architecture']} | {model['backbone']} | {model['dataset']} | {model['crop_size']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main():
    models = load_models()
    if len(models) != 76:
        print(f"Expected 76 segmentation checkpoints, found {len(models)}", file=sys.stderr)
        return 1
    catalog = parse_readmes()
    scripts = training_scripts()
    provenance = []
    unresolved = []
    for model in models:
        chosen = choose_config(model, catalog, scripts)
        if chosen is None:
            unresolved.append(model["model_id"])
            chosen = {
                "config": "",
                "origin": "unavailable",
                "checkpoint": "",
                "pretrain": "",
                "iterations": "",
                "optimizer": "",
                "lr": "",
                "optimizer_extra": "",
                "augmentation": "",
                "augmentation_file": "",
                "training_scripts": "",
                "note": "No config file matched this checkpoint.",
                "published_miou": "",
            }
        model.update(chosen)
        provenance.append(
            {
                "task": "semantic_segmentation",
                "architecture": model["architecture"],
                "backbone": model["backbone"],
                "dataset": model["dataset"],
                "model_id": model["model_id"],
                "crop_size": model["crop_size"],
                "origin": model["origin"],
                "config": model["config"],
                "config_url": gh(model["config"]) if model["config"] else "",
                "training_scripts": model["training_scripts"],
                "checkpoint": model["checkpoint"],
                "pretraining": model["pretrain"],
                "input_resolution": model["crop_size"],
                "iterations": model["iterations"],
                "epochs": "iteration schedule; epoch count not recorded",
                "optimizer": model["optimizer"],
                "learning_rate": model["lr"],
                "optimizer_settings": model["optimizer_extra"],
                "augmentation": model["augmentation"],
                "augmentation_config": model["augmentation_file"],
                "parameters": model["num_params"],
                "clean_mIoU": model["mIoU_clean"],
                "note": model["note"],
            }
        )

    evaluations = []
    missing = []
    summary = []
    gam_mismatch = []
    narm_mismatch = []
    for model in models:
        for setting in PAPER_SETTINGS:
            kind, name, norm, budget, budget_note = setting_meta(setting)
            miou = num(model.get(f"mIoU_{setting}", ""))
            if miou is None:
                missing.append(
                    {
                        "model_id": model["model_id"],
                        "setting": setting,
                        "reason": "empty cell in semsegbench/Config-Paths.csv",
                    }
                )
                continue
            evaluations.append(
                {
                    "evaluation_id": f"{model['model_id']}__{setting}",
                    "task": "semantic_segmentation",
                    "dataset": model["dataset"],
                    "architecture": model["architecture"],
                    "backbone": model["backbone"],
                    "model_id": model["model_id"],
                    "config": model["config"],
                    "checkpoint": model["checkpoint"],
                    "evaluation_type": kind,
                    "attack_or_corruption": name,
                    "norm": norm,
                    "budget": budget,
                    "budget_note": budget_note,
                    "aAcc": model.get(f"aAcc_{setting}", ""),
                    "mIoU": model.get(f"mIoU_{setting}", ""),
                    "mAcc": model.get(f"mAcc_{setting}", ""),
                    "result_file": "semantic_segmentation/semsegbench/Config-Paths.csv",
                }
            )

        clean = num(model["mIoU_clean"])
        gam, gam_n = worst([num(model[f"mIoU_2d_{name}"]) for name in CORRUPTIONS])
        stored_gam = num(model["mIoU_2d_gam"])
        if stored_gam is not None and gam is not None and abs(stored_gam - gam) > 1e-6:
            gam_mismatch.append((model["model_id"], stored_gam, gam))
        rem = {}
        rem_n = {}
        for norm in NORMS:
            value, count = worst([num(model[f"mIoU_{attack}_{norm}"]) for attack in ATTACKS])
            rem[norm] = value
            rem_n[norm] = count
            stored = num(model[f"mIoU_narm_{norm}"])
            if stored is not None and value is not None and abs(stored - value) > 1e-6:
                narm_mismatch.append((model["model_id"], norm, stored, value))

        def drop(value):
            if value is None or clean in (None, 0):
                return ""
            return fmt((clean - value) / clean)

        summary.append(
            {
                "model_id": model["model_id"],
                "architecture": model["architecture"],
                "backbone": model["backbone"],
                "dataset": model["dataset"],
                "crop_size": model["crop_size"],
                "origin": model["origin"],
                "parameters": model["num_params"],
                "clean_mIoU": model["mIoU_clean"],
                "ReM_linf": fmt(rem["linf"]),
                "ReM_linf_attacks_present": rem_n["linf"],
                "ReM_l2": fmt(rem["l2"]),
                "ReM_l2_attacks_present": rem_n["l2"],
                "stored_narm_linf": model["mIoU_narm_linf"],
                "stored_narm_l2": model["mIoU_narm_l2"],
                "GAM": fmt(gam),
                "GAM_corruptions_present": gam_n,
                "stored_2d_gam": model["mIoU_2d_gam"],
                "relative_drop_ReM_linf": drop(rem["linf"]),
                "relative_drop_GAM": drop(gam),
                "config": model["config"],
            }
        )

    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "model_provenance.csv", provenance, list(provenance[0].keys()))
    write_csv(OUT / "semseg_evaluations.csv", evaluations, list(evaluations[0].keys()))
    write_csv(OUT / "semseg_missing_settings.csv", missing, ["model_id", "setting", "reason"])
    write_csv(OUT / "semseg_model_summary.csv", summary, list(summary[0].keys()))
    coco_rows = list(csv.DictReader((ROOT / DET_COCO_SRC).open(newline="")))
    voc_rows = list(csv.DictReader((ROOT / DET_VOC_SRC).open(newline="")))
    det_evals, det_missing = coco_evaluations(coco_rows)
    det_summary, det_mismatches = detection_summary(coco_rows, DET_COCO_SRC)
    voc_summary, voc_mismatches = detection_summary(voc_rows, DET_VOC_SRC)
    det_mismatches.extend(voc_mismatches)
    write_csv(OUT / "detection_evaluations.csv", det_evals, list(det_evals[0].keys()))
    write_csv(OUT / "detection_missing_settings.csv", det_missing, ["model_id", "setting", "reason"])
    write_csv(OUT / "detection_model_summary.csv", det_summary + voc_summary, list(det_summary[0].keys()))
    voc_settings = ["iid"] + [name for name in DET_CORRUPTIONS if not name.startswith("h265")]
    voc_evals = []
    for row in voc_rows:
        for setting in voc_settings:
            voc_evals.append({
                "evaluation_id": f"{row['architecture']}__{setting}",
                "task": "object_detection",
                "dataset": row["dataset"],
                "architecture": row["method_short"],
                "backbone": row["backbone"],
                "model_id": row["architecture"],
                "evaluation_type": "clean" if setting == "iid" else "corruption",
                "attack_or_corruption": setting,
                "mAP": row[setting],
                "result_file": DET_VOC_SRC,
            })
    write_csv(OUT / "detection_voc_evaluations.csv", voc_evals, list(voc_evals[0].keys()))
    detection_rows = []
    for row in det_summary + voc_summary:
        detection_rows.append({
            "task": "object_detection",
            "dataset": row["dataset"],
            "architecture": row["architecture"],
            "backbone": row["backbone"],
            "model_id": row["model_id"],
            "origin": row["origin"],
            "config": "",
            "checkpoint": "",
            "pretraining": row["pretrained_flag"],
            "input_resolution": "",
            "iterations_or_epochs": "",
            "optimizer": "",
            "augmentation": "",
            "parameters": row["parameters"],
            "clean_mAP": row["clean_mAP"],
            "source": row["result_file"],
            "result_file": row["result_file"],
        })
    write_csv(OUT / "detection_models_manuscript.csv", detection_rows, list(detection_rows[0].keys()))

    for dataset in ("ade20k", "cityscapes", "voc"):
        subset = [row for row in summary if row["dataset"] == dataset]
        panels = []
        for ylabel, key in (("ReM linf", "ReM_linf"), ("ReM l2", "ReM_l2"), ("GAM", "GAM")):
            points = []
            for row in subset:
                x = num(row["clean_mIoU"])
                y = num(row[key])
                if x is None or y is None:
                    continue
                points.append((x, y, row["architecture"]))
            panels.append(("clean mIoU", ylabel, points))
        scatter_svg(FIG / f"{dataset}.svg", f"{dataset}: clean mIoU against worst-case scores", panels)
    coco_points = []
    for row in det_summary:
        x = num(row["clean_mAP"])
        y = num(row["ReM"])
        g = num(row["GAM"])
        if x is None:
            continue
        coco_points.append((x, y, g, row["architecture"]))
    scatter_svg(
        FIG / "coco.svg",
        "MS-COCO: clean mAP against worst-case scores",
        [
            ("clean mAP", "ReM", [(x, y, arch) for x, y, _, arch in coco_points if y is not None]),
            ("clean mAP", "GAM", [(x, g, arch) for x, _, g, arch in coco_points if g is not None]),
        ],
    )

    (SS / "SUPPORTED_MODELS.md").write_text(supported_models_md(models))

    authors = [row for row in provenance if row["origin"] == "trained by authors"]
    print(f"models {len(models)}")
    print(f"recorded evaluations {len(evaluations)}")
    print(f"empty paper slots {len(missing)}")
    print(f"paper slots {len(models) * len(PAPER_SETTINGS)}")
    print(f"author-trained {len(authors)}")
    for row in authors:
        print(f"  {row['model_id']}  {row['config']}")
    print(f"unresolved {unresolved}")
    print(f"gam mismatches {len(gam_mismatch)}")
    for item in gam_mismatch:
        print("  GAM", item)
    print(f"narm mismatches {len(narm_mismatch)}")
    for item in narm_mismatch:
        print("  NARM", item)
    origins = {}
    for row in provenance:
        origins[row["origin"]] = origins.get(row["origin"], 0) + 1
    print("origins", origins)
    print(f"detection models {len(detection_rows)}")
    print(f"detection recorded evaluations {len(det_evals)}")
    print(f"detection placeholder slots {len(det_missing)}")
    print(f"detection paper slots {len(coco_rows) * len(DET_COCO_SETTINGS)}")
    print(f"detection metric mismatches {len(det_mismatches)}")
    if unresolved or narm_mismatch or det_mismatches:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
