import argparse
import csv
import json

from pathlib import Path
from mmengine.config import DictAction
from typing import Dict, Optional

SS_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SS_ROOT.parent


def get_args_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='MMSeg test (and eval) a model')
    parser.add_argument('--config', help='Train config file path')
    parser.add_argument('--checkpoint', help='Checkpoint file path')
    parser.add_argument('--work-dir', help='Directory to save evaluation results as JSON')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help=(
            'Override config options. Format: key=val. For lists use key="[a,b]" or key=a,b. '
            'For nested structures: key="[(a,b),(c,d)]". No spaces allowed.'
        )
    )
    args, _ = parser.parse_known_args()
    return args


def _norm_crop(crop_size: Optional[str]) -> str:
    if not crop_size:
        return ""
    return str(crop_size).lower().replace("×", "x").replace(" ", "")


def documented_config_path(model_name: str, backbone: str, dataset: str, crop_size: str = None) -> Optional[Path]:
    """Return the config recorded for this checkpoint in results/model_provenance.csv."""
    table = REPO_ROOT / "results" / "model_provenance.csv"
    if not table.is_file():
        return None
    crop = _norm_crop(crop_size)
    hits = []
    with table.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["architecture"] != model_name or row["backbone"] != backbone or row["dataset"] != dataset:
                continue
            if crop and row["crop_size"] != crop:
                continue
            if row.get("config"):
                hits.append(row)
    if not hits:
        return None
    if len(hits) > 1:
        crops = ", ".join(sorted({row["crop_size"] for row in hits}))
        raise FileNotFoundError(
            f"Several checkpoints match {model_name}, {backbone}, {dataset}. Pass crop_size: {crops}."
        )
    path = REPO_ROOT / hits[0]["config"]
    if not path.is_file():
        raise FileNotFoundError(f"Recorded config does not exist: {path}")
    return path


def _backbone_tokens(model_name: str, backbone: str):
    if model_name == "unet" and backbone == "custom":
        return ["unet-s5-d16_fcn"]
    if backbone in ("beit-b", "beit-base"):
        return ["beit-base"]
    if backbone in ("beit-l", "beit-large"):
        return ["beit-large"]
    tokens = [backbone]
    if "resnet" in backbone:
        tokens.append(backbone.replace("resnet", "r"))
    if "internimage-" in backbone:
        tokens.append(backbone.replace("-", "_"))
    return tokens


def get_config_path(model_name: str, backbone: str, dataset: str, crop_size: str = None) -> Path:
    model_name = model_name.lower()
    backbone = backbone.lower()
    dataset = dataset.lower()
    crop_size = _norm_crop(crop_size)

    recorded = documented_config_path(model_name, backbone, dataset, crop_size)
    if recorded is not None:
        return recorded

    tokens = _backbone_tokens(model_name, backbone)
    search_dirs = [
        SS_ROOT / "configs" / model_name,
        SS_ROOT / "mmsegmentation" / "configs" / model_name,
    ]
    if model_name == "upernet" and "beit" in backbone:
        search_dirs.append(SS_ROOT / "mmsegmentation" / "configs" / "beit")

    configs_in_dir = []
    for directory in search_dirs:
        if not directory.is_dir():
            continue
        for config_path in directory.iterdir():
            if not config_path.suffix == ".py":
                continue
            stem = config_path.stem
            if model_name != "upernet" and not stem.startswith(model_name):
                continue
            if not any(token in stem for token in tokens):
                continue
            if dataset not in stem and not (dataset == "voc" and "voc12" in stem):
                continue
            if crop_size and crop_size not in stem:
                square = crop_size.split("x")
                if not (len(square) == 2 and square[0] == square[1] and f"_{square[0]}_" in f"_{stem}_"):
                    continue
            configs_in_dir.append(config_path)

    if not configs_in_dir:
        raise FileNotFoundError(
            f"No config found for model '{model_name}' with backbone '{backbone}' on dataset '{dataset}'."
        )
    if len(configs_in_dir) > 1:
        names = ", ".join(path.name for path in configs_in_dir)
        raise FileNotFoundError(f"Several configs match {model_name}, {backbone}, {dataset}: {names}")

    return configs_in_dir[0]


def get_checkpoint_path(model_name: str, backbone: str, dataset: str, crop_size: str = None) -> Path:
    model_name = model_name.lower()
    backbone = backbone.lower()
    dataset = dataset.lower()
    crop_size = _norm_crop(crop_size)
    # Published MMSeg filenames use r50 / r101 / r18, not resnet50.
    tokens = _backbone_tokens(model_name, backbone)

    checkpoint_files_path = SS_ROOT / "checkpoints" / model_name
    if not checkpoint_files_path.is_dir():
        checkpoint_files_path = Path("checkpoints") / model_name
    if not checkpoint_files_path.is_dir():
        raise FileNotFoundError(f"No checkpoint directory for model '{model_name}'")

    checkpoints_in_dir = []
    for checkpoint_path in checkpoint_files_path.iterdir():
        if checkpoint_path.suffix != ".pth":
            continue
        stem = checkpoint_path.stem
        if not any(token in stem for token in tokens):
            continue
        if dataset not in stem:
            continue
        if crop_size and crop_size not in stem:
            continue
        checkpoints_in_dir.append(checkpoint_path)

    if len(checkpoints_in_dir) == 0:
        raise FileNotFoundError(
            f"No checkpoint file found for model '{model_name}' with backbone '{backbone}' on dataset '{dataset}'"
        )
    if len(checkpoints_in_dir) > 1:
        raise ValueError(
            f"Multiple checkpoints found for model '{model_name}' with backbone '{backbone}' on dataset '{dataset}'"
        )

    return checkpoints_in_dir[0]


def get_results(work_dir: str) -> Dict[str, Optional[float]]:
    work_dir = Path(work_dir)
    result_dirs = [d for d in work_dir.iterdir() if d.is_dir()]
    if not result_dirs:
        raise FileNotFoundError(f"No result directories found in {work_dir}")
    
    # the result should be in the lastest timestamped directory
    latest_dir = max(result_dirs, key=lambda d: d.name)
    json_path = latest_dir / f"{latest_dir.name}.json"
    if not json_path.is_file():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    
    with open(json_path, "r") as f:
        data = json.load(f)
        
    metrics = ['aAcc', 'mIoU', 'mAcc']
    results = {metric: data.get(metric) for metric in metrics}
    return results