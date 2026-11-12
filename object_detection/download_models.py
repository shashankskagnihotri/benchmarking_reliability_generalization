#!/usr/bin/env python3
"""Download public segmentation checkpoints listed in results/model_provenance.csv.

    python download_models.py --list
    python download_models.py

Files land in semantic_segmentation/checkpoints/<architecture>/.
The name is the published filename when that name already matches
get_checkpoint_path. Otherwise the backbone token, dataset, and crop
are prefixed.

Author-trained weights are not released. Detector checkpoint URLs are
not in this repository. Both are printed and skipped.
"""

from __future__ import annotations

import argparse
import csv
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "results" / "model_provenance.csv"
DEST_ROOT = ROOT / "semantic_segmentation" / "checkpoints"


def backbone_token(architecture: str, backbone: str) -> str:
    # Same tokens as semsegbench.utils._backbone_tokens.
    if architecture == "unet" and backbone == "custom":
        return "unet-s5-d16_fcn"
    if backbone in ("beit-b", "beit-base"):
        return "beit-base"
    if backbone in ("beit-l", "beit-large"):
        return "beit-large"
    if "resnet" in backbone:
        return backbone.replace("resnet", "r")
    if "internimage-" in backbone:
        return backbone.replace("-", "_")
    return backbone


def published_url(checkpoint: str) -> str:
    if checkpoint.startswith("http"):
        return checkpoint.split()[0]
    return ""


def destination(row: dict) -> Path | None:
    url = published_url(row["checkpoint"])
    if not url:
        return None
    published = url.rstrip("/").split("/")[-1]
    if not published.endswith(".pth"):
        published += ".pth"
    token = backbone_token(row["architecture"], row["backbone"])
    dataset = row["dataset"]
    crop = row["crop_size"]
    stem = published[:-4]
    matches = token in stem and dataset in stem and (not crop or crop in stem)
    name = published if matches else f"{token}_{dataset}_{crop}_{published}"
    return DEST_ROOT / row["architecture"] / name


def load_rows():
    with TABLE.open(newline="") as handle:
        return list(csv.DictReader(handle))


def selectable(rows):
    """A file is selectable when the loader's filters keep exactly that file."""
    by_arch = {}
    chosen = {}
    for row in rows:
        path = destination(row)
        if path is None:
            continue
        by_arch.setdefault(row["architecture"], []).append(path.name)
        chosen[row["model_id"]] = path.name
    problems = []
    for row in rows:
        path = destination(row)
        if path is None:
            continue
        token = backbone_token(row["architecture"], row["backbone"])
        hits = []
        for name in by_arch[row["architecture"]]:
            stem = name[:-4]
            if token not in stem or row["dataset"] not in stem:
                continue
            if row["crop_size"] and row["crop_size"] not in stem:
                continue
            hits.append(name)
        if hits != [chosen[row["model_id"]]]:
            problems.append((row["model_id"], hits))
    return problems


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    urllib.request.urlretrieve(url, partial)
    partial.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="print destinations and do not download")
    args = parser.parse_args()
    rows = load_rows()
    problems = selectable(rows)
    if problems:
        for model_id, hits in problems:
            print(f"ambiguous filename for {model_id}: {hits}", file=sys.stderr)
        return 1

    downloaded = skipped = 0
    for row in rows:
        url = published_url(row["checkpoint"])
        path = destination(row)
        if path is None:
            skipped += 1
            if args.list:
                print(f"skip\t{row['model_id']}\tno checkpoint URL in this repository")
            continue
        if args.list:
            state = "exists" if path.is_file() else "download"
            print(f"{state}\t{path.relative_to(ROOT)}\t{url}")
            continue
        if path.is_file():
            print(f"exists {path.relative_to(ROOT)}")
            continue
        print(f"download {row['model_id']}")
        download(url, path)
        downloaded += 1
    if not args.list:
        print(f"downloaded {downloaded}")
        print(f"skipped {skipped} checkpoints with no URL in this repository")
        print("Those are the author-trained segmentation weights and every detector.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
