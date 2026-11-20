"""Run the manifest row selected by the Slurm array task."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "semantic_segmentation"))

from semsegbench.vlseg.runner import run_experiment


def main():
    group = os.environ["VLSEG_GROUP"]
    index_text = os.environ.get("SLURM_ARRAY_TASK_ID")
    if index_text is None:
        index_text = os.environ["VLSEG_INDEX"]
    index = int(index_text)
    run_experiment(group, index)


if __name__ == "__main__":
    main()
