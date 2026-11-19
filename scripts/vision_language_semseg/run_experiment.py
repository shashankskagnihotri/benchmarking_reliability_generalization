"""Run the manifest row selected by the Slurm array task."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "semantic_segmentation"))

from semsegbench.vlseg.runner import run_experiment


def main():
    group = os.environ["VLSEG_GROUP"]
    index = int(os.environ.get("SLURM_ARRAY_TASK_ID", os.environ["VLSEG_INDEX"]))
    run_experiment(group, index)


if __name__ == "__main__":
    main()
