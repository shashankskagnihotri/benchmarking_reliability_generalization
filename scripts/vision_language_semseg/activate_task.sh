#!/bin/bash
# Select the conda environment for this array task. The lookup uses the
# repository manifest and does not import a model.
set -euo pipefail
REPO=/ceph/sagnihot/projects/IJCV_github_repo/benchmarking_reliability_generalization
cd "$REPO"
source /ceph/sagnihot/miniconda3/etc/profile.d/conda.sh
ENV_NAME=$(PYTHONPATH="$REPO/semantic_segmentation" python - << 'PY'
import os
from semsegbench.vlseg.manifest import experiment_for
spec = experiment_for(os.environ["VLSEG_GROUP"], int(os.environ["SLURM_ARRAY_TASK_ID"]))
envs = {
    "san": "vlseg_san",
    "catseg": "vlseg_catseg",
    "sed": "vlseg_sed",
    "fcclip": "vlseg_fcclip",
    "maftp": "vlseg_maft",
    "sam3": "vlseg_sam3",
}
print(envs[spec["model_key"]])
PY
)
conda activate "$ENV_NAME"
echo "environment ${ENV_NAME}"
