#!/bin/bash
# Select the conda environment for this array task. The lookup uses the
# repository manifest and does not import a model.
set -euo pipefail
REPO=/ceph/sagnihot/projects/IJCV_github_repo/benchmarking_reliability_generalization
cd "$REPO"
source /ceph/sagnihot/miniconda3/etc/profile.d/conda.sh
PYBIN=/ceph/sagnihot/miniconda3/bin/python
ENV_NAME=$(PYTHONPATH="$REPO/semantic_segmentation" "$PYBIN" - << 'PY'
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
# conda’s activate.d scripts reference unset variables.
set +u
conda activate "$ENV_NAME"
set -u
# Stock mmengine 0.10.6. The shared OpenMMLab env points at a RAW-image
# fork of mmengine that adds parameters absent from these checkpoints.
export PYTHONPATH="$REPO/semantic_segmentation/.vlseg_py:$REPO/semantic_segmentation/mmsegmentation:$REPO/semantic_segmentation${PYTHONPATH:+:$PYTHONPATH}"
echo "environment ${ENV_NAME}"
