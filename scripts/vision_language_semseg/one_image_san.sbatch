#!/bin/bash
#SBATCH --job-name=vlsemseg_san1
#SBATCH --output=slurm/vision_language_semseg/clean/%j.out
#SBATCH --error=slurm/vision_language_semseg/clean/%j.err
#SBATCH --account=ml-staff
#SBATCH --qos=walltime5d
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem=100G
#SBATCH --time=00:40:00
#SBATCH --partition=gpu-vram-94gb
#SBATCH --gres-flags=enforce-binding

# One ADE20K image on one H100. Not one of the 258 full evaluations.
set -euo pipefail
cd /ceph/sagnihot/projects/IJCV_github_repo/benchmarking_reliability_generalization
source /ceph/sagnihot/miniconda3/etc/profile.d/conda.sh
set +u
conda activate vlseg_san
set -u
export PYTHONPATH="semantic_segmentation/.vlseg_py:semantic_segmentation/mmsegmentation:semantic_segmentation${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
echo "start $(date --iso-8601=seconds) host $(hostname) gpu ${CUDA_VISIBLE_DEVICES:-} job ${SLURM_JOB_ID:-}"
python scripts/vision_language_semseg/one_image_san.py
status=$?
echo "end $(date --iso-8601=seconds) exit ${status}"
exit ${status}
