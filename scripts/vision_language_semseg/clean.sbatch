#!/bin/bash
#SBATCH --job-name=vlsemseg_clean
#SBATCH --output=slurm/vision_language_semseg/clean/%A_%a.out
#SBATCH --error=slurm/vision_language_semseg/clean/%A_%a.err
#SBATCH --account=ml-staff
#SBATCH --qos=walltime5d
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem=100G
#SBATCH --time=7:00:00
#SBATCH --partition=gpu-vram-94gb
#SBATCH --array=0-11%8
#SBATCH --mail-type=ALL
#SBATCH --mail-user=shashank.agnihotri@uni-mannheim.de
#SBATCH --gres-flags=enforce-binding

# 12 CPUs, not 16: these H100 nodes set RestrictedCoresPerGPU=12.
# ml-staff/walltime5d is the association that can submit here.
set -euo pipefail
cd /ceph/sagnihot/projects/IJCV_github_repo/benchmarking_reliability_generalization
export VLSEG_GROUP=clean
source scripts/vision_language_semseg/activate_task.sh
export PYTHONUNBUFFERED=1
echo "start $(date --iso-8601=seconds) host $(hostname) gpu ${CUDA_VISIBLE_DEVICES:-} job ${SLURM_JOB_ID:-} task ${SLURM_ARRAY_TASK_ID:-}"
python scripts/vision_language_semseg/run_experiment.py
status=$?
echo "end $(date --iso-8601=seconds) exit ${status}"
exit ${status}
