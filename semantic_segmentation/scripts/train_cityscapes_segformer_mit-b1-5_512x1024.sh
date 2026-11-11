#!/usr/bin/env bash
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_4
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=16
#SBATCH --job-name=train_cityscapes_segformer_mit-b1-5_512x1024
#SBATCH --output=slurm/train_cityscapes_segformer_mit-b%a_512x1024.out
#SBATCH --array=2-5
#SBATCH --mail-type=ALL
#SBATCH --mail-user=david.schader@students.uni-mannheim.de

echo "Started at $(date)";

start=$(date +%s)

cd mmsegmentation

if [[ $SLURM_ARRAY_TASK_ID -eq 2 ]]
then
    python tools/train.py ../configs/segformer/segformer_mit-b2_8xb1-160k_cityscapes-512x1024.py
elif [[ $SLURM_ARRAY_TASK_ID -eq 3 ]]
then
    python tools/train.py ../configs/segformer/segformer_mit-b3_8xb1-160k_cityscapes-512x1024.py
elif [[ $SLURM_ARRAY_TASK_ID -eq 4 ]]
then
    python tools/train.py ../configs/segformer/segformer_mit-b4_8xb1-160k_cityscapes-512x1024.py
elif [[ $SLURM_ARRAY_TASK_ID -eq 5 ]]
then
    python tools/train.py ../configs/segformer/segformer_mit-b5_8xb1-160k_cityscapes-512x1024.py
else
    echo "All submitted"
fi


end=$('date +%s')
runtime=$((end-start))

echo Runtime: $runtime