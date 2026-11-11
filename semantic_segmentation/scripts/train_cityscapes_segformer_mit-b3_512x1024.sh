#!/usr/bin/env bash
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_4
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=16
#SBATCH --job-name=train_cityscapes_segformer_mit-b3_512x1024
#SBATCH --output=slurm/ftrain_cityscapes_segformer_mit-b3_512x1024t.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=david.schader@students.uni-mannheim.de

echo "Started at $(date)";

cd mmsegmentation

python tools/train.py ../configs/segformer/segformer_mit-b3_8xb1-160k_cityscapes-512x1024.py

end=$('date +%s')
runtime=$((end-start))

echo Runtime: $runtime