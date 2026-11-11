#!/usr/bin/env bash
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_4
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=16
#SBATCH --job-name=train_pascal_segformer_mit-b0_512x512
#SBATCH --output=slurm/train_pascal_segformer_mit-b0_512x512.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=david.schader@students.uni-mannheim.de

echo "Started at $(date)";

start=$(date +%s)

cd mmsegmentation

python tools/train.py ../configs/segformer/segformer_mit-b0_8xb2-160k_voc12aug-512x512.py

end=$('date +%s')
runtime=$((end-start))

echo Runtime: $runtime