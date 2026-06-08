#!/bin/bash
#SBATCH --job-name=molgpt_classic
#SBATCH --output=logs/molgpt_classic_%j.out
#SBATCH --error=logs/molgpt_classic_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:l40s:1
# For H200 instead: --gres=gpu:h200:1

module load conda
conda activate molgpt

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs checkpoints
export PYTHONPATH="${PWD}:${PYTHONPATH}"

python train/train.py \
    --run_name molgpt_classic \
    --data_name molgpt_classic \
    --tokenization_mode classic \
    --aug_prob 0.0
