#!/bin/bash
#SBATCH --job-name=molgpt_block
#SBATCH --output=logs/molgpt_block_%j.out
#SBATCH --error=logs/molgpt_block_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
# For H200 instead: --gres=gpu:h200:1

module load conda
source activate molgpt

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs checkpoints
export PYTHONPATH="${PWD}:${PYTHONPATH}"

python train/train.py \
    --run_name molgpt_block \
    --data_name molgpt_block \
    --tokenization_mode block \
    --block_vocab_path datasets/block_vocab.txt \
    --aug_prob 0.0
