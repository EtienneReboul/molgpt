#!/bin/bash
#SBATCH --job-name=gen_classic
#SBATCH --output=logs/gen_classic_%j.out
#SBATCH --error=logs/gen_classic_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
# For H200 instead: --gres=gpu:h200:1

module load conda
source activate molgpt

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# Max len: 54, vocab_size: 94 (from molgpt_classic training log)
python generate/generate.py \
    --model_weight checkpoints/molgpt_classic.pt \
    --data_name datasets/molgpt_classic \
    --csv_name gen_classic_300k \
    --tokenization_mode classic \
    --gen_size 300000 \
    --batch_size 512 \
    --vocab_size 94 \
    --block_size 54
