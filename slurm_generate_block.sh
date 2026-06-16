#!/bin/bash
#SBATCH --job-name=gen_block
#SBATCH --output=logs/gen_block_%j.out
#SBATCH --error=logs/gen_block_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
# For H200 instead: --gres=gpu:h200:1

module load conda
source activate molgpt

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# Max len: 23, vocab_size: 119647 (from molgpt_block training log)
python generate/generate.py \
    --model_weight checkpoints/molgpt_block.pt \
    --data_name datasets/molgpt_block \
    --csv_name gen_block_300k \
    --tokenization_mode block \
    --block_vocab_path datasets/block_vocab.txt \
    --gen_size 300000 \
    --batch_size 512 \
    --vocab_size 119647 \
    --block_size 23
