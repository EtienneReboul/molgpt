#!/bin/bash
set -e

cd "$(dirname "$0")"

export PYTHONPATH="${PWD}:${PYTHONPATH}"

conda run -n molgpt python train/train.py \
    --run_name molgpt_classic \
    --data_name molgpt_classic \
    --tokenization_mode classic \
    --aug_prob 0.0

conda run -n molgpt python train/train.py \
    --run_name molgpt_block \
    --data_name molgpt_block \
    --tokenization_mode block \
    --block_vocab_path datasets/block_vocab.txt \
    --aug_prob 0.0
