#!/bin/bash
set -euo pipefail

module load miniconda
conda activate bird

export PROJECT_ROOT="$HOME/BIRD"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src"

export DATA_DIR="$PROJECT_ROOT/src/bird/data"
export RUN_SAVE_PATH="$PROJECT_ROOT/GPT"
export NUM_SEQ=10000
export TASK="splicing"

mkdir -p "$RUN_SAVE_PATH"

LOGFILE="$RUN_SAVE_PATH/logs/finetune_splicing.log"
: > "$LOGFILE"
exec &> >(tee -a "$LOGFILE")

echo "Starting SFT for task ${TASK}"

PYTHONPATH=src python -m bird.finetune.finetune_gpt \
  --task $TASK \
  --data_file $DATA_DIR/dna_${NUM_SEQ}_seq.jsonl \
  --pretrained_dir $RUN_SAVE_PATH/checkpoints/gpt_k3_overlap/best \
  --output_dir $RUN_SAVE_PATH/finetuned/$TASK \
  --batch_size 16 \
  --epochs 10 \
  --lr 5e-5 \
  --k 3

echo "Finetuning complete!"