#!/bin/bash
set -euo pipefail

module load miniconda
conda activate bird

export PROJECT_ROOT="$HOME/BIRD"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src"

export DATA_DIR="$PROJECT_ROOT/src/bird/data"
export RUN_SAVE_PATH="$PROJECT_ROOT/BERT"

export NUM_SEQ=100000

mkdir -p "$RUN_SAVE_PATH"

LOGFILE="$RUN_SAVE_PATH/logs/pretrain.log"
: > "$LOGFILE"
exec &> >(tee -a "$LOGFILE")

export DATASET="$DATA_DIR/dna_${NUM_SEQ}_seq.jsonl"

echo "Starting BERT pretraining..."

python3 -m bird.pretrain.pretrain_bert \
  --data_path "$DATASET" \
  --output_dir "$RUN_SAVE_PATH/checkpoints/bert_k3_overlap" \
  --k 3 \
  --overlapping \
  --max_length 256 \
  --mlm_probability 0.15 \
  --batch_size 16 \
  --epochs 10 \
  --learning_rate 2e-4 \
  --weight_decay 0.01 \
  --d_model 256 \
  --num_heads 8 \
  --d_ff 512 \
  --num_layers 4 \
  --dropout 0.1

echo "BERT pretraining complete!"

echo "Plotting pretraining loss..."

python3 - <<EOF
from bird.pretrain.pretrain_utils import plot_pretraining_loss_from_log

plot_pretraining_loss_from_log(
    log_path="${LOGFILE}",
    output_path="${RUN_SAVE_PATH}/pretrain_loss.png",
)

print("Saved loss plot to ${RUN_SAVE_PATH}/pretrain_loss.png")
EOF