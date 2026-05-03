#!/bin/bash
set -euo pipefail

module load miniconda
conda activate bird

export PROJECT_ROOT="$HOME/BIRD"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src"

export DATA_DIR="$PROJECT_ROOT/src/bird/data"
export RUN_SAVE_PATH="$PROJECT_ROOT/BART"

export NUM_SEQ=10000
export TASK="splicing"

mkdir -p "$RUN_SAVE_PATH"
mkdir -p "$RUN_SAVE_PATH/finetuned/$TASK"

LOGFILE="$RUN_SAVE_PATH/logs/finetune_splicing.log"
: > "$LOGFILE"
exec &> >(tee -a "$LOGFILE")

export DATASET="$DATA_DIR/dna_${NUM_SEQ}_seq.jsonl"

echo "Starting BART fine-tuning for task ${TASK}..."

python3 -m bird.finetune.finetune_bart \
  --task "$TASK" \
  --data_file "$DATASET" \
  --pretrained_dir "$RUN_SAVE_PATH/checkpoints/bart_k3_overlap/best" \
  --output_dir "$RUN_SAVE_PATH/finetuned/$TASK" \
  --batch_size 16 \
  --epochs 10 \
  --lr 5e-5 \
  --k 3 \
  --max_length 256

echo "BART fine-tuning complete!"