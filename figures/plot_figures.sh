module load miniconda
conda activate bird

python finetune_comparison_plots.py \
  --project_root "$HOME/BIRD" \
  --num_seq 10000