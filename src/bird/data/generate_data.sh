#!/bin/bash
set -euo pipefail

module load miniconda
conda activate bird

# Check that at least one argument is provided
if [ "$#" -eq 0 ]; then
    echo "Usage: $0 <num_sequences1> [<num_sequences2> ...]"
    echo "Example: $0 10000 100000 1000000"
    exit 1
fi

# Loop over all provided sequence counts
for NUM_SEQ in "$@"; do
    echo "Generating data with ${NUM_SEQ} sequences..."

    python3 - <<EOF
from bird.data.generate_data import generate_and_save

generate_and_save(
    num_units=${NUM_SEQ},
    filename=f"dna_${NUM_SEQ}_seq.jsonl",
)
EOF

done

echo "All datasets generated."