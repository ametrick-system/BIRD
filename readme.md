# Benchmarks for Interpreting and Representing DNA (BIRD)
> Yale CPSC 4770 Final Project

> Authors: Amy Metrick, Elliot Lichtman, and Jacob Leshnower

## Environment Setup
```bash
# Clone github into home directory
git clone git@github.com:ametrick-system/BIRD.git

# [IN HOME DIRECTORY] create and activate virtual environment
module load miniconda
conda create -n bird python=3.10
conda activate bird

# Check what CUDA version is running on your setup
nvidia-smi

# Install Pytorch for YOUR CUDA version (below example is done for CUDA 12.4)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# [IN PROJECT DIRECTORY] install the project as an editable package and install project dependencies
pip install -e .
```

## AI Acknowledgements
- The `models` directory was written with the aid of ChatGPT-5.3

## WORK IN PROGRESS NOTES (delete later)
- Support more activation functions (right now just relu, gelu)