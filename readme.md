# Benchmarks for Interpreting and Representing DNA (BIRD)

**Summary:** A framework for comparing the performance of different transformer architectures (GPT, BERT, BART) on DNA sequence modeling tasks

**Authors:** Amy Metrick, Elliot Lichtman, and Jacob Leshnower

Yale CPSC 4770 final project

## Environment Setup
```bash
# Clone github into home directory
cd ~/
git clone https://github.com/ametrick-system/BIRD.git

# Create and activate virtual environment
module load miniconda
conda create -n bird python=3.10
conda activate bird

# Check what CUDA version is running on your setup
nvidia-smi

# Install Pytorch for YOUR CUDA version (below example is done for CUDA 12.4)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install other necessary software
pip install numpy
pip install tqdm
pip install matplotlib

# [IN PROJECT DIRECTORY] install the project as an editable package
cd BIRD
pip install -e .

# Make all scripts executable
find . -name "*.sh" -exec chmod +x {} +
```

## Generating Data
If you would like to generate more datasets of varying sizes, navigate to the data directory and run `./generate_data.sh` with any number of arguments denoting the size of the datasets you would like. For example, to get separate datasets of size 100, 1000, and 10000, you would run:  
```bash
./generate_data.sh 100 1000 10000
```
