#!/bin/bash

script_name=$1
script_full_path="scripts.$script_name"

echo "Starting $script_full_path"

nvidia-smi

source ~/miniconda3/bin/activate /fs/scratch/PAS2836/pqd/conda_env/steering

python -m $script_full_path

echo "End of Bash Script"
