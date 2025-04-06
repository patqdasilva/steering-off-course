#!/bin/bash
#SBATCH --account=your_account
#SBATCH --job-name=CIE
#SBATCH --output=./slurm_out/cie/%x_%j_%u.out

echo `date`


model_name=$1
data_name=$2
model_fp=$3

# activate your python environment

cd ../function_vectors
python ./src/compute_indirect_effect.py \
  --dataset_name "$data_name" \
  --model_fp "$model_fp" \
  --model_name "$model_name" \
  --root_data_dir ./dataset_files \
  --save_path_root "./results/$model_name"

# time = 40 min 7B | 4 hr 70B