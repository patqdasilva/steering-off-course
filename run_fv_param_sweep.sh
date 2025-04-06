#!/bin/bash
#SBATCH --account=your_account
#SBATCH --job-name=func_vector
#SBATCH --time=02:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --output=./slurm_out/fv/%x_%j_%u.out

model_name=$1
task=$2
model_fp=$3

echo `date`

# activate your python environment

cd ../function_vectors
python experiments/run_param_sweep.py \
  --model_name "$model_name" \
  --data_name "$task" \
  --model_fp "$model_fp" \
  --nshot_baseline True