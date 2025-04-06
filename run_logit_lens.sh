#!/bin/bash
#SBATCH --account=your_account
#SBATCH --job-name=logit_lens
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --output=./slurm_out/logit_lens/%x_%j_%u.out

model_name=$1
task=$2
model_fp=$3
n_shots=$4
max_samples=$5

echo `date`

# activate your python environment

cd ../function_vectors
python logit_lens/logit_lens.py \
  --model_name "$model_name" \
  --model_fp "$model_fp" \
  --data_name "$task" \
  --n_shots "$n_shots" \
  --max_samples "$max_samples"


