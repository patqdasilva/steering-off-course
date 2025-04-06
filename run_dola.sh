#!/bin/bash
#SBATCH --account=your_account
#SBATCH --job-name=dola
#SBATCH --time=00:50:00
#SBATCH --output=./slurm_out/dola_debug/%x_%j_%u.out

echo `date`

model_name=$1
model_fp=$2
task=$3
n_gpu=$4
alpha=$5
ln_type=$6
early_exit_layers=$7

if [ -n "$early_exit_layers" ]; then
    early_exit_arg="--early-exit-layers $early_exit_layers"
    out_name=lntype-"$ln_type"-alpha-"$alpha"-"$early_exit_layers"
else
    early_exit_arg=""
    out_name=ln_type-"$ln_type"-alpha-none-baseline
fi

if [ "$task" == "tfqa" ]; then
    python_file="tfqa_mc_eval.py"
    data_fp="./tfqa"
elif [ "$task" == "factor" ]; then
    python_file="factor_eval.py"
    data_fp="./factor/news_factor.csv"
else
    echo "Invalid task:" $task
    exit 1
fi

# activate your python environment

cd ../DoLa
python "$python_file" \
    --model-name "$model_name" \
    --model-fp "$model_fp" \
    --data-path "$data_fp" \
    --output-path results2/"$task"/"$model_name"/"$out_name" \
    --num-gpus $n_gpu \
    $early_exit_arg \
    --relative_top $alpha \
    --post_softmax "y" \
    --ln_type $ln_type \
    --len_bias_const 0