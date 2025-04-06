# *Steering off Course*: Reliability Challenges in Steering Language Models
### Official code for ***Steering off Course*: Reliability Challenges in Steering Language Models**

## [Arxiv Preprint](https://arxiv.org/)

# Getting Started
1. Create a fresh environment using each repository's guidelines to ensure version stability
### Note about huggingface versions:
1. Some models require newer versions of huggingface than others.
2. Please follow the official model cards of these models (e.g. [OLMo2](https://huggingface.co/allenai/OLMo-2-1124-7B)) to determine when this is the case.
## Repository Structure
- ./DoLA, ./function_vectors, and ./icl_task_vectors
    - These contain each codebase we use from their corresponding papers.
- ./notebooks
    - contains scripts for replicating our results, detailed below.
- ./config.py
    - contains helpful information about model paths and naming conventions
- ./figures
    - contains all the figures from all the results from all the methods, models, and tasks we tested.
    - To replicate our plots, please use each of the ./notebooks/plot*.ipynb

# Running our Experiments
## To run using Bash or Slurm
- We set up a convenient Jupyter Notebook ./notebooks/run_slurm.ipynb to run all experiments with easy-to-set hyperparameters.
- This is our main method for running our experiments, and helps with managing the many input parameters.
    - The inputs to the ./*.sh files are the same as detailed below in the [Python guide](#to-run-using-python)
- Please modify the Slurm directives to your own details in the corresponding "./*.sh" files.
- To run without slurm, please remove the 'sbatch', and '*slurm_cmd' from the subprocess.run call

## To run using Python
### Function Vectors
1. In order to create a function vector, you must first compute the indirect effect on a task
```bash
cd function_vectors
python ./src/compute_indirect_effect.py \
    --model_name [model_name] \ # model name
    --model_fp [model_fp] \ # absolute file path to model
    --dataset_name [dataset_name] \ # dataset (see ./function_vectors/dataset_files for tasks)
    --root_data_dir ./dataset_files \ # where the data is stored
    --save_path_root ./results/[model_name] # where to save results
```
2. Then, it is possible to replicate our parameter search
```bash
cd function_vectors
python ./experiments/run_param_sweep.py \
  --model_name [model_name] \ # model name
  --model_fp [model_fp] \ # absolute file path to model
  --data_name [task] \ # dataset (see ./function_vectors/dataset_files for tasks)
  --nshot_baseline True # whether to use the predefined N-shot examples
```
### Task Vectors
```bash
cd icl_task_vectors
./run_script.sh experiments.main # you must set up your desired models in ./configs before running
```
### DoLA
For TruthfulQA Multiple Choice
```bash
cd DoLa
python tfqa_mc_eval.py \
    --model-name [model_name] \ # model name for naming purposes 
    --model-fp [model_fp] \ # absolute file path to model
    --data-path "./tfqa" \ # location where tfqa is downloaded.
    --output-path [out_fp] \ # output file path
    --num-gpus $n_gpu \ # number of gpus
    --early-exit-layers [early_exit_layers] \ # e.g. for a model with 32 layers, a 0-25% bucket is '0,2,4,6,8,32'
    --relative_top $alpha \ # alpha hyperparameter as discussed in the paper 
    --post_softmax "y" \ # whether or not to include a softmax, as discussed in 3.1 "Evaluation"
    --ln_type $ln_type \ # Type of layernorm to apply before unembedding. We set this to 'none'
    --len_bias_const 0 # length bias constant as described in section 3.1 Evaluation, and Table 7
```
For FACTOR
```bash
cd ../DoLa
python factor_eval.py \
    --model-name [model_name] \ # model name for naming purposes 
    --model-fp [model_fp] \ # absolute file path to model
    --data-path "./factor/news_factor.csv" \ # location where news_factor is downloaded.
    --output-path [out_fp] \ # output file path
    --num-gpus $n_gpu \ # number of gpus
    --early-exit-layers [early_exit_layers] \ # e.g. for a model with 32 layers, a 0-25% bucket is '0,2,4,6,8,32'
    --relative_top $alpha \ # alpha hyperparameter as discussed in the paper 
    --post_softmax "y" \ # whether or not to include a softmax, as discussed in 3.1 "Evaluation"
    --ln_type $ln_type \ # Type of layernorm to apply before unembedding. We set this to 'none'
    --len_bias_const 0 # length bias constant as described in section 3.1 Evaluation, and Table 7
```
### Logit Lens
```bash
cd function_vectors
python ./logit_lens/logit_lens.py \
  --model_name [model_name] \ # model name
  --model_fp [model_fp] \ # absolute file path to model
  --data_name [task] \ # dataset (see ./function_vectors/dataset_files for tasks, or 'tqa')
  --n_shots [n_shots] \ # number of examples to be included in the prompt
  --max_samples [max_samples] # maximum number of samples
```