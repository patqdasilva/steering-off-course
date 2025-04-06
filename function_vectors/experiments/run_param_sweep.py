import os, re, json, gc
import time
from tqdm import tqdm
import argparse
import torch, numpy as np
from baukit import TraceDict

import sys

sys.path.append("./src")
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.append(base_dir)
torch.set_grad_enabled(False)

from utils.extract_utils import compute_function_vector
from utils.model_utils import load_gpt_model_and_tokenizer
from utils.prompt_utils import load_dataset
from utils.eval_utils import compute_dataset_baseline, n_shot_eval
import utils.custom_utils as cutil

from config import IMPLEMENTED_MODELS, MODEL_FP_MAP

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default=None, help="model name")
parser.add_argument(
    "--data_name", type=str, default=None, help="dataset to use (see dataset_files)"
)
parser.add_argument(
    "--model_fp", type=str, default=None, help="fp of model to be loaded"
)
parser.add_argument(
    "--nshot_baseline", type=bool, default=False, help="generate baseline n_shot perf"
)
args = parser.parse_args()

# Params
N_TEST = 50
if float(args.model_name.split("-")[1][:-1]) > 27:
    LAMBDAS = [1, 4, 16]
    N_HEADS = [1024, 512, 64, 2][::-1]
else:
    LAMBDAS = [0.5, 1, 2, 4, 8, 16, 32, 64]
    N_HEADS = [2, 16, 32, 64, 128, 256, 512][::-1]

model, tokenizer, model_config = load_gpt_model_and_tokenizer(
    args.model_fp, args.model_name
)

# Load Preliminaries
dataset = load_dataset(args.data_name, root_data_dir="./dataset_files", seed=0)
mean_activations_path = f"./results/{args.model_name}/{args.data_name}/{args.data_name}_mean_head_activations.pt"
indirect_effect_path = (
    f"./results/{args.model_name}/{args.data_name}/{args.data_name}_indirect_effect.pt"
)
mean_activations = torch.load(mean_activations_path, weights_only=False)
indirect_effect = torch.load(indirect_effect_path, weights_only=False)

# Generate baseline performance with n+1 -> 0 shots for a dataset
if args.nshot_baseline:
    results = compute_dataset_baseline(
        dataset,
        model,
        model_config,
        tokenizer,
        n_shots=10,
        seed=0,
        filter_set=range(N_TEST),
    )
    from pprint import pprint

    pprint(results)
    clean_results = {}
    for n_shot in range(len(results)):
        acc = np.mean(np.array(results[n_shot]["clean_rank_list"]) == 0)
        clean_results[n_shot] = results[n_shot]["clean_topk"][0][1]
    cutil.save_json(args.model_name, args.data_name, clean_results)


for n_top_heads in N_HEADS:
    if n_top_heads > model_config["n_heads"]*model_config["n_layers"]:
        print("skipping n_top_heads", n_top_heads, "as model only has", model_config["n_heads"]*model_config["n_layers"])
        continue
    for lda in LAMBDAS:
        start_time = time.time()
        already_ran = cutil.check_fin(
            args.model_name,
            args.data_name,
            model.config._name_or_path,
            n_top_heads,
            lda,
            N_TEST,
        )
        if already_ran:
            print(
                "ALREADY COMPLETE:", "n_top_heads", n_top_heads, "lambda", lda, sep="\t"
            )
            continue
        print("n_top_heads", n_top_heads, "lambda", lda)
        clean = {lid: [] for lid in range(model_config["n_layers"])}
        interv = {lid: [] for lid in range(model_config["n_layers"])}
        FV, top_heads = compute_function_vector(
            mean_activations,
            indirect_effect,
            model,
            model_config,
            n_top_heads=n_top_heads,
            token_class_idx=-1,
        )
        FV *= lda
        fv_norms_all = {lid: [] for lid in range(model_config["n_layers"])}
        hv_norms2_all = {lid: [] for lid in range(model_config["n_layers"])}
        for EDIT_LAYER in tqdm(
            range(model_config["n_layers"]), total=model_config["n_layers"]
        ):
            if any(x == EDIT_LAYER for x in range(model_config["n_layers"])):
                results, (hv_norms1, hv_norms2, hv_std, hv_ms, hv_spar, fv_norms) = (
                    n_shot_eval(
                        dataset,
                        FV,
                        edit_layer=EDIT_LAYER,
                        n_shots=0,
                        model=model,
                        model_config=model_config,
                        tokenizer=tokenizer,
                        filter_set=range(N_TEST),
                    )
                )

                fv_norms_all[EDIT_LAYER] = np.mean(fv_norms)
                hv_norms2_all[EDIT_LAYER] = np.mean(hv_norms2)
                clean[EDIT_LAYER] = results["clean_topk"][0][1]
                interv[EDIT_LAYER] = results["intervention_topk"][0][1]
        n_heads, lda = cutil.clean_numbers(n_top_heads, lda)
        title = f"{os.path.basename(model.config._name_or_path)} | {args.data_name} | {n_heads}-top-heads | {lda}-lambda | {N_TEST}-test-samples"
        if lda == 1:
            cutil.save_json2(
                args.model_name, args.data_name, "hv_norms2", "hv_norms2", hv_norms2_all
            )
            cutil.save_json2(
                args.model_name, args.data_name, "fv_norms", "fv_norms", fv_norms_all
            )
        cutil.save_json2(args.model_name, args.data_name, title, "0_shot_w_FV", interv)

        end_time = time.time()
        print(
            "n_top_heads",
            n_heads,
            "lambda",
            lda,
            "time:",
            end_time - start_time,
            sep="\t",
        )
        del clean, interv, hv_norms2_all, fv_norms_all
        gc.collect()
