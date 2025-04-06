import os
import sys
import argparse

this_file_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(this_file_dir)

from tqdm import tqdm

import pandas as pd
from functools import reduce

import torch
import torch.nn.functional as F
from torch.nn import Softmax
import numpy as np

from utils import ll_helpers as llh

sys.path.append(os.path.abspath(".."))
from src.utils import model_utils as mu, prompt_utils as pu, eval_utils as eu

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default=None, help="model name")
parser.add_argument(
    "--data_name",
    type=str,
    default=None,
    help="dataset to use (see dataset_files for tasks, or 'tqa')",
)
parser.add_argument(
    "--model_fp", type=str, default=None, help="fp of model to be loaded"
)
parser.add_argument(
    "--n_shots",
    type=str,
    default=None,
    help="number of examples to be included in the prompt",
)
parser.add_argument(
    "--max_samples",
    type=str,
    default=None,
    help="maximum number of samples (not guaranteed to all exist)",
)
args = parser.parse_args()
model_name = args.model_name
data_name = args.data_name
model_fp = args.model_fp
n_shots = int(args.n_shots)
max_samples = int(args.max_samples)

mu.set_seed(0)

##
print("Loading Model and Placing Hooks...")
## https://web.stanford.edu/~nanbhas/blog/forward-hooks-pytorch/
##
activations = {"h_mha": [], "h_mlp": []}


def getActivation(name):
    def hook(model, input, output):
        activations[name].append(output[:, -1, :].to("cuda:0"))

    return hook


all_hooks = []


def find_hook(m):
    module_name = type(m).__name__
    for k, v in m._forward_hooks.items():
        all_hooks.append((module_name, k, v.__name__))


torch.set_grad_enabled(False)
model, tokenizer, model_config = mu.load_gpt_model_and_tokenizer(model_fp, model_name)
n_layers = model_config["n_layers"]

hooks = []
for i in range(n_layers):
    if "pythia" in model_name:
        hooks.append(
            model.gpt_neox.layers[i].attention.dense.register_forward_hook(
                getActivation("h_mha")
            )
        )
        hooks.append(
            model.gpt_neox.layers[i].mlp.dense_4h_to_h.register_forward_hook(
                getActivation("h_mlp")
            )
        )
    else:
        hooks.append(
            model.model.layers[i].self_attn.o_proj.register_forward_hook(
                getActivation("h_mha")
            )
        )
        hooks.append(
            model.model.layers[i].mlp.down_proj.register_forward_hook(
                getActivation("h_mlp")
            )
        )
# Print hooks
model.apply(find_hook)
print("number of hooks:", len(all_hooks))
print("hooks:", all_hooks)


##
print("Loading Data...")
##
if data_name == "tqa":
    dataset, few_shot, correct_tokens, all_other_tokens = llh.format_truthfulqa(
        max_samples, n_shots
    )
else:
    full_dataset = pu.load_dataset(data_name, root_data_dir="../dataset_files", seed=0)
    dataset = full_dataset["train"]
    few_shot = full_dataset["valid"]


##
print("Performing inference and collect activations...")
##
(
    mha_all_prompts,
    mlp_all_prompts,
    h_ls_all_prompts,
    token_ranks,
    topk,
    target_token_ids,
) = ([], [], [], [], [], [])

for i in tqdm(range(len(dataset))):
    activations = {"h_mha": [], "h_mlp": []}
    if data_name == "tqa":
        if n_shots > 0:
            sentence = [few_shot + "\n\n" + dataset[i]]
        else:
            sentence = [dataset[i]]
        target = correct_tokens[i]
    else:
        # following 2 are unused for ICL tasks
        correct_tokens = [""] * len(dataset)
        all_other_tokens = [""] * len(dataset)
        if n_shots == 0:
            word_pairs = {"input": [], "output": []}
        else:
            n_shot_idx = np.random.choice(len(few_shot), n_shots, replace=False)
            word_pairs = few_shot[n_shot_idx]
        word_pairs_test = dataset[i]
        # Uncomment the following to give the model the correct answer.
        # see https://openreview.net/forum?id=6NNA0MxhCH for motivation
        # word_pairs_test = {'input': word_pairs['input'][-1], 'output': word_pairs['output'][-1]}
        prompt_data = eu.word_pairs_to_prompt_data(
            word_pairs,
            query_target_pair=word_pairs_test,
            prepend_bos_token=False,
            shuffle_labels=False,
        )
        # Get relevant parts of the Prompt
        query, target = (
            prompt_data["query_target"]["input"],
            prompt_data["query_target"]["output"],
        )
        query = query[0] if isinstance(query, list) else query
        sentence = [eu.create_prompt(prompt_data)]

    # Tokenize inputs
    if i == 0:
        print("\nExample prompt")
        print(sentence[0], flush=True)
    target_token_id = eu.get_answer_id(sentence[0], target, tokenizer)
    target_token_ids.append(target_token_id)
    inputs = tokenizer(sentence, return_tensors="pt").to("cuda:0")
    original_pred_idx = len(inputs.input_ids.squeeze()) - 1

    output = model(**inputs, output_hidden_states=True)

    # Process last layer outputs
    logits = output.logits[:, -1, :]  # last token
    clean_rank = eu.compute_individual_token_rank(logits, target_token_id)
    token_ranks.append(clean_rank)
    topk.append(tokenizer.decode(llh.topk_logits(logits, 5)))

    # Get hidden states
    h_ls = [emb[:, -1, :].to("cuda:0") for emb in output.hidden_states]
    h_ls_all_prompts.append(
        torch.stack(h_ls)
    )  # shape: [max_samples, n_layer, 1, resid_size]
    mha_all_prompts.append(torch.stack(activations["h_mha"]))
    mlp_all_prompts.append(torch.stack(activations["h_mlp"]))


h_ls_all_prompts = torch.stack(h_ls_all_prompts)
mha_all_prompts = torch.stack(mha_all_prompts)
mlp_all_prompts = torch.stack(mlp_all_prompts)

print("token ranks (0 means correct, highest=vocab_size):")
print(token_ranks)
scores = 0
for rank in token_ranks:
    scores += int(rank == 0)
print(scores / len(token_ranks), flush=True)

print("Sample top tokens at final layer")
print(topk[0:10])

# LayerNorm all non-last layers
S, L, _, D = h_ls_all_prompts.shape
if "pythia" in model_name:
    normed = model.gpt_neox.final_layer_norm(
        h_ls_all_prompts[:, :-1, :, :].reshape(-1, D)
    ).view(S, L - 1, 1, D)
else:
    normed = (
        model.model.norm(h_ls_all_prompts[:, :-1, :, :].reshape(-1, D))
        .view(S, L - 1, 1, D)
        .to("cuda:0")
    )
normed = torch.cat([normed, h_ls_all_prompts[:, -1:, :, :]], dim=1)

# Apply logit lens
token_logits = llh.logit_lens(model, normed, model_name)
token_probs = F.softmax(token_logits, 3)

##
## Get accuracy by layer
##
probs_shape = token_logits.shape[:2]
pred_acc_by_layer = torch.empty(probs_shape)
corr_acc_by_layer = torch.empty(probs_shape)
second_acc_by_layer = torch.empty(probs_shape)

for sampid in range(S):
    predicted_token = torch.argmax(token_probs[sampid][-1, :, :]).item()
    second_choice_tokens = [
        tokenizer.encode(token)[-1] for token in all_other_tokens[sampid]
    ]
    correct_token = target_token_ids[sampid][0]
    for layerid in range(L):
        pred_acc_by_layer[sampid][layerid] = (
            eu.compute_individual_token_rank(
                token_logits[sampid][layerid], predicted_token
            )
            + 1
        )
        second_acc_by_layer[sampid][layerid] = max(
            [
                eu.compute_individual_token_rank(token_logits[sampid][layerid], token)
                + 1
                for token in second_choice_tokens
            ]
        )
        corr_acc_by_layer[sampid][layerid] = (
            eu.compute_individual_token_rank(
                token_logits[sampid][layerid], correct_token
            )
            + 1
        )
##
## Get token probabilities across layers
##
pred_prob = torch.empty(probs_shape)
top_prob = torch.empty(probs_shape)
correct_prob = torch.empty(probs_shape)

second_prob = torch.empty(probs_shape)

for sampid in range(S):
    predicted_token = torch.argmax(token_probs[sampid][-1, :, :]).item()
    second_choice_tokens = [
        tokenizer.encode(token)[-1] for token in all_other_tokens[sampid]
    ]
    correct_token = target_token_ids[sampid][0]
    for layerid in range(L):
        pred_prob[sampid][layerid] = token_probs[sampid][layerid].squeeze()[
            predicted_token
        ]
        second_prob[sampid][layerid] = max(
            [
                token_probs[sampid][layerid].squeeze()[token]
                for token in second_choice_tokens
            ]
        )
        top_prob[sampid][layerid] = token_probs[sampid][layerid].max()
        correct_prob[sampid][layerid] = token_probs[sampid][layerid].squeeze()[
            correct_token
        ]

##
## Calculate norms over layers
##
layerwise_shape = token_logits[:, :-1].shape[:2]
layer_act_norm = {
    "h_mha": np.empty(layerwise_shape),
    "h_mid": np.empty(layerwise_shape),
    "h_mlp": np.empty(layerwise_shape),
    "h_l": np.empty(layerwise_shape),
}

for sampid in range(S):
    for layerid in range(L - 1):
        # Set hidden state at layer
        h_mha = mha_all_prompts[sampid][layerid].squeeze()
        h_mlp = mlp_all_prompts[sampid][layerid].squeeze()
        h_l = h_ls_all_prompts[sampid][layerid].squeeze()
        h_mid = h_l + h_mha
        # Calculate L2 norm
        layer_act_norm["h_mha"][sampid][layerid] = torch.norm(h_mha, p=2).item()
        layer_act_norm["h_mid"][sampid][layerid] = torch.norm(h_mid, p=2).item()
        layer_act_norm["h_mlp"][sampid][layerid] = torch.norm(h_mlp, p=2).item()
        layer_act_norm["h_l"][sampid][layerid] = torch.norm(h_l, p=2).item()

norms_over_layers = llh.di_to_df(layer_act_norm, "norm", S, L)

##
## Calculate cosine similarity between layers
##
layer_act_cossim = {
    "h_mha": np.empty(layerwise_shape),
    "h_l_h_mid": np.empty(layerwise_shape),
    "h_mha_h_mlp": np.empty(layerwise_shape),
    "h_mlp": np.empty(layerwise_shape),
    "h_l": np.empty(layerwise_shape),
}

for sampid in range(S):
    for layerid in range(L - 1):
        # Set hidden state at layer
        h_mha = mha_all_prompts[sampid][layerid].squeeze()
        h_mlp = mlp_all_prompts[sampid][layerid].squeeze()
        h_l = h_ls_all_prompts[sampid][layerid].squeeze()
        h_lb = h_ls_all_prompts[sampid][layerid + 1].squeeze()
        h_mid = h_l + h_mha
        # Calculate cosine similarity
        layer_act_cossim["h_mha"][sampid][layerid] = F.cosine_similarity(
            h_l, h_mha, dim=-1
        ).item()
        layer_act_cossim["h_l_h_mid"][sampid][layerid] = F.cosine_similarity(
            h_mid, h_l, dim=-1
        ).item()
        layer_act_cossim["h_mha_h_mlp"][sampid][layerid] = F.cosine_similarity(
            h_mha, h_mlp, dim=-1
        ).item()
        layer_act_cossim["h_mlp"][sampid][layerid] = F.cosine_similarity(
            h_mid, h_mlp, dim=-1
        ).item()
        layer_act_cossim["h_l"][sampid][layerid] = F.cosine_similarity(
            h_l, h_lb, dim=-1
        ).item()

cossim_over_layers = llh.di_to_df(layer_act_cossim, "cossim", S, L)

##
## Calculate apathy over layers
##
apathy = {
    "h_mha": (1 + layer_act_cossim["h_mha"])
    * (layer_act_norm["h_l"][:, :n_layers] - layer_act_norm["h_mha"][:, :n_layers]),
    "h_l_h_mid": (1 + layer_act_cossim["h_l_h_mid"])
    * (layer_act_norm["h_l"] - layer_act_norm["h_mid"]),
    "h_mha_h_mlp": (1 + layer_act_cossim["h_mha_h_mlp"])
    * (layer_act_norm["h_mha"] - layer_act_norm["h_mlp"]),
    "h_mlp": (1 + layer_act_cossim["h_mlp"])
    * (layer_act_norm["h_mid"][:, :n_layers] - layer_act_norm["h_mlp"][:, :n_layers]),
}
apathy_over_layers = llh.di_to_df(apathy, "apathy", S, L)

##
## Save data
##
out_fp = f"./results/token_probs/{model_name}/{data_name}"
os.makedirs(out_fp, exist_ok=True)

token_probabilities = [
    (pred_prob, "pred_prob"),
    (second_prob, "second_prob"),
    (top_prob, "top_prob"),
    (correct_prob, "correct_prob"),
]
all_probs = [
    llh.melt_data(data, metric_name) for (data, metric_name) in token_probabilities
]
all_probs = reduce(lambda x, y: pd.merge(x, y, on=["sampid", "layerid"]), all_probs)
all_probs.to_csv(os.path.join(out_fp, "token_probabilities.csv"), index=False)

token_ranks = [
    (pred_acc_by_layer, "pred_rank"),
    (second_acc_by_layer, "second_rank"),
    (corr_acc_by_layer, "correct_rank"),
]
all_ranks = [llh.melt_data(data, metric_name) for (data, metric_name) in token_ranks]
all_ranks = reduce(lambda x, y: pd.merge(x, y, on=["sampid", "layerid"]), all_ranks)
all_ranks.to_csv(os.path.join(out_fp, "token_rank_by_layer.csv"), index=False)

layer_analyses = [
    norms_over_layers,
    cossim_over_layers,
    apathy_over_layers,
]
layer_analyses = reduce(
    lambda x, y: pd.merge(x, y, on=["sampid", "layerid"]), layer_analyses
)
layer_analyses.to_csv(os.path.join(out_fp, "layer_analyses.csv"), index=False)
