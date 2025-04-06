import json
import string
import random
import itertools

import torch
import numpy as np

import pandas as pd


def format_truthfulqa(n_samples, n_shots):
    tqa_fp = "./data/mc_task.json"
    tqa = json.load(open(tqa_fp))
    tqa_qs, few_shot, correct_tokens, all_other_tokens = [], [], [], []
    for i, sample in enumerate(tqa[0:n_samples]):
        other_tokens = []
        caps_alphabet = string.ascii_uppercase
        letters = itertools.cycle(caps_alphabet)
        prompt = f"Question: {sample['question']}\n"
        correct_letter = "N/A"
        while sample["mc1_targets"]:
            answer = random.choice(list(sample["mc1_targets"].keys()))
            correctness = sample["mc1_targets"].pop(answer)
            current_letter = f" {next(letters)}"
            prompt += f"{current_letter}: {answer}\n"
            if correctness == 1:
                correct_letter = current_letter
                correct_tokens.append(correct_letter)
            else:
                other_tokens.append(current_letter)
        prompt += "Answer:"
        if i < n_shots:
            few_shot.append(prompt + correct_letter)
        else:
            tqa_qs.append(prompt)
        all_other_tokens.append(other_tokens)
    few_shot_examples = "\n\n".join(few_shot)
    return tqa_qs, few_shot_examples, correct_tokens, all_other_tokens


def topk_logits(logits, k):
    topk_vals, topk_inds = torch.topk(logits.view(-1), k, largest=True)
    return topk_inds


def logit_lens(model, h_ls_all_prompts, model_name):
    # [n_samples, n_layer, 1, vocab_size])
    if "pythia" in model_name:
        return model.embed_out(h_ls_all_prompts)
    else:
        return model.lm_head(h_ls_all_prompts)


def melt_data(data, metric_name):
    df = pd.DataFrame(data)
    df = df.reset_index(names="sampid")
    return pd.melt(df, id_vars=["sampid"], var_name="layerid", value_name=metric_name)


def di_to_df(metric, metric_name, S, L):
    li = []
    for sampid in range(S):
        for layerid in range(L - 1):
            for hidden_name in metric.keys():
                li.append(
                    [sampid, layerid, hidden_name, metric[hidden_name][sampid][layerid]]
                )
    return pd.DataFrame(li, columns=["sampid", "layerid", "hidden_name", metric_name])


def minmax(h):
    return (h - np.min(h)) / (np.max(h) - np.min(h))
