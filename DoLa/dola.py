import argparse
import time
import csv
import tqdm
import os
import json

import torch
import torch.nn.functional as F
from torch.nn.utils.parametrize import remove_parametrizations
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

import argparse
import warnings
import pandas as pd
import numpy as np


class DoLa:
    def __init__(self, model_name, model_fp, device, num_gpus, max_gpu_memory=27):
        self.model_name = model_name
        self.model_fp = model_fp
        self.device = device
        self.num_gpus = num_gpus
        self.stopping_criteria = None
        self.max_gpu_memory = max_gpu_memory

        self.model, self.tokenizer = self.load_model(model_name, model_fp)

    def load_model(self, model_name, model_fp):
        tokenizer = AutoTokenizer.from_pretrained(model_fp)
        max_memory = {
            0: "0GiB",
            1: "78GiB",
            2: "78GiB",
        }
        # For large models
        if float(model_name.split("-")[1][:-1]) > 27:
            model = AutoModelForCausalLM.from_pretrained(
                model_fp,
                trust_remote_code=False,
                device_map="balanced",
                max_memory=max_memory,
                torch_dtype=torch.bfloat16,
            )
        # For small models
        else:
            if float(model_name.split("-")[1][:-1]) >= 13:
                torch_dtype = torch.bfloat16
            else:
                torch_dtype = "float16"
            model = AutoModelForCausalLM.from_pretrained(
                model_fp,
                device_map="auto",
                torch_dtype=torch_dtype,
                trust_remote_code=False,
            )
        return model, tokenizer

    # def set_stop_words(self, stop_words):
    #     self.stop_words = stop_words
    #     self.stopping_criteria = StoppingCriteriaList()
    #     list_stop_word_ids = []
    #     for stop_word in self.stop_words:
    #         stop_word_ids = self.tokenizer.encode("\n" + stop_word)[3:]
    #         list_stop_word_ids.append(stop_word_ids)
    #         print(
    #             "Added stop word: ",
    #             stop_word,
    #             "with the ids",
    #             stop_word_ids,
    #             flush=True,
    #         )
    #     self.stopping_criteria.append(LLamaQaStoppingCriteria(list_stop_word_ids))

    # def generate(
    #     self,
    #     input_text,
    #     max_new_tokens=256,
    #     top_p=0.95,
    #     top_k=0,
    #     temperature=0.8,
    #     mature_layer=None,
    #     premature_layer=None,
    #     candidate_premature_layers=[],
    #     mode="baseline",
    #     verbose=True,
    #     remove_stop_words=False,
    #     relative_top=0.1,
    #     **kwargs,
    # ):
    #     with torch.no_grad():

    #         input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(
    #             self.device
    #         )
    #         max_len = input_ids.shape[-1] + max_new_tokens

    #         if mode == "baseline":
    #             outputs = self.model.generate(
    #                 input_ids,
    #                 max_length=max_len,
    #                 num_return_sequences=1,
    #                 output_scores=True,
    #                 return_dict_in_generate=True,
    #                 dola_decoding=False,
    #                 top_p=top_p,
    #                 top_k=top_k,
    #                 temperature=temperature,
    #                 stopping_criteria=self.stopping_criteria,
    #                 **kwargs,
    #             )
    #         elif mode == "dola-static":
    #             assert mature_layer is not None, "mature_layer must be specified"
    #             assert premature_layer is not None, "premature_layer must be specified"
    #             outputs = self.model.generate(
    #                 input_ids,
    #                 max_length=max_len,
    #                 num_return_sequences=1,
    #                 output_scores=True,
    #                 return_dict_in_generate=True,
    #                 dola_decoding=True,
    #                 mature_layer=mature_layer,
    #                 premature_layer=premature_layer,
    #                 top_p=top_p,
    #                 top_k=top_k,
    #                 temperature=temperature,
    #                 stopping_criteria=self.stopping_criteria,
    #                 relative_top=relative_top,
    #                 **kwargs,
    #             )
    #         elif mode == "dola":
    #             assert mature_layer is not None, "mature_layer must be specified"
    #             assert (
    #                 candidate_premature_layers is not None
    #             ), "candidate_premature_layers must be specified"
    #             outputs = self.model.generate(
    #                 input_ids,
    #                 max_length=max_len,
    #                 num_return_sequences=1,
    #                 output_scores=True,
    #                 return_dict_in_generate=True,
    #                 dola_decoding=True,
    #                 top_p=top_p,
    #                 top_k=top_k,
    #                 temperature=temperature,
    #                 stopping_criteria=self.stopping_criteria,
    #                 relative_top=relative_top,
    #                 mature_layer=mature_layer,
    #                 premature_layer=None,
    #                 candidate_premature_layers=candidate_premature_layers,
    #                 **kwargs,
    #             )
    #             premature_layer_dist = outputs.premature_layer_dist
    #         sequences, scores = outputs.sequences, outputs.scores

    #         # skip the tokens in the input prompt
    #         gen_sequences = sequences[:, input_ids.shape[-1] :][0, :]
    #         gen_arr = gen_sequences.cpu().numpy()

    #         output_str = self.tokenizer.decode(gen_sequences, skip_special_tokens=True)

    #         if verbose:
    #             print("MODEL OUTPUT: \n{0}".format(output_str))

    #         if remove_stop_words:
    #             for stop_word in self.stop_words:
    #                 length_to_remove = len(stop_word)
    #                 if output_str[-length_to_remove:] == stop_word:
    #                     output_str = output_str[:-length_to_remove]
    #             output_str = output_str.strip()

    #     if self.device:
    #         torch.cuda.empty_cache()

    #     return output_str, (premature_layer_dist if mode == "dola" else None)

    def get_relative_top_filter(
        self,
        scores: torch.FloatTensor,
        relative_top: float = 0.1,
        min_tokens_to_keep: int = 1,
    ):
        scores_normalized = scores.log_softmax(dim=-1)
        sorted_logits, sorted_indices = torch.sort(scores_normalized, descending=True)
        min_thresh = sorted_logits[..., min_tokens_to_keep - 1]
        probs_max = torch.max(scores_normalized, dim=-1).values
        probs_thresh = probs_max + np.log(relative_top)
        probs_thresh = torch.min(min_thresh, probs_thresh)
        probs_thresh = probs_thresh.unsqueeze(-1)
        return scores_normalized < probs_thresh

    def lm_score(
        self,
        input_text1,
        input_text2,
        pmi=False,
        max_new_tokens=256,
        top_p=0.95,
        top_k=0,
        temperature=0.8,
        mature_layer=None,
        premature_layer=None,
        candidate_premature_layers=[],
        mode="baseline",
        verbose=True,
        remove_stop_words=False,
        relative_top=0.1,
        relative_top_value=-1000.0,
        post_softmax=True,
        length_norm=False,
        len_bias_const=0,
        ln_type="next_layer",
        **kwargs,
    ):
        with torch.no_grad():
            input_text = input_text1 + input_text2
            input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(
                self.device
            )
            prefix_ids = self.tokenizer(input_text1, return_tensors="pt").input_ids.to(
                self.device
            )
            continue_ids = input_ids[0, prefix_ids.shape[-1] :]
            if mode == "baseline":
                outputs = self.model(input_ids)[0].squeeze(0)
                outputs = outputs.log_softmax(-1)  # logits to log probs

                if len_bias_const > 0:
                    max_logit = torch.min(outputs, dim=-1, keepdim=True).values
                    outputs -= max_logit + len_bias_const

                # skip tokens in the prompt -- we only care about the answer
                outputs = outputs[prefix_ids.shape[-1] - 1 : -1, :]

                # get logprobs for each token in the answer
                log_probs = outputs[range(outputs.shape[0]), continue_ids].sum().item()

            elif mode == "dola-static":
                dict_outputs, outputs = self.model(
                    input_ids=input_ids,
                    return_dict=True,
                    output_attentions=False,
                    output_hidden_states=False,
                    early_exit_layers=[premature_layer, mature_layer],
                )

                assert premature_layer is not None
                base_logits = dict_outputs[premature_layer][
                    0, prefix_ids.shape[-1] - 1 : -1, :
                ]
                final_logits = dict_outputs[mature_layer][
                    0, prefix_ids.shape[-1] - 1 : -1, :
                ]
                final_logits = final_logits.log_softmax(dim=-1)
                base_logits = base_logits.log_softmax(dim=-1)
                diff_logits = final_logits - base_logits
                if post_softmax:
                    diff_logits = diff_logits.log_softmax(dim=-1)
                if relative_top > 0.0:
                    relative_top_mask = self.get_relative_top_filter(
                        final_logits, relative_top
                    )
                    diff_logits = torch.where(
                        relative_top_mask, relative_top_value, diff_logits
                    )

                log_probs = (
                    diff_logits[range(diff_logits.shape[0]), continue_ids].sum().item()
                )

            elif mode == "dola":
                premature_layer_dist = {l: 0 for l in candidate_premature_layers}
                picked_logits = []
                result_dict = {}
                premature_layers = []

                outputs = self.model(
                    input_ids=input_ids,
                    output_attentions=False,
                    output_hidden_states=True,
                )
                
                lm_head = self.model.get_output_embeddings().weight
                lm_head = lm_head.to('cuda:0')

                ##
                ## No layernorm
                ##
                if ln_type == "none":
                    hidden_states = outputs.hidden_states
                ##
                ## Apply next layer's LN to all layers but final
                ##
                elif (
                    ln_type == "next_layer"
                ):  
                    if "pythia" in self.model_name.lower():
                        hidden_states = []
                        for i, layer in enumerate(outputs.hidden_states):
                            curr_hidden_state = layer
                            if i < len(self.model.gpt_neox.layers):
                                layer_i_plus_1 = self.model.gpt_neox.layers[i]
                                input_ln_i_plus_1 = layer_i_plus_1.input_layernorm
                                curr_hidden_state = input_ln_i_plus_1(curr_hidden_state)
                            else:
                                curr_hidden_state = (
                                    self.model.gpt_neox.final_layer_norm(
                                        curr_hidden_state
                                    )
                                )
                            hidden_states.append(curr_hidden_state)
                    else:
                        hidden_states = []
                        for i, layer in enumerate(outputs.hidden_states):
                            curr_hidden_state = layer
                            if i < len(self.model.model.layers):
                                layer_i_plus_1 = self.model.model.layers[i]
                                input_ln_i_plus_1 = layer_i_plus_1.input_layernorm
                                curr_hidden_state = input_ln_i_plus_1(curr_hidden_state)
                            else:
                                curr_hidden_state = self.model.model.norm(
                                    curr_hidden_state
                                )
                            hidden_states.append(curr_hidden_state)
                ##
                ## Apply last LN to all but final layer
                ##
                elif (
                    ln_type == "last_layer"
                ):
                    if "pythia" in self.model_name.lower():
                        last_layer_ln = self.model.gpt_neox.final_layer_norm
                    else:
                        last_layer_ln = self.model.model.norm
                    hidden_states = []
                    for i, h in enumerate(outputs.hidden_states):
                        if i == len(outputs.hidden_states) - 1:
                            hidden_states.append(h)
                        else:
                            hidden_states.append(last_layer_ln(h))
                            
                hidden_states = [h.to('cuda:0') for h in hidden_states]
                dict_outputs = [hidden_state @ lm_head.T for hidden_state in hidden_states]

                for seq_i in range(prefix_ids.shape[-1] - 1, input_ids.shape[-1] - 1):
                    # Pick the less like layer to contrast with
                    # 1. Stacking all premature_layers into a new dimension
                    stacked_premature_layers = torch.stack(
                        [
                            dict_outputs[i][:, seq_i, :]
                            for i in candidate_premature_layers
                        ],
                        dim=0,
                    )

                    # 2. Calculate the softmax values for mature_layer and all premature_layers
                    softmax_mature_layer = F.softmax(
                        dict_outputs[mature_layer][:, seq_i, :], dim=-1
                    )  # shape: (batch_size, num_features)
                    softmax_premature_layers = F.softmax(
                        stacked_premature_layers, dim=-1
                    )  # shape: (num_premature_layers, batch_size, num_features)

                    # 3. Calculate M, the average distribution
                    M = 0.5 * (
                        softmax_mature_layer[None, :, :] + softmax_premature_layers
                    )  # shape: (num_premature_layers, batch_size, num_features)

                    # 4. Calculate log-softmax for the KL divergence
                    log_softmax_mature_layer = F.log_softmax(
                        dict_outputs[mature_layer][:, seq_i, :], dim=-1
                    )  # shape: (batch_size, num_features)
                    log_softmax_premature_layers = F.log_softmax(
                        stacked_premature_layers, dim=-1
                    )  # shape: (num_premature_layers, batch_size, num_features)

                    # 5. Calculate the KL divergences and then the JS divergences
                    kl1 = F.kl_div(
                        log_softmax_mature_layer[None, :, :], M, reduction="none"
                    ).mean(
                        -1
                    )  # shape: (num_premature_layers, batch_size)
                    kl2 = F.kl_div(
                        log_softmax_premature_layers, M, reduction="none"
                    ).mean(
                        -1
                    )  # shape: (num_premature_layers, batch_size)
                    js_divs = 0.5 * (
                        kl1 + kl2
                    )  # shape: (num_premature_layers, batch_size)

                    # 6. Reduce the batchmean
                    js_divs = js_divs.mean(-1)  # shape: (num_premature_layers,)
                    premature_layer = candidate_premature_layers[
                        int(js_divs.argmax().cpu().item())
                    ]
                    premature_layer_dist[premature_layer] += 1

                    premature_layers.append(premature_layer)

                base_logits = torch.zeros_like(
                    dict_outputs[mature_layer][0, prefix_ids.shape[-1] - 1 : -1]
                )
                for i, l in enumerate(premature_layers):
                    base_logits[i] = dict_outputs[l][0, prefix_ids.shape[-1] - 1 + i]
                final_logits = dict_outputs[mature_layer][
                    0, prefix_ids.shape[-1] - 1 : -1
                ]
                final_logits = final_logits.log_softmax(dim=-1)
                base_logits = base_logits.log_softmax(dim=-1)
                diff_logits = final_logits - base_logits
                if post_softmax:
                    diff_logits = diff_logits.log_softmax(dim=-1)

                if relative_top > 0.0:
                    relative_top_mask = self.get_relative_top_filter(
                        final_logits, relative_top
                    )
                    diff_logits = torch.where(
                        relative_top_mask, relative_top_value, diff_logits
                    )

                if diff_logits.device != continue_ids.device:
                    continue_ids = continue_ids.to(diff_logits.device)
                log_probs = (
                    diff_logits[range(diff_logits.shape[0]), continue_ids].sum().item()
                )
        if length_norm:
            log_probs /= len(range(prefix_ids.shape[-1] - 1, input_ids.shape[-1] - 1))
        del hidden_states
        torch.cuda.empty_cache()
        return log_probs, (premature_layer_dist if mode == "dola" else None)
