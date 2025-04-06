import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import subprocess
import random


def load_gpt_model_and_tokenizer(model_fp: str, model_name: str, device="cuda"):
    """
    Loads a huggingface model and its tokenizer

    Parameters:
    model_fp = file path to downloaded model
    model_name: huggingface name of the model to load
    device: 'cuda' or 'cpu'

    Returns:
    model: huggingface model
    tokenizer: huggingface tokenizer
    MODEL_CONFIG: config variables w/ standardized names

    """
    assert model_fp is not None

    print("Loading: ", model_fp)

    if "gptj" in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_fp)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_fp,
            torch_dtype=torch.float16,
            device_map="cuda",
            attn_implementation="eager",
        )
        MODEL_CONFIG = {
            "n_heads": model.config.n_head,
            "n_layers": model.config.n_layer,
            "resid_dim": model.config.n_embd,
            "name_or_path": model.config.name_or_path,
            "name": model_name,
            "attn_hook_names": [
                f"transformer.h.{layer}.attn.out_proj"
                for layer in range(model.config.n_layer)
            ],
            "layer_hook_names": [
                f"transformer.h.{layer}" for layer in range(model.config.n_layer)
            ],
            "prepend_bos": False,
            "attn_dim": 4096,
        }
    elif "pythia" in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_fp)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_fp,
            torch_dtype=torch.float16,
            device_map="cuda",
            attn_implementation="eager",
        )
        MODEL_CONFIG = {
            "n_heads": model.config.num_attention_heads,
            "n_layers": model.config.num_hidden_layers,
            "resid_dim": model.config.hidden_size,
            "name_or_path": model.config._name_or_path,
            "name": model_name,
            "attn_hook_names": [
                f"gpt_neox.layers.{layer}.attention.dense"
                for layer in range(model.config.num_hidden_layers)
            ],
            "layer_hook_names": [
                f"gpt_neox.layers.{layer}"
                for layer in range(model.config.num_hidden_layers)
            ],
            "prepend_bos": False,
            "attn_dim": 4096,
        }
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_fp)
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
        max_memory = {
            0: "65GiB",
            1: "90GiB",
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
        print("Memory after model finished loading:")
        subprocess.run(["nvidia-smi"], shell=True)
        MODEL_CONFIG = {
            "n_heads": model.config.num_attention_heads,
            "n_layers": model.config.num_hidden_layers,
            "resid_dim": model.config.hidden_size,
            "name_or_path": model.config._name_or_path,
            "name": model_name,
            "attn_hook_names": [
                f"model.layers.{layer}.self_attn.o_proj"
                for layer in range(model.config.num_hidden_layers)
            ],
            "layer_hook_names": [
                f"model.layers.{layer}"
                for layer in range(model.config.num_hidden_layers)
            ],
            "prepend_bos": False,
            "attn_dim": model.model.layers[0].self_attn.q_proj.out_features,
        }

    return model, tokenizer, MODEL_CONFIG


def set_seed(seed: int) -> None:
    """
    Sets the seed to make everything deterministic, for reproducibility of experiments

    Parameters:
    seed: the number to set the seed to

    Return: None
    """

    # Random seed
    random.seed(seed)

    # Numpy seed
    np.random.seed(seed)

    # Torch seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # os seed
    os.environ["PYTHONHASHSEED"] = str(seed)
