import os
import json

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


def save_json(model_name, data_name, data):
    """ """
    dp = f"./results/{model_name}/{data_name}"
    if not os.path.exists(dp):
        os.makedirs(dp)
    file_path = os.path.join(dp, "baseline_n_shots.json")
    try:
        with open(file_path, "w") as json_file:
            json.dump(data, json_file, indent=4)
        print(f"Data successfully saved to {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


def save_json2(model_name, data_name, title, collection, data):
    """ """
    dp = f"./results/{model_name}/{data_name}/{collection}"
    if not os.path.exists(dp):
        os.makedirs(dp)
    file_path = os.path.join(dp, f"{title.replace(' | ', '_')}.json")
    try:
        with open(file_path, "w") as json_file:
            json.dump(data, json_file, indent=4)
        print(f"Data successfully saved to {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


def read_json(model_name, data_name):
    """ """
    file_path = f"./results/{model_name}/{data_name}/baseline_n_shots.json"
    try:
        with open(file_path, "r") as json_file:
            data = json.load(json_file)
        return data
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def clean_numbers(n_heads, lda):
    if int(n_heads) < 10:
        n_heads = "00" + str(n_heads)
    elif int(n_heads) < 100:
        n_heads = "0" + str(n_heads)
    if int(lda) < 10:
        lda = "00" + str(lda)
    elif int(lda) < 100:
        lda = "0" + str(lda)
    return n_heads, lda


def plot_perf(
    model,
    model_name,
    data_name,
    n_heads,
    n_layers,
    lda,
    n_test,
    clean,
    interv,
    hv_norms_all,
    fv_norms_all,
):
    baseline_n_shots = read_json(model_name, data_name)
    # fig, axs = plt.subplots(1, 2, figsize=(12,4))
    one_shot = dict(
        zip([i for i in range(n_layers)], [baseline_n_shots["1"]] * n_layers)
    )
    three_shot = dict(
        zip([i for i in range(n_layers)], [baseline_n_shots["3"]] * n_layers)
    )
    ten_shot = dict(
        zip([i for i in range(n_layers)], [baseline_n_shots["10"]] * n_layers)
    )

    # sns.lineplot(pd.DataFrame(
    #     {'0 shot': clean,
    #      '1 shot': one_shot,
    #      '3 shot': three_shot,
    #      '10 shot': ten_shot,
    #      '0 shot w/ FV': interv}),
    # legend=True, ax=axs[0])

    # axs[0].tick_params('x', labelrotation=90)
    # axs[0].set_xlabel('Projection Layer')
    # axs[0].set_ylabel('Projected Probability')
    # axs[0].set_yticks(np.arange(0, 1.1, 0.1))
    # axs[0].set_title('Colors ICL Performance over Layers')

    # sns.lineplot(pd.DataFrame({
    #     'Avg Hidden Vector': hv_norms_all,
    #     'Function Vector': fv_norms_all}),
    # legend=True, ax=axs[1])
    # axs[1].tick_params('x', labelrotation=90)
    # axs[1].set_xlabel('Projection Layer')
    # axs[1].set_ylabel('L2 Norm')
    # max_y = max([max(hv_norms_all.values()),max(fv_norms_all.values())])
    # axs[1].set_yticks(np.linspace(0, max_y, 10))
    # axs[1].set_title('Norms over Layers')

    n_heads, lda = clean_numbers(n_heads, lda)

    title = f"{os.path.basename(model.config._name_or_path)} | {data_name} | {n_heads}-top-heads | {lda}-lambda | {n_test}-test-samples"
    # plt.suptitle(title)

    dp = f"./clean_figs/{model_name}/{data_name}"
    if not os.path.exists(dp):
        os.makedirs(dp)
    file_path = os.path.join(dp, f"sep_{title.replace(' | ', '_')}.png")

    # plt.savefig(file_path)


def plot_all(
    model,
    model_name,
    data_name,
    n_heads,
    n_layers,
    lda,
    n_test,
    clean,
    interv,
    hv1,
    hv2,
    hv_norms1_all,
    hv_norms2_all,
    hv_std_all,
    hv_ms_all,
    hv_spar_all,
    fv_norms_all,
):
    baseline_n_shots = read_json(model_name, data_name)

    def normalize_dis(di1, di2):
        di_min = min(min(di1.values()), min(di2.values()))
        di_max = max(max(di1.values()), max(di2.values()))
        minmax_di1 = {k: (v - di_min) / (di_max - di_min) for k, v in di1.items()}
        minmax_di2 = {k: (v - di_min) / (di_max - di_min) for k, v in di2.items()}
        return minmax_di1, minmax_di2

    # hv_norms_normalized = normalize_di(hv_norms_all)
    # fv_norms_normalized = normalize_di(fv_norms_all)

    hv_norms_minmax, fv_norms_minmax = normalize_dis(hv_norms2_all, fv_norms_all)

    # Create the figure
    # plt.figure(figsize=(6, 4))

    # First plot: Projected Probability
    one_shot = dict(
        zip([i for i in range(n_layers)], [baseline_n_shots["1"]] * n_layers)
    )
    three_shot = dict(
        zip([i for i in range(n_layers)], [baseline_n_shots["3"]] * n_layers)
    )
    ten_shot = dict(
        zip([i for i in range(n_layers)], [baseline_n_shots["10"]] * n_layers)
    )

    # Plot the probability data (first plot)
    sns.lineplot(
        pd.DataFrame(
            {
                "0 shot": clean,
                "1 shot": one_shot,
                #'3 shot': three_shot,
                "10 shot": ten_shot,
                "0 shot w/ FV": interv,
            }
        ),
        legend=True,
        dashes=[(1, 1), (1, 1), (1, 1), (2, 2)],
    )

    # Overlay the normalized second plot (L2 Norm) on the same y-axis
    sns.lineplot(
        pd.DataFrame(
            {"avg HV (mix-max)": hv_norms_minmax, "FV (mix-max)": fv_norms_minmax}
        ),
        legend=True,
        palette=["#E66100", "#5D3A9B"],
        dashes=[(), ()],
    )

    # Configure the x-axis
    # plt.tick_params('x', labelrotation=90)
    # plt.xlabel('Projection Layer')

    # Configure the y-axis to show the probabilities (0-1 range)
    # plt.ylabel('Projected Probability & L2 Norm')
    # plt.yticks(np.arange(0, 1.1, 0.1))

    # Set the title
    n_heads, lda = clean_numbers(n_heads, lda)
    title = f"{os.path.basename(model.config._name_or_path)} | {data_name} | {n_heads}-top-heads | {lda}-lambda | {n_test}-test-samples"
    # plt.title(title)

    # plt.legend(fontsize=8, loc='right', bbox_to_anchor=(1.33, 0.5))

    save_json2(model_name, data_name, title, "hv_norms2", hv_norms2_all)
    save_json2(model_name, data_name, title, "fv_norms", fv_norms_all)
    save_json2(model_name, data_name, title, "0_shot_w_FV", interv)

    dp = f"./clean_figs/{model_name}/{data_name}"
    if not os.path.exists(dp):
        os.makedirs(dp)
    file_path = os.path.join(dp, f"ov_{title.replace(' | ', '_')}.png")

    # plt.savefig(file_path, bbox_inches='tight')


def check_fin(model_name, data_name, model_fp, n_top_heads, lda, n_test):
    collection = "0_shot_w_FV"
    n_top_heads, lda = clean_numbers(n_top_heads, lda)
    saved_fname = f"{os.path.basename(model_fp)}_{data_name}_{n_top_heads}-top-heads_{lda}-lambda_{n_test}-test-samples"
    dp = f"./results/{model_name}/{data_name}/{collection}"
    file_path = os.path.join(dp, f"{saved_fname}.json")
    if os.path.isfile(file_path):
        return True
    else:
        return False
