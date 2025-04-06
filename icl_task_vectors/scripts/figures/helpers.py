import os
import pickle
import pandas as pd


from scripts.utils import main_experiment_results_dir, overriding_experiment_results_dir

MODEL_DISPLAY_NAME_MAPPING = {
    "llama2_7b": "LLaMA2 7B",
    "llama3_8b": "LLaMA3.1 8B",
    "mistral1_7b": "Mistral1 7B",
    "mistral3_7b": "Mistral3 7B",
    # "gemma2_9b": "Gemma2 9B",
    "qwen2_7b": "Qwen2 7B",
    "gptj_6b": "GPT-J 6B",
    "olmo_7b": "OLMo 7B",
}


def load_main_results(experiment_id: str = "camera_ready"):
    results = {}
    experiment_dir = main_experiment_results_dir(experiment_id)

    for results_file in os.listdir(experiment_dir):
        model_name = results_file[:-4]
        file_path = os.path.join(experiment_dir, results_file)
        with open(file_path, "rb") as f:
            results[model_name] = pickle.load(f)

    return results


def load_overriding_results(experiment_id: str = "camera_ready"):
    results = {}
    overriding_results_dir = overriding_experiment_results_dir(experiment_id)

    for results_file in os.listdir(overriding_results_dir):
        model_name = results_file[:-4]
        file_path = os.path.join(overriding_results_dir, results_file)
        with open(file_path, "rb") as f:
            results[model_name] = pickle.load(f)

    return results


def extract_accuracies(results):
    accuracies = {}
    for model_name, model_results in results.items():
        accuracies[model_name] = {}
        for task_name, task_results in model_results.items():
            accuracies[model_name][task_name] = {
                "bl": task_results["baseline_accuracy"],
                "icl": task_results["icl_accuracy"],
                "tv": task_results["tv_accuracy"],
            }

    return accuracies


def create_accuracies_df(results):
    accuracies = extract_accuracies(results)

    data = []
    for model_name, model_acc in accuracies.items():
        for task_full_name, task_acc in model_acc.items():
            task_type = task_full_name.split("_")[0]
            task_name = "_".join(task_full_name.split("_")[1:])

            data.append([model_name, task_type, task_name, "Baseline", task_acc["bl"]])
            data.append(
                [model_name, task_type, task_name, "Hypothesis", task_acc["tv"]]
            )
            data.append([model_name, task_type, task_name, "Regular", task_acc["icl"]])

    df = pd.DataFrame(
        data, columns=["model", "task_type", "task_name", "method", "accuracy"]
    )

    df["model"] = df["model"].map(MODEL_DISPLAY_NAME_MAPPING)

    # order the tasks by alphabetical order, using the task_full_name
    task_order = sorted(
        zip(df["task_type"].unique(), df["task_name"].unique()), key=lambda x: x[0]
    )
    task_order = [x[1] for x in task_order]

    # df["task_name"] = pd.Categorical(df["task_name"], categories=task_order, ordered=True)

    return df


def create_grouped_accuracies_df(accuracies_df):
    grouped_accuracies_df = accuracies_df.pivot_table(
        index=["model", "task_type", "task_name"],
        columns="method",
        values="accuracy",
        aggfunc="first",
    ).reset_index()
    return grouped_accuracies_df
