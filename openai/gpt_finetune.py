import openai

import argparse
import evaluate
import os
import ujson
from pathlib import Path
import pickle
import warnings

from prompts import (
    get_newsgroup_data_for_ft,
    get_samsum_data_for_ft,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

# Obtain from OpenAI's website
openai.organization = os.getenv("OPENAI_ORG_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")

metric = evaluate.load("rouge")
warnings.filterwarnings("ignore")


def compute_metrics_decoded(decoded_labs, decoded_preds, args):
    if args.task_type == "summarization":
        rouge = metric.compute(
            predictions=decoded_preds, references=decoded_labs, use_stemmer=True
        )
        metrics = {metric: round(rouge[metric] * 100.0, 3) for metric in rouge.keys()}

    elif args.task_type == "classification":
        metrics = {
            "micro_f1": f1_score(decoded_labs, decoded_preds, average="micro"),
            "macro_f1": f1_score(decoded_labs, decoded_preds, average="macro"),
            "precision": precision_score(decoded_labs, decoded_preds, average="micro"),
            "recall": recall_score(decoded_labs, decoded_preds, average="micro"),
            "accuracy": accuracy_score(decoded_labs, decoded_preds),
        }

    return metrics


def openai_api_call(
    content: str,
    test_x: str,
    custom_model,
    max_new_tokens: int,
    top_p: float = 0.95,
    temperature: float = 1e-3,
):
    try:
        response = openai.ChatCompletion.create(
            model=custom_model,
            messages=[
                {"role": "system", "content": content},
                {"role": "user", "content": test_x},
            ],
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        generation = response["choices"][0]["message"]["content"]
    except:
        generation = ""

    return generation


def write_jsonl(file_path, lines):
    """Create a .jsonl file and dump contents.
    file_path (unicode / Path): The path to the output file.
    lines (list): The JSON-serializable contents of each line.
    """
    data = [ujson.dumps(line, escape_forward_slashes=False) for line in lines]
    Path(file_path).open("w", encoding="utf-8").write("\n".join(data))


def prepare_data_in_openai_format(content, train_x, train_y):
    FINETUNE_FORMAT = {
        "messages": [
            {
                "role": "system",
                "content": content,
            },
            {"role": "user", "content": train_x},
            {"role": "assistant", "content": train_y},
        ]
    }

    return FINETUNE_FORMAT


def upload_training_file(args):
    if args.task_type == "summarization":
        train_x, train_y, _, _ = get_samsum_data_for_ft()
        content = "You are a helpful assistant who can summarize conversations."
        save_dir = os.path.join("data", args.task_type)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        file_path = os.path.join(save_dir, "train_samsum.jsonl")
    else:
        train_x, train_y, _, _ = get_newsgroup_data_for_ft(args.train_sample_fraction)
        content = "You are a helpful assistant who can classify newsletters into the right categories."
        save_dir = os.path.join(
            "data", f"{args.task_type}_sample-fraction-{args.train_sample_fraction}"
        )
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        file_path = os.path.join(save_dir, "train_newsgroup.jsonl")

    data = []
    for x, y in zip(train_x, train_y):
        example = prepare_data_in_openai_format(content, x, y)
        data.append(example)

    write_jsonl(file_path, data)
    print("JSON file written.....")

    openai.File.create(file=open(file_path, "rb"), purpose="fine-tune")
    print("File created.....")


def submit_finetuning_job(args):
    openai.FineTuningJob.create(
        training_file=args.training_file_id,
        model=args.model,
        hyperparameters={"n_epochs": args.epochs},
    )
    print("Finetuning job submitted")


def infer_finetuned_model(args):
    if args.task_type == "summarization":
        _, _, test_x, test_y = get_samsum_data_for_ft()
        content = "You are a helpful assistant who can summarize conversations."
    else:
        _, _, test_x, test_y = get_newsgroup_data_for_ft(args.train_sample_fraction)
        content = "You are a helpful assistant who can classify newsletters into the right categories."

    save_path = f"{args.task_type}_{args.model_id}_metrics.pkl"
    ctr = 0
    results, labels = [], []
    if os.path.exists(save_path):
        with open(save_path, "rb") as handle:
            metrics = pickle.load(handle)
            results = metrics["predictions"]
            labels = metrics["labels"]
            ctr = len(results)

    for i, (x, y) in enumerate(zip(test_x, test_y)):
        if i < ctr:
            continue
        if not isinstance(x, str):
            continue
        if not isinstance(y, str):
            continue

        result = openai_api_call(
            content,
            x,
            args.model_id,
            max_new_tokens=20 if args.task_type == "classification" else 50,
            top_p=0.95,
            temperature=1e-3,
        )

        results.append(result)
        labels.append(y)

        print(f"Example {ctr}/{len(test_x)} | GT: {y} | Pred: {result}")
        ctr += 1

        # Save every 100 iterations in case of network errors
        if ctr % 50 == 0:
            metrics = compute_metrics_decoded(labels, results, args)
            print(metrics)
            metrics["predictions"] = results
            metrics["labels"] = labels

            with open(save_path, "wb") as handle:
                pickle.dump(metrics, handle)

    metrics = compute_metrics_decoded(labels, results, args)
    print(metrics)
    metrics["predictions"] = results
    metrics["labels"] = labels
    with open(save_path, "wb") as handle:
        pickle.dump(metrics, handle)

    print(f"Completed inference {save_path}")
    print("----------------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--job_type",
        default="upload_data",
        choices=["upload_data", "submit_job", "infer_finetuned_model"],
    )

    parser.add_argument("--task_type", default="summarization", type=str)
    parser.add_argument("--train_sample_fraction", default=0.25, type=float)

    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--model", default="gpt-3.5-turbo")
    parser.add_argument("--training_file_id", default="file-abc123")

    parser.add_argument(
        "--model_id", default="ft:gpt-3.5-turbo-0613:georgian::87Dj76DB", type=str
    )

    args = parser.parse_args()

    if args.job_type == "upload_data":
        upload_training_file(args)
    elif args.job_type == "submit_job":
        submit_finetuning_job(args)
    elif args.job_type == "infer_finetuned_model":
        infer_finetuned_model(args)
