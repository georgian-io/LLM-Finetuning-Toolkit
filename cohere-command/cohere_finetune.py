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

import cohere


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


def cohere_api_call(
    content: str,
    test_x: str,
    model_id,
):
    try:
        test_x = content + test_x

        if args.task_type == "summarization":
            response = co.generate(
	        model=model_id,
	        prompt=test_x,
            )
            generation = response.generations[0].text
    
        elif args.task_type == "classification":
            response = co.classify(
                model=model_id,
                inputs=[test_x],
            )
            generation = response.classifications[0].predictions[0]
        
    except:
        print(f"---------Bad prompt--------: {test_x}")
        generation = ""

    return generation


def write_jsonl(file_path, lines):
    """Create a .jsonl file and dump contents.
    file_path (unicode / Path): The path to the output file.
    lines (list): The JSON-serializable contents of each line.
    """
    data = [ujson.dumps(line, escape_forward_slashes=False) for line in lines]
    Path(file_path).open("w", encoding="utf-8").write("\n".join(data))


def prepare_data_in_cohere_format(task, x, y):
    if task == "summarization":
        FINETUNE_FORMAT = {
                "prompt": "Summarize the following conversation: " + x,
                "completion": y,
        }
    else:
        FINETUNE_FORMAT = {
                "text": "Classify the following data: " + x,
                "label": y,
        }

    return FINETUNE_FORMAT




def create_training_file(args):
    if args.task_type == "summarization":
        train_x, train_y, test_x, test_y = get_samsum_data_for_ft()
        save_dir = os.path.join("data", args.task_type)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        train_file_path = os.path.join(save_dir, "train_samsum.jsonl")
        test_file_path = os.path.join(save_dir, "test_samsum.jsonl")
    else:
        train_x, train_y, test_x, test_y = get_newsgroup_data_for_ft(args.train_sample_fraction)
        save_dir = os.path.join(
            "data", f"{args.task_type}_sample-fraction-{args.train_sample_fraction}"
        )
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        train_file_path = os.path.join(save_dir, "train_newsgroup.jsonl")
        test_file_path = os.path.join(save_dir, "test_newsgroup.jsonl")

    train_data, test_data = [], []
    for x, y in zip(train_x, train_y):
        example = prepare_data_in_cohere_format(args.task_type, x, y)
        train_data.append(example)
    for x, y in zip(test_x, test_y):
        example = prepare_data_in_cohere_format(args.task_type, x, y)
        test_data.append(example)

    write_jsonl(train_file_path, train_data)
    write_jsonl(test_file_path, test_data)
    print("JSON file written.....")


def infer_finetuned_model(args):
    if args.task_type == "summarization":
        _, _, test_x, test_y = get_samsum_data_for_ft()
        content = "Summarize the following conversation: "
    else:
        _, _, test_x, test_y = get_newsgroup_data_for_ft(args.train_sample_fraction)
        content = "Classify the following data: "

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

        result = cohere_api_call(
            content,
            x,
            args.model_id,
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
        default="create_data",
        choices=["create_data", "infer_finetuned_model"],
    )

    parser.add_argument("--api_key", type=str)
    parser.add_argument("--task_type", default="summarization", type=str)
    parser.add_argument("--train_sample_fraction", default=0.25, type=float)
    parser.add_argument("--model_id", type=str)

    args = parser.parse_args()

    global co
    co = cohere.Client(args.api_key)
 
    if args.job_type == "create_data":
        create_training_file(args)
    elif args.job_type == "infer_finetuned_model":
        infer_finetuned_model(args)




