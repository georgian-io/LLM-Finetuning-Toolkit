import openai

import argparse
import evaluate
import os
import pandas as pd
import pickle
import numpy as np
import warnings
import requests
import time

from datasets import load_dataset
from prompts import (
    ZERO_SHOT_CLASSIFIER_PROMPT,
    FEW_SHOT_CLASSIFIER_PROMPT,
    ZERO_SHOT_SUMMARIZATION_PROMPT,
    FEW_SHOT_SUMMARIZATION_PROMPT,
    get_newsgroup_data,
    get_samsum_data,
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


def openai_api_call(
    prompt: str,
    model,
    custom_model,
    max_new_tokens: int,
    top_p: float = 0.95,
    temperature: float = 1e-3,
):
    try:
        if custom_model == "":
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        else:
            response = openai.ChatCompletion.create(
                model=custom_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )

        generation = response["choices"][0]["message"]["content"]

    except:
        generation = ""

    return generation


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


def main(args):
    if args.custom_model == "":
        save_dir = os.path.join(
            "baseline_results", args.model_type, args.task_type, args.prompt_type
        )
    else:
        save_dir = os.path.join(
            "baseline_results",
            "custom",
            args.custom_model,
            args.task_type,
            args.prompt_type,
        )
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if args.task_type == "classification":
        dataset = load_dataset("rungalileo/20_Newsgroups_Fixed")
        test_dataset = dataset["test"]
        test_data, test_labels = test_dataset["text"], test_dataset["label"]

        newsgroup_classes, few_shot_samples, _ = get_newsgroup_data()

    elif args.task_type == "summarization":
        dataset = load_dataset("samsum")
        test_dataset = dataset["test"]
        test_data, test_labels = test_dataset["dialogue"], test_dataset["summary"]

        few_shot_samples = get_samsum_data()

    if args.prompt_type == "zero-shot":
        if args.task_type == "classification":
            prompt = ZERO_SHOT_CLASSIFIER_PROMPT
        elif args.task_type == "summarization":
            prompt = ZERO_SHOT_SUMMARIZATION_PROMPT

    elif args.prompt_type == "few-shot":
        if args.task_type == "classification":
            prompt = FEW_SHOT_CLASSIFIER_PROMPT
        elif args.task_type == "summarization":
            prompt = FEW_SHOT_SUMMARIZATION_PROMPT

    elif args.prompt_type == "fine-tuned":
        prompt = ""

    times = []
    results = []
    good_data, good_labels = [], []
    ctr = 0

    # Load existing results
    if os.path.exists(os.path.join(save_dir, "metrics.pkl")):
        with open(os.path.join(save_dir, "metrics.pkl"), "rb") as handle:
            metrics = pickle.load(handle)
            results = metrics["predictions"]
            good_labels = metrics["labels"]
            good_data = metrics["data"]
            times = metrics["response_times"]
            ctr = len(results)

    # for instruct, label in zip(instructions, labels):
    for i, (data, label) in enumerate(zip(test_data, test_labels)):
        # skip already processed
        if i < ctr:
            continue
        if not isinstance(data, str):
            continue
        if not isinstance(label, str):
            continue

        if args.prompt_type == "zero-shot":
            if args.task_type == "classification":
                example = prompt.format(
                    newsgroup_classes=newsgroup_classes,
                    sentence=data,
                )
            elif args.task_type == "summarization":
                example = prompt.format(
                    dialogue=data,
                )

        elif args.prompt_type == "few-shot":
            if args.task_type == "classification":
                example = prompt.format(
                    newsgroup_classes=newsgroup_classes,
                    few_shot_samples=few_shot_samples,
                    sentence=data,
                )
            elif args.task_type == "summarization":
                example = prompt.format(
                    few_shot_samples=few_shot_samples,
                    dialogue=data,
                )

        elif args.prompt_type == "fine-tuned":
            example = data + "\n"

        start = time.time()
        result = openai_api_call(
            example,
            model=args.model_type,
            custom_model=args.custom_model,
            max_new_tokens=20 if args.task_type == "classification" else 50,
            top_p=0.95,
            temperature=1e-3,
        )
        diff = time.time() - start

        # Extract the generated text, and do basic processing
        result = result.replace("\n", "").lstrip().rstrip()
        results.append(result)
        good_labels.append(label)
        good_data.append(data)
        times.append(diff)

        print(f"Example {ctr}/{len(test_data)} | GT: {label} | Pred: {result}")
        ctr += 1

        # Save every 100 iterations in case of network errors
        if ctr % 50 == 0:
            metrics = compute_metrics_decoded(good_labels, results, args)
            print(metrics)
            metrics["predictions"] = results
            metrics["labels"] = good_labels
            metrics["data"] = good_data
            metrics["response_times"] = times

            with open(os.path.join(save_dir, "metrics.pkl"), "wb") as handle:
                pickle.dump(metrics, handle)

    metrics = compute_metrics_decoded(good_labels, results, args)
    print(metrics)
    metrics["predictions"] = results
    metrics["labels"] = good_labels
    metrics["data"] = good_data
    metrics["response_times"] = times

    with open(os.path.join(save_dir, "metrics.pkl"), "wb") as handle:
        pickle.dump(metrics, handle)

    print(f"Completed experiment {save_dir}")
    print("----------------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_type", default="zero-shot")
    parser.add_argument("--task_type", default="classification")
    parser.add_argument("--model_type", default="gpt-3.5-turbo")
    parser.add_argument("--custom_model", default="")
    args = parser.parse_args()

    main(args)
