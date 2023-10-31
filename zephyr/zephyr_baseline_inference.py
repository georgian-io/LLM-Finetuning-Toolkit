import argparse
import os
import evaluate
import warnings
import json
import pandas as pd
import pickle
import torch
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
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

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


def main(args):
    save_dir = os.path.join(
        "baseline_results", args.pretrained_ckpt, args.task_type, args.prompt_type
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

    # BitsAndBytesConfig int-4 config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        args.pretrained_ckpt,
        quantization_config=bnb_config,
        use_cache=False,
        device_map="auto",
        use_flash_attention_2=args.use_flash_attention,
    )
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_ckpt)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    results = []
    good_data, good_labels = [], []
    ctr = 0
    # for instruct, label in zip(instructions, labels):
    for data, label in zip(test_data, test_labels):
        if not isinstance(data, str):
            continue
        if not isinstance(label, str):
            continue

        # example = instruct[:-len(label)] # remove the answer from the example
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

        input_ids = tokenizer(
            example, return_tensors="pt", truncation=True
        ).input_ids.cuda()

        with torch.inference_mode():
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=20 if args.task_type == "classification" else 50,
                do_sample=True,
                top_p=0.95,
                temperature=1e-3,
            )
            result = tokenizer.batch_decode(
                outputs.detach().cpu().numpy(), skip_special_tokens=True
            )[0]

            # Extract the generated text, and do basic processing
            result = result[len(example) :].replace("\n", "").lstrip().rstrip()
            results.append(result)
            good_labels.append(label)
            good_data.append(data)

        print(f"Example {ctr}/{len(test_data)} | GT: {label} | Pred: {result}")
        ctr += 1

    metrics = compute_metrics_decoded(good_labels, results, args)
    print(metrics)
    metrics["predictions"] = results
    metrics["labels"] = good_labels
    metrics["data"] = good_data

    with open(os.path.join(save_dir, "metrics.pkl"), "wb") as handle:
        pickle.dump(metrics, handle)

    print(f"Completed experiment {save_dir}")
    print("----------------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_ckpt", default="HuggingFaceH4/zephyr-7b-beta")
    parser.add_argument("--prompt_type", default="zero-shot")
    parser.add_argument("--task_type", default="classification")
    parser.add_argument("--use_flash_attention", action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    main(args)
