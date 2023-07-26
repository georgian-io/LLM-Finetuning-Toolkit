import argparse
import torch
import pickle
import evaluate
import numpy as np
import os

from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_from_disk, load_dataset
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from utils import clean_text

# Metric
metric = evaluate.load("rouge")


def evaluate_peft_model(model, tokenizer, prompt, max_target_length=50):
    # generate summary
    input_ids = tokenizer(
        "summarize: " + prompt, return_tensors="pt", truncation=True
    ).input_ids.cuda()

    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_ids,
            do_sample=True,
            top_p=0.9,
            max_new_tokens=max_target_length,
        )
        prediction = tokenizer.batch_decode(
            outputs.detach().cpu().numpy(), skip_special_tokens=True
        )[0]

    return prediction


def main(args):
    dataset_id = "samsum"
    dataset = load_dataset(dataset_id)
    test_dataset = dataset["test"]
    peft_model_id = os.path.join(args.adapter_type, args.experiment, "assets")

    config = PeftConfig.from_pretrained(peft_model_id)

    # load base LLM model and tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(
        config.base_model_name_or_path,
        # load_in_8bit=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

    # Load the Lora model
    model = PeftModel.from_pretrained(model, peft_model_id)
    model.cuda()
    model.eval()

    print("Peft model loaded")
    predictions, references = [], []
    for sample in tqdm(test_dataset):
        prediction = evaluate_peft_model(model, tokenizer, sample["dialogue"])
        predictions.append(prediction)
        summary = sample["summary"]
        references.append(summary)

    # compute metric
    rouge = metric.compute(
        predictions=predictions, references=references, use_stemmer=True
    )

    # print results
    print(f"rouge1: {rouge['rouge1']* 100:2f}%")
    print(f"rouge2: {rouge['rouge2']* 100:2f}%")
    print(f"rougeL: {rouge['rougeL']* 100:2f}%")
    print(f"rougeLsum: {rouge['rougeLsum']* 100:2f}%")

    metrics = {metric: round(rouge[metric] * 100, 2) for metric in rouge.keys()}

    metrics_dir = os.path.join(args.adapter_type, args.experiment, "metrics")
    if not os.path.exists(metrics_dir):
        os.makedirs(metrics_dir)
    with open(os.path.join(metrics_dir, "metric.pkl"), "wb") as handle:
        pickle.dump(metrics, handle)
    print(f"Inference over for {args.experiment}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_type", default="experiments")
    parser.add_argument("--experiment", default="lora_epochs-5_r-16_dropout-0.1")
    args = parser.parse_args()

    main(args)
