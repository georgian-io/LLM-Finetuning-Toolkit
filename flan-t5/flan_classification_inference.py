import argparse
import torch
import pickle
import numpy as np
import os

from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_from_disk, load_dataset
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from utils import clean_text


def evaluate_peft_model(model, tokenizer, sample, max_target_length=20):
    # generate summary
    input_ids = tokenizer(
        "Classify the following sentence into a category: "
        + sample.replace("\n", " ")
        + " The answer is: ",
        return_tensors="pt",
        truncation=True,
    ).input_ids.cuda()

    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_ids,
            do_sample=True,
            top_p=0.9,
            max_new_tokens=max_target_length,
            temperature=1e-3,
        )
        prediction = tokenizer.batch_decode(
            outputs.detach().cpu().numpy(), skip_special_tokens=True
        )[0]

    return prediction


def main(args):
    # load test dataset from distk
    dataset_id = "rungalileo/20_Newsgroups_Fixed"
    # Load dataset from the hub
    dataset = load_dataset(dataset_id)
    test_df = clean_text(dataset["test"]["text"], dataset["test"]["label"])

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
    for idx, row in tqdm(test_df.iterrows()):
        sample = row["text"]
        label = row["label"]

        pred = evaluate_peft_model(model, tokenizer, sample)
        predictions.append(pred)
        references.append(label)

    metrics = {}
    metrics["accuracy"] = accuracy_score(references, predictions)
    metrics["f1-macro"] = f1_score(references, predictions, average="macro")
    metrics["f1-weighted"] = f1_score(references, predictions, average="weighted")
    print(metrics)

    metrics_dir = os.path.join(args.adapter_type, args.experiment, "metrics")
    if not os.path.exists(metrics_dir):
        os.makedirs(metrics_dir)
    with open(os.path.join(metrics_dir, "metric.pkl"), "wb") as handle:
        pickle.dump(metrics, handle)
    print(f"Inference over for {args.experiment}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_type", default="experiments")
    parser.add_argument(
        "--experiment", default="lora_samples-10557_epochs-5_r-16_dropout-0.1"
    )
    args = parser.parse_args()

    main(args)
