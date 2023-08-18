import argparse
import torch
import os
import pandas as pd
import evaluate
import datasets
from datasets import load_dataset
import pickle
import warnings

from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoTokenizer,
)

from peft import (
    PeftConfig,
    PeftModel,
)

from prompts import INFERENCE_SUMMARIZATION_PROMPT_v2

metric = evaluate.load("rouge")
warnings.filterwarnings("ignore")


def prepare_instructions(dialogues, summaries):
    instructions = []

    prompt = INFERENCE_SUMMARIZATION_PROMPT_v2

    for dialogue, summary in zip(dialogues, summaries):
        example = prompt.format(
            dialogue=dialogue,
        )
        instructions.append(example)

    return instructions


def prepare_samsum_data():
    dataset = load_dataset("samsum")
    val_dataset = dataset["test"]

    dialogues = val_dataset["dialogue"]
    summaries = val_dataset["summary"]
    val_instructions = prepare_instructions(dialogues, summaries)

    return val_instructions, summaries


def compute_metrics_decoded(decoded_labs, decoded_preds):
    def cleanup(x):
        if ":" in x:
            ind = x.find(":")
            return x[:ind]
        else:
            return x

    decoded_preds = [cleanup(d) for d in decoded_preds]

    metrics = {
        "micro_f1": f1_score(decoded_labs, decoded_preds, average="micro"),
        "macro_f1": f1_score(decoded_labs, decoded_preds, average="macro"),
        "precision": precision_score(decoded_labs, decoded_preds, average="micro"),
        "recall": recall_score(decoded_labs, decoded_preds, average="micro"),
        "accuracy": accuracy_score(decoded_labs, decoded_preds),
    }

    joint = [(decoded_preds[i], decoded_labs[i]) for i in range(len(decoded_preds))]
    mismatches = []
    for i, el in enumerate(joint):
        if el[0] != el[1]:
            mismatches.append([i, el])
    
    return metrics


def main(args):
    val_instructions, summaries = prepare_samsum_data()

    peft_model_id = f"{args.experiment_dir}/assets"

    config = PeftConfig.from_pretrained(peft_model_id)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        quantization_config=bnb_config,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, peft_model_id)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    results = []
    for instruct, summary in zip(val_instructions, summaries):
        input_ids = tokenizer(
            instruct, return_tensors="pt", truncation=True
        ).input_ids.cuda()
        with torch.inference_mode():
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=100,
                do_sample=True,
                top_p=0.9,
                temperature=1e-2,
            )
            result = tokenizer.batch_decode(
                outputs.detach().cpu().numpy(), skip_special_tokens=True
            )[0]
            result = result[len(instruct) :]
            results.append(result)
            print(f"Instruction:{instruct}")
            print(f"Summary:{summary}")
            print(f"Generated:{result}")
            print("----------------------------------------")

    # compute metric
    rouge = metric.compute(predictions=results, references=summaries, use_stemmer=True)

    metrics = {metric: round(rouge[metric] * 100, 2) for metric in rouge.keys()}

    save_dir = os.path.join(f"{args.experiment_dir}", "metrics")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(os.path.join(save_dir, "metrics.pkl"), "wb") as handle:
        pickle.dump(metrics, handle)

    print(f"Completed experiment {peft_model_id}")
    print("----------------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_dir", default="experiments/summarization_epochs-1_rank-4_dropout-0.1")
    args = parser.parse_args()

    main(args)
