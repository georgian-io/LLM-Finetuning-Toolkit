import argparse
import torch
import os
import pandas as pd
import evaluate
from datasets import load_dataset
import pickle
import warnings

from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

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

    return val_instructions[:10], summaries[:10]


def main(args):
    val_instructions, summaries = prepare_samsum_data()

    experiment = args.experiment_dir
    peft_model_id = f"{experiment}/assets"

    # load base LLM model and tokenizer
    model = AutoPeftModelForCausalLM.from_pretrained(
        peft_model_id,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        load_in_4bit=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(peft_model_id)

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

    save_dir = os.path.join(experiment, "metrics")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(os.path.join(save_dir, "metrics.pkl"), "wb") as handle:
        pickle.dump(metrics, handle)

    print(f"Completed experiment {peft_model_id}")
    print("----------------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_dir",
        default="experiments/summarization_epochs-1_rank-64_dropout-0.1",
    )

    args = parser.parse_args()
    main(args)
