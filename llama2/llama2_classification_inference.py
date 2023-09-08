import argparse
import torch
import os
import pandas as pd
import evaluate
import pickle
import warnings
from tqdm import tqdm

from llama_patch import unplace_flash_attn_with_attn
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from prompts import get_newsgroup_data_for_ft

metric = evaluate.load("rouge")
warnings.filterwarnings("ignore")


def main(args):
    _, test_dataset = get_newsgroup_data_for_ft(mode="inference")

    experiment = args.experiment_dir
    peft_model_id = f"{experiment}/assets"

    # unpatch flash attention
    unplace_flash_attn_with_attn()

    # load base LLM model and tokenizer
    model = AutoPeftModelForCausalLM.from_pretrained(
        peft_model_id,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        load_in_4bit=True,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(peft_model_id)

    results = []
    oom_examples = []
    instructions, labels = test_dataset["instructions"], test_dataset["labels"]

    for instruct, label in tqdm(zip(instructions, labels)):
        input_ids = tokenizer(
            instruct, return_tensors="pt", truncation=True
        ).input_ids.cuda()

        with torch.inference_mode():
            try:
                outputs = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=20,
                    do_sample=True,
                    top_p=0.95,
                    temperature=1e-3,
                )
                result = tokenizer.batch_decode(
                    outputs.detach().cpu().numpy(), skip_special_tokens=True
                )[0]
                result = result[len(instruct) :]
            except:
                result = ""
                oom_examples.append(input_ids.shape[-1])

            results.append(result)

    metrics = {
        "micro_f1": f1_score(labels, results, average="micro"),
        "macro_f1": f1_score(labels, results, average="macro"),
        "precision": precision_score(labels, results, average="micro"),
        "recall": recall_score(labels, results, average="micro"),
        "accuracy": accuracy_score(labels, results),
        "oom_examples": oom_examples,
    }
    print(metrics)

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
        default="experiments/classification-sampleFraction-0.1_epochs-5_rank-8_dropout-0.1",
    )

    args = parser.parse_args()
    main(args)
