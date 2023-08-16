import argparse
import os
import pandas as pd
import evaluate
import pickle
import torch
import warnings
from tqdm import tqdm

from peft import (
    PeftConfig,
    PeftModel,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
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

    experiment = args.experiment
    peft_model_id = f"experiments/{experiment}/assets"

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

    ctr = 0
    results = []
    instructions, labels = test_dataset["instructions"], test_dataset["labels"]

    for instruct, label in zip(instructions, labels):
        input_ids = tokenizer(
            instruct, return_tensors="pt", truncation=True
        ).input_ids.cuda()

        with torch.inference_mode():
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
            results.append(result)
            ctr += 1
            print(f"Example {ctr} / {len(instructions)}:")
            print(f"Label:{label}")
            print(f"Generated:{result}")
            print("----------------------------------------")

    metrics = {
        "micro_f1": f1_score(labels, results, average="micro"),
        "macro_f1": f1_score(labels, results, average="macro"),
        "precision": precision_score(labels, results, average="micro"),
        "recall": recall_score(labels, results, average="micro"),
        "accuracy": accuracy_score(labels, results),
    }
    print(metrics)

    save_dir = os.path.join(f"experiments/{args.experiment}", "metrics")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(os.path.join(save_dir, "metrics.pkl"), "wb") as handle:
        pickle.dump(metrics, handle)

    print(f"Completed experiment {peft_model_id}")
    print("----------------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="1-8-0.1")
    args = parser.parse_args()

    main(args)
