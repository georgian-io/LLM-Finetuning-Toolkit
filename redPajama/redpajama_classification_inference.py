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
    instructions, labels = test_dataset["instructions"], test_dataset["labels"]

    # redpajama cannot handle input_ids that are greater than 2048 -- filter out those.
    good_labels = []

    for instruct, label in zip(instructions, labels):
        input_ids = tokenizer(
            instruct, return_tensors="pt", truncation=True
        ).input_ids.cuda()

        dimension = input_ids.shape[-1]
        if dimension >= 2048:
            continue

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

            except:
                print("Scalar type Half but found Float")

            result = result[len(instruct) :]
            results.append(result)
            good_labels.append(label)
            print(f"Instruction:{instruct}")
            print(f"Label:{label}")
            print(f"Generated:{result}")
            print("----------------------------------------")

    metrics = {
        "micro_f1": f1_score(good_labels, results, average="micro"),
        "macro_f1": f1_score(good_labels, results, average="macro"),
        "precision": precision_score(good_labels, results, average="micro"),
        "recall": recall_score(good_labels, results, average="micro"),
        "accuracy": accuracy_score(good_labels, results),
    }
    print(metrics)
    save_dir = os.path.join(f"{args.experiment_dir}", "metrics")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(os.path.join(save_dir, "metrics.pkl"), "wb") as handle:
        pickle.dump(metrics, handle)

    print(f"Completed experiment {peft_model_id}")
    print("----------------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_dir", default="experiments/classification_sampleFraction-0.01_epochs-1_rank-8_dropout-0.1")
    args = parser.parse_args()

    main(args)
