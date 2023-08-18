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

def main(args):

    peft_model_id = os.path.join(args.adapter_type, args.experiment, "assets")
    merged_model_id = os.path.join(args.adapter_type, args.experiment, "assets_merged")
    config = PeftConfig.from_pretrained(peft_model_id)

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, peft_model_id)

    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = model.merge_and_unload()
    model.save_pretrained(merged_model_id, push_to_hub=True, repo_id=args.repo_id)
    tokenizer.save_pretrained(merged_model_id, push_to_hub=True, repo_id=args.repo_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_type", default="experiments")
    parser.add_argument("--experiment", default="class_redp_7b")
    parser.add_argument("--repo_id")
    args = parser.parse_args()

    main(args)