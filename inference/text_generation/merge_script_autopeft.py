import argparse
import torch
import os
import pandas as pd
import evaluate
import datasets
from datasets import load_dataset
import pickle
import warnings
from optimum.bettertransformer import BetterTransformer

from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
)

from peft import (
    AutoPeftModelForCausalLM,
)

def main(args):

    peft_model_id = os.path.join(args.adapter_type, args.experiment, "assets")
    merged_model_id = os.path.join(args.adapter_type, args.experiment, "assets_merged")
  
    model = AutoPeftModelForCausalLM.from_pretrained(
        peft_model_id,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(peft_model_id)
    model = model.merge_and_unload()
    model.save_pretrained(merged_model_id, push_to_hub=True, repo_id=args.repo_id)
    tokenizer.save_pretrained(merged_model_id, push_to_hub=True, repo_id=args.repo_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_type", default="experiments")
    parser.add_argument("--experiment")
    parser.add_argument("--repo_id")
    args = parser.parse_args()

    main(args)