import argparse
import torch
import os
import pandas as pd
import evaluate
import datasets
from datasets import load_dataset
import pickle
import warnings
from pathlib import Path
from optimum.bettertransformer import BetterTransformer
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
)

from peft import (
    AutoPeftModelForCausalLM,
)

def main(args):

    directory = f"{args.model_path}_merged" 
    model_path_merged = Path(directory).mkdir(parents=True, exist_ok=True)
    
    if args.model_type == "seq2seq":
        config = PeftConfig.from_pretrained(str(args.model_path))
    
        model = AutoModelForSeq2SeqLM.from_pretrained(
            config.base_model_name_or_path, 
        )
        tokenizer = AutoTokenizer.from_pretrained(
            config.base_model_name_or_path
        )
    else:
        model = AutoPeftModelForCausalLM.from_pretrained(
            args.model_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = model.merge_and_unload()
    model.save_pretrained(model_path_merged, push_to_hub=True, repo_id=args.repo_id)
    tokenizer.save_pretrained(model_path_merged, push_to_hub=True, repo_id=args.repo_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path")
    parser.add_argument("--model_type")
    parser.add_argument("--repo_id")
    args = parser.parse_args()

    main(args)