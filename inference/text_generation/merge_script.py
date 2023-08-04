import argparse
import torch
import pickle
import evaluate
import numpy as np
from pathlib import Path

from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, PushToHubCallback
from datasets import load_from_disk, load_dataset
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score


def main(args):

    peft_model_id = Path(args.adapter_type, args.experiment, "assets")
    merged_model_id = Path(args.adapter_type, args.experiment, "assets_merged")

    config = PeftConfig.from_pretrained(str(peft_model_id))
    
    model = AutoModelForSeq2SeqLM.from_pretrained(
        config.base_model_name_or_path, 
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model_name_or_path
    )
    model = PeftModel.from_pretrained(
        model,
        str(peft_model_id)
    )
    
    model = model.merge_and_unload()
    model.save_pretrained(str(merged_model_id), push_to_hub=True, repo_id=args.repo_id)
    tokenizer.save_pretrained(str(merged_model_id), push_to_hub=True, repo_id=args.repo_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_type", default="experiments")
    parser.add_argument("--experiment", default="lora_epochs-5_r-16_dropout-0.1")
    parser.add_argument("--repo_id")
    args = parser.parse_args()

    main(args)
