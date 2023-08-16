import argparse
import torch
import os
import numpy as np
import pandas as pd
import pickle

from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

from prompts import get_newsgroup_data_for_ft


def main(args):
    train_dataset, test_dataset = get_newsgroup_data_for_ft(
        mode="train", train_sample_fraction=args.train_sample_fraction
    )
    print(f"Sample fraction:{args.train_sample_fraction}")
    print(f"Training samples:{train_dataset.shape}")

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_ckpt)
    tokenizer.pad_token = tokenizer.eos_token

    print("Getting PEFT method")

    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        lora_alpha=32,
        r=args.lora_r,
        lora_dropout=args.dropout,
        target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model_name_or_path = args.pretrained_ckpt
    model = AutoModelForCausalLM.from_pretrained(
        args.pretrained_ckpt,
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map={"": 0},
    )
    model.config.use_cache = False

    results_dir = f"experiments/classification-sampleFraction-{args.train_sample_fraction}_epochs-{args.epochs}_rank-{args.lora_r}_dropout-{args.dropout}"

    # Define training args
    training_args = TrainingArguments(
        logging_steps=100,
        report_to="none",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        output_dir=results_dir,
        learning_rate=2e-4,
        num_train_epochs=args.epochs,
        logging_dir=f"{results_dir}/logs",
        fp16=True,
        optim="paged_adamw_32bit",
        lr_scheduler_type="constant",
        max_grad_norm=0.3,
        warmup_ratio=0.03,
    )

    print(f"training_args = {training_args}")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        peft_config=peft_config,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        max_seq_length=512,
        dataset_text_field="instructions",
        packing=True,
    )

    trainer_stats = trainer.train()
    train_loss = trainer_stats.training_loss
    print(f"Training loss:{train_loss}")

    peft_model_id = f"{results_dir}/assets"
    trainer.model.save_pretrained(peft_model_id)
    tokenizer.save_pretrained(peft_model_id)

    with open(f"{results_dir}/results.pkl", "wb") as handle:
        run_result = [
            args.epochs,
            args.lora_r,
            args.dropout,
            train_loss,
        ]
        pickle.dump(run_result, handle)
    print("Experiment over")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_ckpt", default="tiiuae/falcon-7b")
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--max_steps", default=10, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--train_sample_fraction", default=0.99, type=float)

    args = parser.parse_args()
    main(args)
