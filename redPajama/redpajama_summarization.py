from transformers import (
    default_data_collator, get_linear_schedule_with_warmup,
    AutoModelForSeq2SeqLM, 
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer, 
    TrainingArguments, 
    AdamW, 
    get_linear_schedule_with_warmup,
    DataCollatorForLanguageModeling,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    )
from trl import SFTTrainer
from peft import (
    get_peft_config, 
    get_peft_model, 
    get_peft_model_state_dict, 
    TaskType, 
    LoraConfig,
    PrefixTuningConfig, 
    PromptTuningConfig, 
    PromptEncoderConfig,
    PromptTuningInit, 
)
import argparse
import functools
import torch
import datasets
from datasets import load_dataset
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import load_dataset, load_metric, Dataset, concatenate_datasets

from prompts import TRAINING_SUMMARIZATION_PROMPT_v2 


def prepare_instructions(dialogues, summaries):
    instructions = []

    prompt = TRAINING_SUMMARIZATION_PROMPT_v2

    for dialogue, summary in zip(dialogues, summaries):
        example = prompt.format(
            dialogue=dialogue,
            summary=summary,
        )
        instructions.append(example)

    return instructions

def prepare_samsum_data():
    dataset = load_dataset("samsum")
    train_dataset = dataset["train"]
    val_dataset = dataset["test"]

    dialogues = train_dataset["dialogue"]
    summaries = train_dataset["summary"]
    train_instructions = prepare_instructions(dialogues, summaries)
    train_dataset = datasets.Dataset.from_pandas(
        pd.DataFrame(data={"instructions": train_instructions})
    )

    dialgoues = val_dataset["dialogue"]
    summaries = val_dataset["summary"]
    val_instructions = prepare_instructions(dialogues, summaries)
    val_dataset = datasets.Dataset.from_pandas(
        pd.DataFrame(data={"instructions": val_instructions})
    )

    return train_dataset, val_dataset


def main(args):

    train_dataset, val_dataset = prepare_samsum_data()

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_ckpt)
    tokenizer.pad_token = tokenizer.eos_token
    # data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    
    print("Getting PEFT method")
    
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        lora_alpha=32,
        r=args.lora_r,
        lora_dropout=args.dropout,
        target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
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
        device_map={"": 0}
    )
    model.config.use_cache = False

    results_dir = f"experiments/summarization_epochs-{args.epochs}_rank-{args.lora_r}_dropout-{args.dropout}"

    # Define training args
    training_args = TrainingArguments(
        #save_strategy="no",
        #evaluation_strategy="no",
        #logging_strategy="epoch",
        logging_steps=100,
        
        report_to="none",

        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=8,

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
        eval_dataset=val_dataset,
        max_seq_length=512, # https://github.com/lvwerra/trl/issues/362 weird
        dataset_text_field="instructions",
        # data_collator=data_collator,
        packing=True,
        # num_of_sequences=1,
        # compute_metrics=lambda pred: compute_metrics(pred, tokenizer),
    )

    trainer_stats = trainer.train()
    train_loss = trainer_stats.training_loss
    #eval_stats = trainer.evaluate()
    #eval_loss = eval_stats["eval_loss"]
    print(f"Training loss:{train_loss}") # | Val loss:{eval_loss}")

    peft_model_id = f"{results_dir}/assets"
    trainer.model.save_pretrained(peft_model_id)
    tokenizer.save_pretrained(peft_model_id)

    with open(f"{results_dir}/results.pkl", "wb") as handle:
        run_result = [
            args.epochs,
            args.lora_r,
            args.dropout,
            train_loss,
            # eval_loss
        ]
        pickle.dump(run_result, handle)
    print("Experiment over")
    

if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_ckpt", default="togethercomputer/RedPajama-INCITE-7B-Base")
    parser.add_argument("--lora_r", default=4, type=int)
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--max_steps", default=10, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)


    args = parser.parse_args()
    main(args)



