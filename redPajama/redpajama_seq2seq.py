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


def get_data():
    df_train = pd.read_csv("sh_traintopk_10052023.csv")
    df_train = df_train[["User utterances", "Gold standard"]]
    df_train = df_train.fillna("")
    df_train = df_train.rename(columns={"User utterances": "input_text", "Gold standard": "output_text"})
    df_train = df_train[["input_text", "output_text"]]

    df_val = pd.read_csv("sh_testtopk_10052023.csv")
    df_val = df_val[["User utterances", "Gold standard"]]
    df_val = df_val.fillna("")
    df_val = df_val.rename(columns={"User utterances": "input_text", "Gold standard": "output_text"})
    df_val = df_val[["input_text", "output_text"]]

    dataset_train = Dataset.from_pandas(df_train)
    dataset_val = Dataset.from_pandas(df_val)

    return dataset_train, dataset_val


def preprocess_function(
    sample: tuple[list[str], list[str]], tokenizer: AutoTokenizer, padding="max_length"
):
    """Preproceses a sample using the tokenizer

    Args:
        sample: the input text with the corresponding label
        tokenizer: a T5Tokenizer
        padding: the type of padding

    Returns:
        a dictionary of the tokenized input
    """
    inputs = []
    target_labels = []

    sample_helper = getattr(
        __import__("ttec_train_helper", fromlist=["sample_helper_topk"]),
        "sample_helper_topk",
    )

    # config.training_fields = USer utterances, Gold standard
    # inputs, target_labels = sample_helper(sample, config.training_fields)
    inputs, target_labels = sample_helper(
            sample, ["input_text", "output_text"]
        )

    model_inputs = tokenizer(
        inputs, max_length=1536, padding=padding, truncation=True
    )

    labels = tokenizer(
        text_target=target_labels,
        max_length=30,
        padding=padding,
        truncation=True,
    )

    # If we are padding here, replace all tokenizer.pad_token_id in the
    # labels by -100 when we want to ignore padding in the loss.
    if padding == "max_length":
        labels["input_ids"] = [
            [(lab if lab != tokenizer.pad_token_id else -100) for lab in label]
            for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    # model_inputs["alternatives"] = alternatives
    model_inputs["utterance"] = inputs

    # manually creating the instruction here for instruction tuning
    instruction_prompts = []
    lengths = []
    max_len = 0
    for question, answer in zip(inputs, target_labels):
        prompt = f"{question}\n The answer is: {answer}"
        instruction_prompts.append(prompt)
        max_len = max(max_len, len(prompt))
        lengths.append(len(prompt))
    model_inputs["lengths"] = lengths

    """
    padded_instruction_prompts = []
    for prompt in instruction_prompts:
        extra_len = max_len - len(prompt)
        extra_chars = tokenizer.pad_token * (extra_len // len(tokenizer.pad_token))
        extra_chars = extra_chars + tokenizer.pad_token[: (extra_len - len(extra_chars))]
        prompt = prompt + extra_chars
        padded_instruction_prompts.append(prompt)
    """

    model_inputs["instructions"] = instruction_prompts 
    # model_inputs["padded_instructions"] = padded_instruction_prompts

    return model_inputs

"""
def compute_metrics(pred, tokenizer):
    import numpy as np
    preds = pred.predictions
    labels = pred.label_ids
    gt = tokenizer.batch_decode(labels, skip_special_tokens=True)
    import ipdb; ipdb.set_trace()
    a = 1
    return a
"""

def main(args):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    device = "cuda"
    
    

    # loading dataset
    #dataset_train, dataset_val = cecilia_data()
    #tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    #tokenizer.pad_token = tokenizer.eos_token
    dataset_train, dataset_val = get_data()
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_ckpt)
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    
    tokenized_dataset_train = dataset_train.map(
        preprocess_function,
        batched=True,
        # remove_columns=config.training_fields + config.other_fields,
        remove_columns=["input_text", "output_text"],
        fn_kwargs={"tokenizer": tokenizer},
    )
    
    tokenized_dataset_val = dataset_val.map(
        preprocess_function,
        batched=True,
        # remove_columns=config.training_fields + config.other_fields,
        remove_columns=["input_text", "output_text"],
        fn_kwargs={"tokenizer": tokenizer},
    )
    
    print("Getting PEFT method")
    if args.peft_method == "lora":
        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            lora_alpha=32,
            # inference_mode=False,
            r=args.lora_r,
            lora_dropout=args.dropout,
            target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
        )
    elif args.peft_method == "prefix":
        peft_config = PrefixTuningConfig(
                task_type=TaskType.SEQ_2_SEQ_LM, 
                inference_mode=False, 
                num_virtual_tokens=args.prefix_tokens, 
                prefix_projection=True if args.prefix_projection else False,
        )
    

    model_name_or_path = args.pretrained_ckpt
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
   
    model = AutoModelForCausalLM.from_pretrained(
        args.pretrained_ckpt, 
        quantization_config=bnb_config, 
        trust_remote_code=True
    )
    model.config.use_cache = False

    results_dir = f"experiments/{args.peft_method}-{args.epochs}-{args.lora_r}-{args.dropout}"

    # Define training args
    training_args = TrainingArguments(
        save_strategy="no",
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        logging_steps=5,
        # save_steps=10,
        # eval_steps=10,
        report_to="none",

        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=8,

        output_dir=results_dir,
        learning_rate=2e-4,
        # max_steps=args.max_steps,
        num_train_epochs=args.epochs,
        logging_dir=f"{results_dir}/logs",

        fp16=True,
        optim="paged_adamw_32bit",
        lr_scheduler_type="constant",
        max_grad_norm=0.3,
        warmup_ratio=0.5,
    )
    
    print(f"training_args = {training_args}")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        peft_config=peft_config,
        tokenizer=tokenizer,
        train_dataset=tokenized_dataset_train,
        eval_dataset=tokenized_dataset_val,
        max_seq_length=300, # https://github.com/lvwerra/trl/issues/362 weird
        dataset_text_field="instructions",
        # data_collator=data_collator,
        packing=True,
        # num_of_sequences=1,
        # compute_metrics=lambda pred: compute_metrics(pred, tokenizer),
    )

    trainer_stats = trainer.train()
    train_loss = trainer_stats.training_loss
    eval_stats = trainer.evaluate()
    eval_loss = eval_stats["eval_loss"]
    print(f"Training loss:{train_loss} | Val loss:{eval_loss}")

    peft_model_id = f"{results_dir}/assets"
    trainer.model.save_pretrained(peft_model_id)
    tokenizer.save_pretrained(peft_model_id)

    with open(f"{results_dir}/results.pkl", "wb") as handle:
        run_result = [
            args.epochs,
            args.lora_r,
            args.dropout,
            train_loss,
            eval_loss
        ]
        pickle.dump(run_result, handle)
    print("Experiment over")
    

if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_ckpt", default="togethercomputer/RedPajama-INCITE-Base-3B-v1")
    parser.add_argument("--peft_method", default="lora")
    parser.add_argument("--lora_r", default=4, type=int)
    parser.add_argument("--epochs", default=2, type=int)
    parser.add_argument("--prefix_tokens", default=20, type=int)
    parser.add_argument("--prefix_projection", default=1, type=int)
    parser.add_argument("--max_steps", default=10, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)

    # parser.add_argument("--label_column", default="label_column")

    args = parser.parse_args()
    main(args)



