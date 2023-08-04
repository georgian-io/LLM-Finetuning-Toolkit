import argparse
import os
import numpy as np
import pandas as pd
import pickle
import nltk

nltk.download("punkt")

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from peft import (
    get_peft_model,
    TaskType,
    LoraConfig,
    PrefixTuningConfig,
)

from utils import get_newsgroup_data


def main(args):
    model_name_or_path = args.pretrained_ckpt

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    # loading dataset
    dataset, max_source_length, max_target_length = get_newsgroup_data(args, tokenizer)
    n_samples = len(dataset["train"]["label"])

    def preprocess_function(sample, padding="max_length"):
        # add prefix to the input for t5
        inputs = [
            "Classify the following sentence into a category: "
            + item.replace("\n", " ")
            + " The answer is: "
            for item in sample["text"]
        ]

        # tokenize inputs
        model_inputs = tokenizer(
            inputs,
            max_length=max_source_length,
            padding=padding,
            truncation=True,
        )

        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(
            text_target=sample["label"],
            max_length=max_target_length,
            padding=padding,
            truncation=True,
        )

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length":
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label]
                for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_dataset = dataset.map(
        preprocess_function, batched=True, remove_columns=["text", "label", "id"]
    )

    print(f"Keys of tokenized dataset: {list(tokenized_dataset['train'].features)}")

    print("Getting PEFT method")
    if args.peft_method == "lora":
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            r=args.lora_r,
            lora_alpha=32,
            lora_dropout=args.dropout,
            target_modules=["q", "v"],
        )
        results_dir = f"experiments/classification_{args.peft_method}_samples-{n_samples}_epochs-{args.epochs}_r-{args.lora_r}_dropout-{args.dropout}"

    elif args.peft_method == "prefix":
        peft_config = PrefixTuningConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            num_virtual_tokens=args.prefix_tokens,
            prefix_projection=True if args.prefix_projection else False,
        )
        results_dir = f"experiments/classification_{args.peft_method}_samples-{n_samples}_epochs-{args.epochs}_prefixTokens-{args.prefix_tokens}_useProjection-{args.prefix_projection}"

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
    model = get_peft_model(model, peft_config)
    print(model.print_trainable_parameters())

    # Define training args
    training_args = Seq2SeqTrainingArguments(
        do_train=True,
        do_eval=True,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="no",
        per_device_eval_batch_size=8,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        output_dir=results_dir,
        auto_find_batch_size=True,
        learning_rate=1e-3,
        num_train_epochs=args.epochs,
        logging_dir=f"{results_dir}/logs",
        report_to="none",
    )

    # we want to ignore tokenizer pad token in the loss
    label_pad_token_id = -100
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8,
    )

    print(f"training_args = {training_args}")
    # Create Trainer instance
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=data_collator,
    )
    model.config.use_cache = False

    trainer_stats = trainer.train()
    train_loss = trainer_stats.training_loss
    eval_stats = trainer.evaluate()
    eval_loss = eval_stats["eval_loss"]
    print(f"Training loss:{train_loss}|Val loss:{eval_loss}")

    peft_model_id = f"{results_dir}/assets"
    trainer.model.save_pretrained(peft_model_id)
    tokenizer.save_pretrained(peft_model_id)

    with open(f"{results_dir}/results.pkl", "wb") as handle:
        run_result = [
            args.epochs,
            args.prefix_tokens,
            args.prefix_projection,
            train_loss,
            eval_loss,
        ]
        pickle.dump(run_result, handle)
    print("Experiment over")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_ckpt", default="google/flan-t5-large")
    parser.add_argument("--peft_method", default="lora")
    parser.add_argument("--lora_r", default=16, type=int)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--prefix_tokens", default=20, type=int)
    parser.add_argument("--prefix_projection", default=1, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--p_tokens", default=20, type=int)
    parser.add_argument("--p_hidden", default=100, type=int)
    parser.add_argument("--prompt_tokens", default=20, type=int)
    parser.add_argument("--train_sample_fraction", default=0.99, type=float)

    args = parser.parse_args()
    main(args)
