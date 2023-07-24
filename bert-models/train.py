from datasets import load_dataset
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import TrainingArguments, Trainer
import datasets
import argparse
import numpy as np
import pandas as pd
import pickle
import evaluate
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

parser = argparse.ArgumentParser()
parser.add_argument("--train_sample_fraction", default=0.025, type=float)
parser.add_argument("--model", default="bert-base-uncased", type=str)
# 0.025 = ~250 
# full train dataset = 10,000, 0.01 = 100, 0.1 = 1000
# ablate over 0.025, 0.05, 0.1, 0.25, 0.5
args = parser.parse_args()

dataset = load_dataset("rungalileo/20_Newsgroups_Fixed")
labels = list(set(dataset["train"]["label"]))
labels = [li for li in labels if isinstance(li, str)]
label2idx = {labels[idx] : idx for idx in range(len(labels))}
idx2label = {value : key for key, value in label2idx.items()}

def clean_text(texts, labels):
    new_texts, new_labels = [], [] 
    for text, label in zip(texts, labels):
        if isinstance(text, str) and isinstance(label, str):
            new_texts.append(text)
            new_labels.append(label2idx[label])
    new_ids = [i for i in range(len(new_texts))]
    df = pd.DataFrame(
        data={"id" : new_ids, "text" : new_texts, "label" : new_labels}
    )

    return df 

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


train_df = clean_text(
    dataset["train"]["text"], dataset["train"]["label"]
)
# sample n points from train_df
train_df, _ = train_test_split(
    train_df, train_size=args.train_sample_fraction,
    stratify=train_df["label"],
)

test_df = clean_text(
    dataset["test"]["text"], dataset["test"]["label"]
)
train_dataset = datasets.Dataset.from_pandas(train_df)
test_dataset = datasets.Dataset.from_pandas(test_df)
dataset = datasets.DatasetDict(
    {"train" : train_dataset, "test" : test_dataset}
)
print(f"Train dataset size: {len(dataset['train'])}")
print(f"Test dataset size: {len(dataset['test'])}")
num_classes = len(set(train_dataset["label"]))
print(f"Number of classes:{num_classes}")


tokenizer = AutoTokenizer.from_pretrained(args.model)
tokenized_datasets = dataset.map(tokenize_function, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(
    args.model, num_labels=num_classes,
)
out_dir = f"{args.model}-{train_df.shape[0]}"

training_args = TrainingArguments(
    output_dir=out_dir,
    evaluation_strategy="epoch",
    num_train_epochs=5,
    save_strategy="no",
    report_to="none",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
)
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    metrics = {}
    metrics["accuracy"] = metric.compute(
        predictions=predictions, references=labels
    )
    metrics["f1-macro"] = f1_score(labels, predictions, average="macro")
    metrics["f1-weighted"] = f1_score(labels, predictions, average="weighted")
    return metrics


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,
)

trainer_stats = trainer.train()
eval_stats = trainer.evaluate()
print("Training stats:", trainer_stats)
print("Eval stats:", eval_stats)

with open(f"{out_dir}/results.pkl", "wb") as handle:
    result = [trainer_stats, eval_stats]
    pickle.dump(result, handle)
print("Experiment over")


