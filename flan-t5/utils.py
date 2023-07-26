import pandas as pd
import datasets

from datasets import load_dataset, concatenate_datasets, DatasetDict
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from typing import List, Union


def clean_text(
    texts: List[Union[str, None]], labels: List[Union[str, None]]
) -> pd.DataFrame:
    """
    The News Group dataset needs to be preprocessed as it has a lot of
    entries with NULL text and/or NULL labels.
    In this function we simply filter out the NULL entries, and
    return a new dataframe with clean texts and labels.
    """
    new_texts, new_labels = [], []
    for text, label in zip(texts, labels):
        if isinstance(text, str) and isinstance(label, str):
            new_texts.append(text)
            new_labels.append(label)
    new_ids = [i for i in range(len(new_texts))]
    df = pd.DataFrame(data={"id": new_ids, "text": new_texts, "label": new_labels})

    return df


def get_newsgroup_data(
    args, tokenizer: AutoTokenizer
) -> List[Union[DatasetDict, int, int]]:
    dataset_id = "rungalileo/20_Newsgroups_Fixed"

    # Load dataset from the hub
    dataset = load_dataset(dataset_id)

    train_df = clean_text(dataset["train"]["text"], dataset["train"]["label"])

    # sample n points from train_df
    train_df, _ = train_test_split(
        train_df,
        train_size=args.train_sample_fraction,
        stratify=train_df["label"],
    )

    test_df = clean_text(dataset["test"]["text"], dataset["test"]["label"])

    train_dataset = datasets.Dataset.from_pandas(train_df)
    test_dataset = datasets.Dataset.from_pandas(test_df)

    dataset = DatasetDict({"train": train_dataset, "test": test_dataset})

    print(f"Train dataset size: {len(dataset['train'])}")
    print(f"Test dataset size: {len(dataset['test'])}")

    tokenized_inputs = concatenate_datasets([dataset["train"], dataset["test"]]).map(
        lambda x: tokenizer(x["text"], truncation=True),
        batched=True,
        remove_columns=["text", "label"],
    )

    max_source_length = max([len(x) for x in tokenized_inputs["input_ids"]])
    print(f"Max source length: {max_source_length}")

    tokenized_targets = concatenate_datasets([dataset["train"], dataset["test"]]).map(
        lambda x: tokenizer(x["label"], truncation=True),
        batched=True,
        remove_columns=["text", "label"],
    )

    max_target_length = max([len(x) for x in tokenized_targets["input_ids"]])
    print(f"Max target length: {max_target_length}")

    return dataset, max_source_length, max_target_length


def get_samsum_data(tokenizer: AutoTokenizer) -> List[Union[DatasetDict, int, int]]:
    dataset_id = "samsum"
    # Load dataset from the hub
    dataset = load_dataset(dataset_id)

    print(f"Train dataset size: {len(dataset['train'])}")
    print(f"Test dataset size: {len(dataset['test'])}")

    tokenized_inputs = concatenate_datasets([dataset["train"], dataset["test"]]).map(
        lambda x: tokenizer(x["dialogue"], truncation=True),
        batched=True,
        remove_columns=["dialogue", "summary"],
    )

    max_source_length = max([len(x) for x in tokenized_inputs["input_ids"]])
    print(f"Max source length: {max_source_length}")

    tokenized_targets = concatenate_datasets([dataset["train"], dataset["test"]]).map(
        lambda x: tokenizer(x["summary"], truncation=True),
        batched=True,
        remove_columns=["dialogue", "summary"],
    )

    max_target_length = max([len(x) for x in tokenized_targets["input_ids"]])
    print(f"Max target length: {max_target_length}")

    return dataset, max_source_length, max_target_length
