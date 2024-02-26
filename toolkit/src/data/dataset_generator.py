import os
from os.path import join, exists
from functools import partial
from typing import Tuple, Union
import pickle

import re
from datasets import Dataset
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel

from src.data.ingestor import Ingestor, get_ingestor


class DatasetGenerator:
    def __init__(
        self,
        file_type: str,
        path: str,
        prompt: str,
        prompt_stub: str,
        test_size: Union[float, int],
        train_size: Union[float, int],
        train_test_split_seed: int,
    ):
        self.ingestor: Ingestor = get_ingestor(file_type)
        self.ingestor: Ingestor = self.ingestor(path)

        self.dataset: Dataset = self.ingestor.to_dataset()
        self.prompt: str = prompt
        self.prompt_stub: str = prompt_stub
        self.test_size = test_size
        self.train_size = train_size
        self.train_test_split_seed: int = train_test_split_seed

        self.train_columns: list = self._get_train_columns()
        self.test_column: str = self._get_test_column()

    def _get_train_columns(self):
        pattern = r"\{([^}]*)\}"
        return re.findall(pattern, self.prompt)

    def _get_test_column(self):
        pattern = r"\{([^}]*)\}"
        return re.findall(pattern, self.prompt_stub)[0]

    # TODO: stratify_by_column
    def _train_test_split(self):
        self.dataset = self.dataset.train_test_split(
            test_size=self.test_size,
            train_size=self.train_size,
            seed=self.train_test_split_seed,
        )

    def _format_one_prompt(self, example, is_test: bool = False):
        train_mapping = {var_name: example[var_name] for var_name in self.train_columns}
        example["formatted_prompt"] = self.prompt.format(**train_mapping)

        if not is_test:
            test_mapping = {self.test_column: example[self.test_column]}
            example["formatted_prompt"] += self.prompt_stub.format(**test_mapping)

        return example

    def _format_prompts(self):
        self.dataset["train"] = self.dataset["train"].map(
            partial(self._format_one_prompt, is_test=False)
        )
        self.dataset["test"] = self.dataset["test"].map(
            partial(self._format_one_prompt, is_test=True)
        )

    def get_dataset(self) -> Tuple[Dataset, Dataset]:
        self._train_test_split()
        self._format_prompts()

        return self.dataset["train"], self.dataset["test"]

    def save_dataset(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        with open(join(save_dir, "dataset.pkl"), "wb") as f:
            pickle.dump(self.dataset, f)

    def load_dataset_from_pickle(self, save_dir: str):
        data_path = join(save_dir, "dataset.pkl")

        if not exists(data_path):
            raise FileNotFoundError(f"Train set pickle not found at {save_dir}")

        with open(data_path, "rb") as f:
            data = pickle.load(f)
            self.dataset = data

        return self.dataset["train"], self.dataset["test"]
