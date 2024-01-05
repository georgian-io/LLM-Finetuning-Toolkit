import os
from os.path import join, exists
from functools import partial
from typing import Tuple
import pickle

import re
from datasets import Dataset
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel

from src.data.ingestor import Ingestor, get_ingestor
from src.utils.rich_print_utils import inject_example_to_rich_layout


class DatasetGenerator:
    def __init__(
        self,
        file_type: str,
        path: str,
        prompt: str,
        prompt_stub: str,
        test_size: float,
        train_size: float,
        train_test_split_seed: int,
        console: Console,
    ):
        self.ingestor: Ingestor = get_ingestor(file_type)
        self.ingestor: Ingestor = self.ingestor(path)

        self.dataset: Dataset = self.ingestor.to_dataset()
        self.prompt: str = prompt
        self.prompt_stub: str = prompt_stub
        self.test_size: float = test_size
        self.train_size: float = train_size
        self.train_test_split_seed: int = train_test_split_seed

        self.train_columns: list = self._get_train_columns()
        self.test_column: str = self._get_test_column()
        self.console: Console = console

    def _get_train_columns(self):
        pattern = r"\{([^}]*)\}"
        return re.findall(pattern, self.prompt)

    def _get_test_column(self):
        pattern = r"\{([^}]*)\}"
        return re.findall(pattern, self.prompt_stub)[0]

    # TODO: stratify_by_column
    def _train_test_split(self):
        self.dataset = self.dataset.train_test_split(
            test_size=self.test_size, train_size=self.train_size, seed=self.train_test_split_seed
        )
        self.console.print(f"Post-Split data size:")
        self.console.print(f"Train: {len(self.dataset['train'])}")
        self.console.print(f"Test: {len(self.dataset['test'])}")

    def _format_one_prompt(self, example, is_test: bool = False):
        train_mapping = {var_name: example[var_name] for var_name in self.train_columns}
        example["formatted_prompt"] = self.prompt.format(**train_mapping)

        if not is_test:
            test_mapping = {self.test_column: example[self.test_column]}
            example["formatted_prompt"] += self.prompt_stub.format(**test_mapping)

        return example

    def _format_prompts(self):
        with self.console.status("Injecting columns into Prompt...", spinner="monkey"):
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
        self.console.print("Saving dataset...")
        os.makedirs(save_dir, exist_ok=True)
        with open(join(save_dir, "dataset.pkl"), "wb") as f:
            pickle.dump(self.dataset, f)
        self.console.print("Dataset saved!")

    def print_one_example(self):
        layout = Layout()
        layout.split_row(
            Layout(Panel("Train Sample"), name="train"),
            Layout(
                Panel("Inference Sample"),
                name="inference",
            ),
        )

        inject_example_to_rich_layout(
            layout["train"], "Train Example", self.dataset["train"][0]
        )
        inject_example_to_rich_layout(
            layout["inference"], "Inference Example", self.dataset["test"][0]
        )

        self.console.print(layout)

    def load_dataset_from_pickle(self, save_dir: str):
        self.console.print(f"Loading formatted dataset from directory {save_dir}")
        data_path = join(save_dir, "dataset.pkl")

        if not exists(data_path):
            raise FileNotFoundError(f"Train set pickle not found at {save_dir}")

        with open(data_path, "rb") as f:
            data = pickle.load(f)
            self.dataset = data

        return self.dataset["train"], self.dataset["test"]
