from functools import partial
from typing import Tuple

import re
from datasets import Dataset

from toolkit.data.ingestor import Ingestor, get_ingestor


class DatasetGenerator():
    def __init__(self,
                 type:str,
                 path:str,
                 prompt: str,
                 prompt_stub: str,
                 test_size: float,
                 stratefy_by: str
                 ):
        self.ingestor: Ingestor = get_ingestor(type)
        self.dataset: Dataset = self.ingestor.to_datasets(path)
        self.prompt: str = prompt
        self.prompt_stub: str = prompt_stub
        self.test_size: float = test_size
        self.stratefy_by: str = stratefy_by
        self.train_columns: list = self._get_train_columns()
        self.test_column: str = self._get_test_column()

    def _get_train_columns(self):
        pattern = r'\{([^}]*)\}'
        return re.findall(pattern, self.prompt)

    def _get_column_names_from_prompt(self):
        pattern = r'\{([^}]*)\}'
        return re.findall(pattern, self.prompt_stub)[0]

    def _train_test_split(self):
        self.dataset = self.dataset.train_test_split(self.test_size, self.stratefy_by)

    def _format_one_prompt(self, example, is_test:bool = False):
        train_mapping = {var_name: example[var_name] for var_name in self.train_columns}
        example['formatted_prompt'] = self.prompt.format(**train_mapping)

        if is_test:
            test_mapping = {self.test_column: example[self.test_column]}
            example['formatted_prompt'] += self.prompt_stub.format(**test_mapping)

        return example
        
    def _format_prompts(self):
        self.dataset['train'] = self.dataset['train'].map(partial(self._format_one_prompt, is_test=False))
        self.dataset['test'] = self.dataset['test'].map(partial(self._format_one_prompt, is_test=True))


    def get_dataset(self) -> Tuple[Dataset, Dataset]:
        self._train_test_split()
        self._format_prompts()
        return self.dataset['train'], self.dataset['test']
