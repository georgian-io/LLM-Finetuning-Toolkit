from abc import ABC, abstractmethod
from functools import partial

import ijson
import csv
from datasets import Dataset, load_dataset, concatenate_datasets


def get_ingestor(data_type: str):
    if data_type == "json":
        return JsonIngestor
    elif data_type == "csv":
        return CsvIngestor
    elif data_type == "huggingface":
        return HuggingfaceIngestor
    else:
        raise ValueError("data 'type' must be one of 'json', 'csv', or 'huggingface'")


class Ingestor(ABC):
    @abstractmethod
    def to_datasets(self):
        pass


class JsonIngestor(Ingestor):
    def __init__(self, path: str):
        self.path = path

    def _json_generator(self):
        with open(self.path, "rb") as f:
            for item in ijson.items(f, "item"):
                yield item

    def to_datasets(self):
        return Dataset.from_generator(self._json_generator)


class CsvIngestor(Ingestor):
    def __init__(self, path: str):
        self.path = path

    def _csv_generator(self):
        with open(self.path) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                yield row

    def to_datasets(self):
        return Dataset.from_generator(self._csv_generator)


class HuggingfaceIngestor(Ingestor):
    def __init__(self, path: str):
        self.path = path

    def to_datasets(self):
        ds = load_dataset(self.path)
        return concatenate_datasets(ds.values())
