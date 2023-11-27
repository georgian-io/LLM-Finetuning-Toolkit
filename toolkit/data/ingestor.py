from abc import ABC, abstractclassmethod

import ijson
import csv
from datasets import Dataset, load_dataset


def get_ingestor(data_type: str):
    if data_type == 'json': return JsonIngestor()
    elif data_type == 'csv': return CsvIngestor()
    elif data_type == 'huggingface': return HuggingfaceIngestor()
    else:
        raise ValueError("data 'type' must be one of 'json', 'csv', or 'huggingface'")
    

class Ingestor(ABC):
    @abstractclassmethod
    def to_datasets(cls, path: str):
        pass

class JsonIngestor(Ingestor):
    @classmethod
    def _json_generator(cls, path: str):
        with open(path, 'rb') as f:
            for item in ijson.items(f, ''):
                yield item
    @classmethod
    def to_datasets(cls, path:str):
        return Dataset.from_generator(cls._json_generator(path))


class CsvIngestor(Ingestor):
    @classmethod
    def _csv_generator(path: str):
        with open(path) as csvfile:
            reader = csv.reader(csvfile)

            for row in reader:
                yield row

    @classmethod
    def to_datasets(cls, path:str):
        return Dataset.from_generator(cls._csv_generator(path))


class HuggingfaceIngestor(Ingestor):
    @classmethod
    def to_datasets(cls, path:str):
        return load_dataset(path)

