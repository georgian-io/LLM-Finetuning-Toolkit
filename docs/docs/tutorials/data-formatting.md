---
sidebar_position: 1
---

# Data Formatting

To format your data and obtain a Hugging Face `Dataset` output, follow these steps:

1. Create an instance of the `DatasetGenerator` class from `src/data/dataset_generator.py`, passing the required parameters such as `file_type`, `path`, `prompt`, `prompt_stub`, `test_size`, `train_size`, and `train_test_split_seed`.
2. Call the `get_dataset` method on the `DatasetGenerator` instance to obtain the formatted train and test datasets.

```python title="Example"
from src.data.dataset_generator import DatasetGenerator

dataset_generator = DatasetGenerator(
    file_type="csv",
    path="path/to/your/data.csv",
    prompt="Your prompt template with {column_name}",
    prompt_stub="Your prompt stub with {column_name}",
    test_size=0.1,
    train_size=0.9,
    train_test_split_seed=42
)

train_dataset, test_dataset = dataset_generator.get_dataset()
```

The `train_dataset` and `test_dataset` variables will contain the formatted Hugging Face `Dataset` objects.
