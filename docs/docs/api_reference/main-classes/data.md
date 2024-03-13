---
sidebar_label: Data
sidebar_position: 1
---

# Data

## Ingestors

### Ingestor

```python
class src.data.ingestor.Ingestor(path: str)
"""
path: The path of the dataset.
"""
```

> The Ingestor class is an abstract base class for data ingestors.
>
> #### Methods
>
> ```python
> to_dataset(self) -> Dataset
> ```
>
> > An abstract method to be implemented by subclasses. Converts the input data to a Dataset object.
> >
> > **Returns:** The converted Dataset object.

### JSON Ingestor

```python
class src.data.ingestor.JsonIngestor(path: str)
"""
path: The path of the JSON dataset.
"""
```

> The `JsonIngestor` class is a subclass of Ingestor for ingesting JSON data.
>
> #### Methods
>
> ```python
> to_dataset(self) -> Dataset
> ```
>
> > Converts the JSON data to a Dataset object.
> >
> > **Returns:** The converted Dataset object.

### CSV Ingestor

```python
class src.data.ingestor.CsvIngestor(path: str)
"""
path: The path of the CSV dataset.
"""
```

> The `CsvIngestor` class is a subclass of Ingestor for ingesting CSV data.
>
> #### Methods
>
> ```python
> to_dataset(self) -> Dataset
> ```
>
> > Converts the CSV data to a Dataset object.
> >
> > **Returns:** The converted Dataset object.

### Huggingface Ingestor

```python
class src.data.ingestor.HuggingfaceIngestor(path: str)
"""
path: The path or name of the HuggingFace dataset.
"""
```

> The `HuggingfaceIngestor` class is a subclass of Ingestor for ingesting data from a HuggingFace dataset.
>
> #### Methods
>
> ```python
> to_dataset(self) -> Dataset
> ```
>
> > Converts the CSV data to a Dataset object.
> >
> > **Returns:** The converted Dataset object.

### Utilities

```python
class src.data.ingestor.get_ingestor(data_type: str)
# data_type: The type of data ("json", "csv", or "huggingface")
```

> A function to get the appropriate ingestor class based on the data type.
>
> **Returns:** The corresponding ingestor class.

## Dataset Generator

### Dataset Generator

```python
class src.data.dataset_generator.DatasetGenerator(file_type: str, path: str, prompt: str, prompt_stub: str, test_size: Union[float, int], train_size: Union[float, int], train_test_split_seed: int)
"""
file_type: The type of input file ("json", "csv", or "huggingface").
path: The path to the input file or HuggingFace dataset.
prompt: The prompt template for formatting the dataset.
prompt_stub: The prompt stub used during training.
test_size: The size of the test set (float for proportion, int for number of examples).
train_size: The size of the training set (float for proportion, int for number of examples).
train_test_split_seed: The random seed for splitting the dataset.
"""
```

> The `DatasetGenerator` class is responsible for generating and formatting datasets for training and testing.
>
> #### Methods
>
> ```python
> get_dataset(self) -> Tuple[Dataset, Dataset]
> ```
>
> > Generates and returns the formatted train and test datasets.
> >
> > **Returns:** A tuple containing the train and test datasets.
>
> ```python
> save_dataset(self, save_dir: str) -> None
> """
> save_dir: The directory to save the dataset.
> """
> ```
>
> > Saves the generated dataset to the specified directory.
>
> ```python
> load_dataset_from_pickle(self, save_dir: str) -> Tuple[Dataset, Dataset]
> """
> save_dir: The directory containing the dataset pickle file.
> """
> ```
>
> > Saves the generated dataset to the specified directory.
> >
> > **Returns**: A tuple containing the loaded train and test datasets.
