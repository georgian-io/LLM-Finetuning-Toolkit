---
sidebar_position: 2
---

# Data

The `data` section defines how the input data is loaded and preprocessed. It includes the following parameters:

## Parameters

- `file_type`: The type of the input file, which can be "json", "csv", or "huggingface".
- `path`: The path to the input file or the name of the HuggingFace dataset.
- `prompt`: The prompt template used for formatting the input data. Use {} brackets to specify column names.
- `prompt_stub`: The prompt stub used during training (i.e. this will be omitted during inference for completion). Use {} brackets to specify the column name.
- `train_size`: The size of the training set, either as a float (proportion) or an integer (number of examples).
- `test_size`: The size of the test set, either as a float (proportion) or an integer (number of examples).
- `train_test_split_seed`: The random seed used for splitting the data into train and test sets.

## Example

```yaml
data:
  file_type: "csv"
  path: "path/to/your/dataset.csv"
  prompt: >-
    Below is an instruction that describes a task.
    Write a response that appropriately completes the request.
    ### Instruction: {instruction}
    ### Input: {input}
    ### Output:
  prompt_stub: >-
    {output}
  test_size: 0.1
  train_size: 0.9
  train_test_split_seed: 42
```
