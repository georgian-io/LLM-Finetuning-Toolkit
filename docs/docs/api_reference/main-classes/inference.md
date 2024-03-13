---
sidebar_label: Inference
sidebar_position: 3
---

# Inference

## Inference Classes

### Inference

```python
class src.inference.inference.Inference()
```

> The `Inference` class is an abstract base class for performing inference.
>
> #### Methods
>
> ```python
> infer_one(self, prompt: str) -> str
> """
> prompt: The input prompt.
> """
> ```
>
> > An abstract method to be implemented by subclasses. Performs inference on a single prompt.
>
> ```python
> infer_all(self) -> None
> ```
>
> > An abstract method to be implemented by subclasses. Performs inference on all test examples.

### LoRAInference

```python
class src.inference.lora.LoRAInference(test_dataset: Dataset, label_column_name: str, config: Config, dir_helper: DirectoryHelper)
"""
test_dataset: The test dataset.
label_column_name: The name of the label column in the test dataset.
config: The configuration object.
dir_helper: The directory helper object.
"""
```

> The `LoRAInference` class is a subclass of Inference for performing inference using LoRA models.
>
> #### Methods
>
> ```python
> infer_all(self) -> None
> ```
>
> > Performs inference on all test examples and saves the results.
>
> ```python
> infer_one(self, prompt: str) -> str
> """
> prompt: The input prompt.
> """
> ```
>
> > Performs inference on a single prompt.
> >
> > **Returns:** The generated text.
