---
sidebar_label: Finetuning
sidebar_position: 2
---

# Finetuning

## Finetuning Classes

### Finetune

```python
class src.finetune.finetune.Finetune()
```

> The `Finetune` class is an abstract base class for finetuning models.
>
> #### Methods
>
> ```python
> finetune(self) -> None
> ```
>
> > An abstract method to be implemented by subclasses. Finetunes the model.
>
> ```python
> save_model(self) -> None
> ```
>
> > An abstract method to be implemented by subclasses. Saves the finetuned model.

### LoRAFinetune

```python
class src.finetune.lora.LoRAFinetune(config: Config, directory_helper: DirectoryHelper)
"""
config: The configuration object.
directory_helper: The directory helper object.
"""
```

> The `LoRAFinetune` class is a subclass of Finetune for finetuning models using LoRA (Low-Rank Adaptation).
>
> #### Methods
>
> ```python
> finetune(self, train_dataset: Dataset) -> None
> """
> train_dataset: The training dataset.
> """
> ```
>
> > Finetunes the model using the provided training dataset.
>
> ```python
> save_model(self) -> None
> ```
>
> > Saves the finetuned model.
