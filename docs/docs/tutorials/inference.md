---
sidebar_position: 3
---

# Inference

To run inference on the test set using the finetuned model, follow these steps:

1. Create an instance of the `LoRAInference` class from `src/inference/lora.py`, passing the required parameters such as `test_dataset`, `label_column_name`, `config`, and `dir_helper`.
2. Call the `infer_all` method on the `LoRAInference` instance to generate predictions for the entire test set.

```python title="Example"
from src.inference.lora import LoRAInference

inference_runner = LoRAInference(
    test_dataset,
    label_column_name="label",
    config=config,
    dir_helper=directory_helper
)

inference_runner.infer_all()
```

The generated predictions will be saved in a `results.csv` file in the directory specified by `dir_helper.save_paths.results`, this is typically `{config.save_dir}/{config_hash}/results/results.csv`.
