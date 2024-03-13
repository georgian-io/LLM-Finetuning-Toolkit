---
sidebar_position: 2
---

# Finetuning

To finetune the model and obtain the finetuned model weights, follow these steps:

1. Create an instance of the `LoRAFinetune` class from `src/finetune/lora.py`, passing the required `config` and `directory_helper` objects.
2. Call the finetune method on the `LoRAFinetune` instance, passing the `train_dataset` obtained from the data formatting step (or a manually created `Dataset` instance).
3. Call the `save_model` method on the `LoRAFinetune` instance to save the finetuned model weights.

```python title="Example"
from src.finetune.lora import LoRAFinetune
from src.pydantic_models.config_model import Config
from src.utils.save_utils import DirectoryHelper

config = Config(...)  # Create a Config object with the desired settings
directory_helper = DirectoryHelper("path/to/config.yml", config)

finetuner = LoRAFinetune(config, directory_helper)
finetuner.finetune(train_dataset)
finetuner.save_model()
```

The finetuned model weights will be saved in the directory specified by `directory_helper.save_paths.weights`, this is typically `{config.save_dir}/{config_hash}/weights`.
