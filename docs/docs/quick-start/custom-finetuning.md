---
sidebar_position: 3
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Custom Finetuning

You can start playing around with config.yaml to launch your own custom training jobs. For a more detailed and nuanced treatment of what you can input into the config file, please reference the "Configuration" section of the documentation.

## Loading Custom Dataset

Change the `file_type` and `path` under `data` in `config.yml` to point to your custom dataset. Ensure your dataset is properly formatted and adjust the prompt accordingly.

<Tabs>
<TabItem value="new" label="New Config" default>
```yaml
 ...
data:
  file_type: "csv"
  path: "path/to/your/dataset.csv"
  prompt: "Your custom prompt template with {column_name} placeholders"
 ...
```
</TabItem>
<TabItem value="old" label="Old Config">
```yaml
 ...
data:
  file_type: "huggingface"
  path: "yahma/alpaca-cleaned"
  prompt:
    >- Below is an instruction that describes a task.
       Write a response that appropriately completes the request.
       ### Instruction: {instruction}
       ### Input: {input}
       ### Output:
 ...
```
</TabItem>
</Tabs>

## Changing LoRA Rank

Adjust the `r` and `lora_alpha` parameters in the `lora` section to experiment with different adaptation strengths.
<Tabs>
<TabItem value="new" label="New Config" default>

```yaml
 ...
lora:
  r: 64
  lora_alpha: 32
 ...
```

</TabItem>
<TabItem value="old" label="Old Config">
```yaml
 ...
lora:
  r: 32
  lora_alpha: 16
 ...
```
</TabItem>
</Tabs>

## Changing Base Model

Modify `hf_model_ckpt` to fine-tune a different base model. Ensure it is compatible with your task and make sure to specify the right modules to tune (different models may have different module names).

<Tabs>
<TabItem value="new" label="New Config" default>

```yaml
 ...
model:
  hf_model_ckpt: "EleutherAI/gpt-neo-1.3B"
  target_modules:
    - c_attn
    - c_proj
    - c_fc
    - c_mlp.0
    - c_mlp.2
 ...
```

</TabItem>
<TabItem value="old" label="Old Config">
```yaml
 ...
model:
  hf_model_ckpt: "NousResearch/Llama-2-7b-hf"
  target_modules:
    - q_proj
    - v_proj
    - k_proj
    - o_proj
    - up_proj
    - down_proj
    - gate_proj
 ...
```
</TabItem>
</Tabs>

In the new config snippet for changing the model, we've updated the hf_model_ckpt to use the "EleutherAI/gpt-neo-1.3B" model instead of "NousResearch/Llama-2-7b-hf". We've also adjusted the target_modules to match the module names specific to the GPT-Neo architecture.

:::warning
Remember to carefully review the documentation and requirements of the new model you choose to ensure compatibility with your task and the toolkit.
:::
