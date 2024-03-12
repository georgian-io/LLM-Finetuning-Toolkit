---
sidebar_position: 9
---

# Putting it All Together

To create a custom configuration file, start by copying the provided template and modify the parameters according to your needs. Pay attention to the structure and indentation of the YAML file to ensure it is parsed correctly.

Once you have defined your configuration, you can run the toolkit with your custom settings. The toolkit will load the configuration file, preprocess the data, train the model, perform inference, and optionally run quality assurance tests and ablation studies based on your configuration.

Remember to adjust the paths, prompts, and other parameters to match your specific use case. Experiment with different settings to find the optimal configuration for your task.

## Example

Here's an example of a complete configuration file combining all the sections:

```yaml
save_dir: "./experiments"

ablation:
  use_ablate: true
  study_name: "ablation_study_1"

data:
  file_type: "csv"
  path: "path/to/your/dataset.csv"
  prompt: "Below is an instruction that describes a task. Write a response that appropriately completes the request. ### Instruction: {instruction} ### Input: {input} ### Output:"
  prompt_stub: "{output}"
  test_size: 0.1
  train_size: 0.9
  train_test_split_seed: 42

model:
  hf_model_ckpt: "NousResearch/Llama-2-7b-hf"
  device_map: "auto"
  quantize: true
  bitsandbytes:
    load_in_4bit: true
    bnb_4bit_compute_dtype: "bf16"
    bnb_4bit_quant_type: "nf4"

lora:
  r: 32
  lora_alpha: 16
  lora_dropout: 0.1
  target_modules:
    - q_proj
    - v_proj
    - k_proj
    - o_proj
    - up_proj
    - down_proj
    - gate_proj
  fan_in_fan_out: false
  modules_to_save: null
  layers_to_transform: null
  layers_pattern: null

training:
  training_args:
    num_train_epochs: 5
    per_device_train_batch_size: 4
    gradient_accumulation_steps: 4
    gradient_checkpointing: true
    optim: "paged_adamw_32bit"
    logging_steps: 100
    learning_rate: 2.0e-4
    bf16: true
    tf32: true
    max_grad_norm: 0.3
    warmup_ratio: 0.03
    lr_scheduler_type: "constant"
  sft_args:
    max_seq_length: 5000
    neftune_noise_alpha: null

inference:
  max_new_tokens: 1024
  use_cache: true
  do_sample: true
  top_p: 0.9
  temperature: 0.8
```
