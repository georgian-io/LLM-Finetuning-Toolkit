---
sidebar_position: 6
---

# Training

The `training` section configures the training process. It includes two subsections:

## Parameters

- `training_args`: General training arguments such as the number of epochs, batch size, gradient accumulation steps, optimizer, learning rate, etc.
  - `num_train_epochs`: Number of training epochs.
  - `per_device_train_batch_size`: Batch size per training device.
  - `gradient_accumulation_steps`: Number of steps for gradient accumulation.
  - `gradient_checkpointing`: Flag to enable gradient checkpointing.
  - `optim`: Optimizer to use for training.
  - `logging_steps`: Number of steps between logging.
  - `learning_rate`: Learning rate for the optimizer.
  - `bf16`: Flag to enable BF16 mixed-precision training.
  - `tf32`: Flag to enable TF32 mixed-precision training.
  - `fp16`: Flag to enable FP16 mixed-precision training.
  - `max_grad_norm`: Maximum gradient norm for gradient clipping.
  - `warmup_ratio`: Ratio of total training steps used for a linear warmup.
  - `lr_scheduler_type`: Type of learning rate scheduler.
- `sft_args`: Arguments specific to the SFT (Supervised Fine-Tuning) process.
  - `max_seq_length`: Maximum sequence length for input sequences.
  - `neftune_noise_alpha`: Alpha parameter for NEFTUNE noise embeddings. If not None, activates NEFTUNE noise embeddings.

## Example

```yaml
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
```
