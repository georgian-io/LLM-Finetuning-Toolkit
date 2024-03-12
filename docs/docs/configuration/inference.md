---
sidebar_position: 7
---

# Inference

The `inference` section sets the parameters for the inference stage. It includes:

## Parameters

- `max_new_tokens`: The maximum number of new tokens to generate.
- `use_cache`: Whether to use the cache during inference.
- `do_sample`: Whether to use sampling during inference.
- `top_p`: The cumulative probability threshold for top-p sampling.
- `temperature`: The temperature value for sampling.

## Example

```yaml
inference:
  max_new_tokens: 1024
  use_cache: true
  do_sample: true
  top_p: 0.9
  temperature: 0.8
```
