---
sidebar_position: 5
---

# LoRA

The `lora` section configures the Low-Rank Adaptation (LoRA) settings. Supplied arguments are used to construct a `peft` [`LoraConfig` object](https://huggingface.co/docs/peft/en/package_reference/lora). It includes the following parameters:

## Parameters

- `task_type`: Type of transformer architecture; for decoder only - use `CAUSAL_LM`. for encoder-decoder - use `SEQ_2_SEQ_LM`
- `r`: The rank of the LoRA adaptation matrices.
- `lora_alpha`: The scaling factor for the LoRA adaptation.
- `lora_dropout`: The dropout probability for the LoRA layers.
- `target_modules`: The list of module names to apply LoRA to.
- `fan_in_fan_out`: Flag to indicate if the layer weights are stored in a (fan_in, fan_out) order.
- `modules_to_save`: List of additional module names to save in the final checkpoint.
- `layers_to_transform`: The list of layer indices to apply LoRA to.
- `layers_pattern`: The regular expression pattern to match layer names for LoRA application.

## Example

```yaml
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
```

## Advanced Usage

### fan_in_fan_out

The `fan_in_fan_out` parameter is a boolean flag that indicates whether the weights of the layers being adapted are stored in a (fan_in, fan_out) order. This is important for correctly applying the LoRA adaptation.

```yaml title="Example"
lora:
  fan_in_fan_out: true
```

In this example, setting `fan_in_fan_out` to `true` indicates that the weights of the layers being adapted are stored in a `(fan_in, fan_out)` order. If the weights are stored in a different order, you should set this parameter to false.

### layers_to_transform

The `layers_to_transform` parameter is used to specify the indices of the layers to which LoRA should be applied. This allows you to selectively apply LoRA to specific layers of the model.

```yaml title="Example"
lora:
  layers_to_transform: [2, 4, 6]
```

In this example, LoRA will be applied to the layers with indices 2, 4, and 6. The layer indices are zero-based, so the first layer has an index of 0, the second layer has an index of 1, and so on.

You can also specify a single layer index:

```yaml title="Example"
lora:
  layers_to_transform: 3
```

In this case, LoRA will be applied only to the layer with index 3.

### layers_pattern

The `layers_pattern` parameter allows you to specify a regular expression pattern to match the names of the layers to which LoRA should be applied. This provides a more flexible way to select layers based on their names.

```yaml title="Example"
lora:
  layers_pattern: "transformer\.h\.\d+\.attn"
```

In this example, the regular expression pattern `transformer\.h\.\d+\.attn` will match the names of the attention layers in a transformer model. The pattern will match layer names like `transformer.h.0.attn`, `transformer.h.1.attn`, and so on.

You can adjust the regular expression pattern to match the specific layer names in your model.
