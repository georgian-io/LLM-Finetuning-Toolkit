---
sidebar_position: 3
---

# Model

The `model` section defines the base model and load settings. It includes the following parameters:

## Parameters

- `hf_model_ckpt`: The path or name of the pre-trained model checkpoint from the [Hugging Face Model Hub](https://huggingface.co/models).
- `device_map`: The device map for model parallelism. Set to "auto" for automatic device mapping or specify a custom device map.
- `quantize`: Boolean flag to enable quantization of the model weights; if true, then loads it with `bitsandbytes` config
- `bitsandbytes`: Settings for quantization using [`BitsAndBytesConfig` object](https://huggingface.co/docs/transformers/main_classes/quantization#transformers.BitsAndBytesConfig) within `transformers`.
  - `load_in_8bit`: Flag to enable 8-bit quantization.
  - `llm_int8_threshold`: Outlier threshold for 8-bit quantization.
  - `llm_int8_skip_modules`: List of module names to exclude from 8-bit quantization.
  - `llm_int8_enable_fp32_cpu_offload`: Flag to enable offloading of non-quantized weights to CPU.
  - `load_in_4bit`: Flag to enable 4-bit quantization using bitsandbytes.
  - `bnb_4bit_compute_dtype`: Compute dtype for 4-bit quantization.
  - `bnb_4bit_quant_type`: Quantization data type for 4-bit quantization.
  - `bnb_4bit_use_double_quant`: Flag to enable double quantization for 4-bit quantization.

## Example

```yaml
model:
  hf_model_ckpt: "NousResearch/Llama-2-7b-hf"
  device_map: "auto"
  quantize: true
  bitsandbytes:
    load_in_4bit: true
    bnb_4bit_compute_dtype: "bf16"
    bnb_4bit_quant_type: "nf4"
```
