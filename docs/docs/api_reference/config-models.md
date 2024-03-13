---
sidebar_position: 2
---

# Pydantic Models

## Main Config

### Config

```python
class src.pydantic_models.config_model.Config(save_dir: Optional[str], ablation: AblationConfig, accelerate: Optional[bool], data: DataConfig, model: ModelConfig, lora: LoraConfig, training: TrainingConfig, inference: InferenceConfig)
"""
save_dir: Folder to save to
ablation: Ablation configuration
accelerate: set to True if you want to use multi-gpu training; then launch with `accelerate launch --config_file ./accelerate_config toolkit.py`
data: Data configuration
model: Model configuration
lora: LoRA configuration
training: Training configuration
inference: Inference configuration
"""
```

> Represents the overall configuration for the toolkit.

## Data Config

### DataConfig

```python
class src.pydantic_models.config_model.DataConfig(file_type: Literal["json", "csv", "huggingface"], path: Union[FilePath, HfModelPath], prompt: str, prompt_stub: str, train_size: Optional[Union[float, int]], test_size: Optional[Union[float, int]], train_test_split_seed: int)
"""
file_type: File type
path: Path to the file or HuggingFace model
prompt: Prompt for the model. Use {} brackets for column name
prompt_stub: Stub for the prompt; this is injected during training. Use {} brackets for column name
train_size: Size of the training set; float for proportion and int for # of examples
test_size: Size of the test set; float for proportion and int for # of examples
train_test_split_seed: Seed used in the train test split. This is used to ensure that the train and test sets are the same across runs
"""
```

> Represents the configuration for data ingestion.

## Model Config

### ModelConfig

```python
class src.pydantic_models.config_model.ModelConfig(hf_model_ckpt: Optional[str], device_map: Optional[str], quantize: Optional[bool], bitsandbytes: BitsAndBytesConfig)
"""
hf_model_ckpt: Path to the model (huggingface repo or local path)
device_map: device onto which to load the model
quantize: Flag to enable quantization
bitsandbytes: Bits and Bytes configuration
"""
```

> Represents the configuration for the model.

### BitsAndBytesConfig

```python
class src.pydantic_models.config_model.BitsAndBytesConfig(load_in_8bit: Optional[bool], llm_int8_threshold: Optional[float], llm_int8_skip_modules: Optional[List[str]], llm_int8_enable_fp32_cpu_offload: Optional[bool], llm_int8_has_fp16_weight: Optional[bool], load_in_4bit: Optional[bool], bnb_4bit_compute_dtype: Optional[str], bnb_4bit_quant_type: Optional[str], bnb_4bit_use_double_quant: Optional[bool])
"""
load_in_8bit: Enable 8-bit quantization with LLM.int8()
llm_int8_threshold: Outlier threshold for outlier detection in 8-bit quantization
llm_int8_skip_modules: List of modules that we do not want to convert in 8-bit
llm_int8_enable_fp32_cpu_offload: Enable splitting model parts between int8 on GPU and fp32 on CPU
llm_int8_has_fp16_weight: Run LLM.int8() with 16-bit main weights
load_in_4bit: Enable 4-bit quantization by replacing the Linear layers with FP4/NF4 layers from bitsandbytes
bnb_4bit_compute_dtype: Computational type for 4-bit quantization
bnb_4bit_quant_type: Quantization data type in the bnb.nn.Linear4Bit layers
bnb_4bit_use_double_quant: Enable nested quantization where the quantization constants from the first quantization are quantized again
"""
```

> Represents the configuration for BitsAndBytes quantization.

## LoRA Config

### LoraConfig

```python
class src.pydantic_models.config_model.LoraConfig(r: Optional[int], task_type: Optional[str], lora_alpha: Optional[int], bias: Optional[str], lora_dropout: Optional[float], target_modules: Optional[List[str]], fan_in_fan_out: Optional[bool], modules_to_save: Optional[List[str]], layers_to_transform: Optional[Union[List[int], int]], layers_pattern: Optional[str])
"""
r: Lora rank
task_type: Base Model task type during training
lora_alpha: The alpha parameter for Lora scaling
bias: Bias type for Lora. Can be 'none', 'all' or 'lora_only'
lora_dropout: The dropout probability for Lora layers
target_modules: The names of the modules to apply Lora to
fan_in_fan_out: Flag to indicate if the layer to replace stores weight like (fan_in, fan_out)
modules_to_save: List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint
layers_to_transform: The layer indexes to transform
layers_pattern: The layer pattern name
"""
```

> Represents the configuration for LoRA (Low-Rank Adaptation).

## Training Config

### TrainingConfig

```python
class src.pydantic_models.config_model.TrainingConfig(training_args: TrainingArgs, sft_args: SftArgs)
"""
training_args: Training arguments
sft_args: SFT arguments
"""
```

> Represents the configuration for training.

### TrainingArgs

```python
class src.pydantic_models.config_model.TrainingArgs(num_train_epochs: Optional[int], per_device_train_batch_size: Optional[int], gradient_accumulation_steps: Optional[int], gradient_checkpointing: Optional[bool], optim: Optional[str], logging_steps: Optional[int], learning_rate: Optional[float], bf16: Optional[bool], tf32: Optional[bool], fp16: Optional[bool], max_grad_norm: Optional[float], warmup_ratio: Optional[float], lr_scheduler_type: Optional[str])
"""
num_train_epochs: Number of training epochs
per_device_train_batch_size: Batch size per training device
gradient_accumulation_steps: Number of steps for gradient accumulation
gradient_checkpointing: Flag to enable gradient checkpointing
optim: Optimizer
logging_steps: Number of logging steps
learning_rate: Learning rate
bf16: Flag to enable bf16
tf32: Flag to enable tf32
fp16: Flag to enable fp16
max_grad_norm: Maximum gradient norm
warmup_ratio: Warmup ratio
lr_scheduler_type: Learning rate scheduler type
"""
```

> Represents the training arguments.

### SftArgs

```python
class src.pydantic_models.config_model.SftArgs(max_seq_length: Optional[int], neftune_noise_alpha: Optional[float])
"""
max_seq_length: Maximum sequence length
neftune_noise_alpha: If not None, this will activate NEFTune noise embeddings. This can drastically improve model performance for instruction fine-tuning.
"""
```

> Represents the arguments for Supervised Fine-Tuning (SFT).

## Inference Config

### InferenceConfig

```python
class src.pydantic_models.config_model.InferenceConfig(max_new_tokens: Optional[int], use_cache: Optional[bool], do_sample: Optional[bool], top_p: Optional[float], temperature: Optional[float], epsilon_cutoff: Optional[float], eta_cutoff: Optional[float], top_k: Optional[int])
"""
max_new_tokens: Maximum new tokens
use_cache: Flag to enable cache usage
do_sample: Flag to enable sampling
top_p: Top p value
temperature: Temperature value
epsilon_cutoff: epsilon cutoff value
eta_cutoff: eta cutoff value
top_k: top-k sampling
"""
```

> Represents the configuration for inference.

## Ablation Config

### AblationConfig

```python
class src.pydantic_models.config_model.AblationConfig(use_ablate: Optional[bool], study_name: Optional[str])
"""
use_ablate: Flag to enable ablation
study_name: Name of the study
"""
```

> Represents the configuration for ablation studies.
