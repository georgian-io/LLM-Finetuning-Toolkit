from typing import Literal, Union, List, Dict, Optional
from pydantic import BaseModel, FilePath, validator, Field

from huggingface_hub.utils import validate_repo_id

import torch

# TODO: Refactor this into multiple files...
HfModelPath = str


class DataConfig(BaseModel):
    file_type: Literal["json", "csv", "huggingface"] = Field(
        None, description="File type"
    )
    path: Union[FilePath, HfModelPath] = Field(
        None, description="Path to the file or HuggingFace model"
    )
    prompt: str = Field(
        None, description="Prompt for the model. Use {} brackets for column name"
    )
    prompt_stub: str = Field(
        None,
        description="Stub for the prompt; this is injected during training. Use {} brackets for column name",
    )
    train_size: Optional[Union[float, int]] = Field(
        0.9,
        description="Size of the training set; float for proportion and int for # of examples",
    )
    test_size: Optional[Union[float, int]] = Field(
        0.1,
        description="Size of the test set; float for proportion and int for # of examples",
    )
    train_test_split_seed: int = Field(
        42,
        description="Seed used in the train test split. This is used to ensure that the train and test sets are the same across runs",
    )

    # @validator("path")
    # def validate_path(cls, v, values, **kwargs):
    #     if "file_type" in values and values["file_type"] == "huggingface":
    #         if not validate_repo_id(v):
    #             raise ValueError("Invalid HuggingFace dataset path")
    #     return v


class BitsAndBytesConfig(BaseModel):
    load_in_8bit: Optional[bool] = Field(
        False, description="Enable 8-bit quantization with LLM.int8()"
    )
    llm_int8_threshold: Optional[float] = Field(
        6.0, description="Outlier threshold for outlier detection in 8-bit quantization"
    )
    llm_int8_skip_modules: Optional[List[str]] = Field(
        None, description="List of modules that we do not want to convert in 8-bit"
    )
    llm_int8_enable_fp32_cpu_offload: Optional[bool] = Field(
        False,
        description="Enable splitting model parts between int8 on GPU and fp32 on CPU",
    )
    llm_int8_has_fp16_weight: Optional[bool] = Field(
        False, description="Run LLM.int8() with 16-bit main weights"
    )

    load_in_4bit: Optional[bool] = Field(
        True,
        description="Enable 4-bit quantization by replacing the Linear layers with FP4/NF4 layers from bitsandbytes",
    )
    bnb_4bit_compute_dtype: Optional[str] = Field(
        torch.bfloat16, description="Computational type for 4-bit quantization"
    )
    bnb_4bit_quant_type: Optional[str] = Field(
        "nf4", description="Quantization data type in the bnb.nn.Linear4Bit layers"
    )
    bnb_4bit_use_double_quant: Optional[bool] = Field(
        True,
        description="Enable nested quantization where the quantization constants from the first quantization are quantized again",
    )


class ModelConfig(BaseModel):
    hf_model_ckpt: Optional[str] = Field(
        "NousResearch/Llama-2-7b-hf",
        description="Path to the model (huggingface repo or local path)",
    )
    device_map: Optional[str] = Field(
        "auto", description="device onto which to load the model"
    )

    quantize: Optional[bool] = Field(False, description="Flag to enable quantization")
    bitsandbytes: BitsAndBytesConfig = Field(
        None, description="Bits and Bytes configuration"
    )

    # @validator("hf_model_ckpt")
    # def validate_model(cls, v, **kwargs):
    #     if not validate_repo_id(v):
    #         raise ValueError("Invalid HuggingFace dataset path")
    #     return v

    @validator("quantize")
    def set_bitsandbytes_to_none_if_no_quantization(cls, v, values, **kwargs):
        if v is False:
            values["bitsandbytes"] = None
        return v

    @validator("device_map")
    def set_device_map_to_none(cls, v, values, **kwargs):
        if v.lower() == "none":
            return None
        return v


class LoraConfig(BaseModel):
    r: Optional[int] = Field(8, description="Lora rank")
    task_type: Optional[str] = Field(
        "CAUSAL_LM", description="Base Model task type during training"
    )

    lora_alpha: Optional[int] = Field(
        16, description="The alpha parameter for Lora scaling"
    )
    bias: Optional[str] = Field(
        "none", description="Bias type for Lora. Can be 'none', 'all' or 'lora_only'"
    )
    lora_dropout: Optional[float] = Field(
        0.1, description="The dropout probability for Lora layers"
    )
    target_modules: Optional[List[str]] = Field(
        None, description="The names of the modules to apply Lora to"
    )
    fan_in_fan_out: Optional[bool] = Field(
        False,
        description="Flag to indicate if the layer to replace stores weight like (fan_in, fan_out)",
    )
    modules_to_save: Optional[List[str]] = Field(
        None,
        description="List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint",
    )
    layers_to_transform: Optional[Union[List[int], int]] = Field(
        None, description="The layer indexes to transform"
    )
    layers_pattern: Optional[str] = Field(None, description="The layer pattern name")
    # rank_pattern: Optional[Dict[str, int]] = Field(
    #     {}, description="The mapping from layer names or regexp expression to ranks"
    # )
    # alpha_pattern: Optional[Dict[str, int]] = Field(
    #     {}, description="The mapping from layer names or regexp expression to alphas"
    # )


# TODO: Get comprehensive Args!
class TrainingArgs(BaseModel):
    num_train_epochs: Optional[int] = Field(1, description="Number of training epochs")
    per_device_train_batch_size: Optional[int] = Field(
        1, description="Batch size per training device"
    )
    gradient_accumulation_steps: Optional[int] = Field(
        1, description="Number of steps for gradient accumulation"
    )
    gradient_checkpointing: Optional[bool] = Field(
        True, description="Flag to enable gradient checkpointing"
    )
    optim: Optional[str] = Field("paged_adamw_32bit", description="Optimizer")
    logging_steps: Optional[int] = Field(100, description="Number of logging steps")
    learning_rate: Optional[float] = Field(2.0e-4, description="Learning rate")
    bf16: Optional[bool] = Field(False, description="Flag to enable bf16")
    tf32: Optional[bool] = Field(False, description="Flag to enable tf32")
    fp16: Optional[bool] = Field(False, description="Flag to enable fp16")
    max_grad_norm: Optional[float] = Field(0.3, description="Maximum gradient norm")
    warmup_ratio: Optional[float] = Field(0.03, description="Warmup ratio")
    lr_scheduler_type: Optional[str] = Field(
        "constant", description="Learning rate scheduler type"
    )


# TODO: Get comprehensive Args!
class SftArgs(BaseModel):
    max_seq_length: Optional[int] = Field(None, description="Maximum sequence length")
    neftune_noise_alpha: Optional[float] = Field(
        None,
        description="If not None, this will activate NEFTune noise embeddings. This can drastically improve model performance for instruction fine-tuning.",
    )


class TrainingConfig(BaseModel):
    training_args: TrainingArgs
    sft_args: SftArgs


# TODO: Get comprehensive Args!
class InferenceConfig(BaseModel):
    max_new_tokens: Optional[int] = Field(None, description="Maximum new tokens")
    use_cache: Optional[bool] = Field(True, description="Flag to enable cache usage")
    do_sample: Optional[bool] = Field(True, description="Flag to enable sampling")
    top_p: Optional[float] = Field(1.0, description="Top p value")
    temperature: Optional[float] = Field(0.1, description="Temperature value")
    epsilon_cutoff: Optional[float] = Field(0.0, description="epsilon cutoff value")
    eta_cutoff: Optional[float] = Field(0.0, description="eta cutoff value")
    top_k: Optional[int] = Field(50, description="top-k sampling")


class AblationConfig(BaseModel):
    use_ablate: Optional[bool] = Field(False, description="Flag to enable ablation")
    study_name: Optional[str] = Field("ablation", description="Name of the study")


class Config(BaseModel):
    save_dir: Optional[str] = Field("./experiments", description="Folder to save to")
    ablation: AblationConfig
    accelerate: Optional[bool] = Field(
        False,
        description="set to True if you want to use multi-gpu training; then launch with `accelerate launch --config_file ./accelerate_config toolkit.py`",
    )
    data: DataConfig
    model: ModelConfig
    lora: LoraConfig
    training: TrainingConfig
    inference: InferenceConfig
