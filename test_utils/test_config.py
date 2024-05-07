from llmtune.pydantic_models.config_model import (
    AblationConfig,
    BitsAndBytesConfig,
    Config,
    DataConfig,
    InferenceConfig,
    LoraConfig,
    ModelConfig,
    SftArgs,
    TrainingArgs,
    TrainingConfig,
)


def get_sample_config():
    """Function to return a comprehensive Config object for testing."""
    return Config(
        save_dir="./test",
        ablation=AblationConfig(
            use_ablate=False,
        ),
        model=ModelConfig(
            hf_model_ckpt="NousResearch/Llama-2-7b-hf",
            device_map="auto",
            torch_dtype="auto",
            quantize=False,
            bitsandbytes=BitsAndBytesConfig(
                load_in_8bit=False,
                load_in_4bit=False,
                bnb_4bit_compute_dtype="float32",
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            ),
        ),
        lora=LoraConfig(
            r=8,
            task_type="CAUSAL_LM",
            lora_alpha=16,
            bias="none",
            lora_dropout=0.1,
            target_modules=None,
            fan_in_fan_out=False,
        ),
        training=TrainingConfig(
            training_args=TrainingArgs(
                num_train_epochs=1,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=1,
                optim="adamw_8bit",
                learning_rate=2.0e-4,
                logging_steps=100,
            ),
            sft_args=SftArgs(max_seq_length=512, neftune_noise_alpha=None),
        ),
        inference=InferenceConfig(
            max_length=128,
            do_sample=False,
            num_beams=5,
            temperature=1.0,
            top_k=50,
            top_p=1.0,
            use_cache=True,
        ),
        data=DataConfig(
            file_type="json",
            path="path/to/dataset.json",
            prompt="Your prompt here {column_name}",
            prompt_stub="Stub for prompt {column_name}",
            train_size=0.9,
            test_size=0.1,
            train_test_split_seed=42,
        ),
    )
