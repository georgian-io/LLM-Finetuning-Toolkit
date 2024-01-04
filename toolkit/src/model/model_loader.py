from os.path import join, exists

import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    AutoTokenizer,
    ProgressCallback,
)
from peft import (
    prepare_model_for_kbit_training,
    get_peft_model,
    AutoPeftModelForCausalLM,
    LoraConfig,
)
from trl import SFTTrainer
from rich.console import Console


from src.pydantic_models.config_model import Config
from src.utils.save_utils import DirectoryHelper


class ModelLoader:
    def __init__(
        self, config: Config, console: Console, directory_helper: DirectoryHelper
    ):
        self._model_config = config.model
        self._training_args = config.training.training_args
        self._sft_args = config.training.sft_args
        self._lora_config = LoraConfig(**config.lora.model_dump())
        self._console: Console = console
        self._directory_helper = directory_helper
        self._weights_path = self._directory_helper.save_paths.weights

        self.model = None
        self.tokenizer = None

    def load_model_and_tokenizer(self):
        self._console.print(f"Loading {self._model_config.hf_model_ckpt}...")
        model = self._get_model()
        tokenizer = self._get_tokenizer()
        self._console.print(f"{self._model_config.hf_model_ckpt} Loaded :smile:")

        self.model = model
        self.tokenizer = tokenizer

    def _get_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            self._model_config.hf_model_ckpt,
            quantization_config=BitsAndBytesConfig(self._model_config.bitsandbytes),
            use_cache=False,
            device_map=self._model_config.device_map,
        )

        model.config.pretraining_tp = 1

        return model

    def _get_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self._model_config.hf_model_ckpt)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        return tokenizer

    def inject_lora(self):
        self._console.print(f"Injecting Lora Modules")

        self.model.gradient_checkpointing_enable()
        self.model = prepare_model_for_kbit_training(self.model)

        self.model = get_peft_model(self.model, self._lora_config)

        self._console.print(f"LoRA Modules Injected!")

    def train(self, train_dataset: Dataset):
        logging_dir = join(self._weights_path, "/logs")
        training_args = TrainingArguments(
            output_dir=self._weights_path,
            logging_dir=logging_dir,
            report_to="none",
            **self._training_args.model_dump(),
        )

        progress_callback = ProgressCallback()

        trainer = SFTTrainer(
            model=self.model,
            train_dataset=train_dataset,
            peft_config=self._lora_config,
            tokenizer=self.tokenizer,
            packing=True,
            args=training_args,
            dataset_text_field="formatted_prompt",  # TODO: maybe move consts to a dedicated folder
            callbacks=[progress_callback],
            **self._sft_args.model_dump(),
        )

        with self._console.status("Training...", spinner="runner"):
            trainer_stats = trainer.train()
        self._console.print(f"Training Complete")

        trainer.model.save_pretrained(self._weights_path)
        self.tokenizer.save_pretrained(self._weights_path)

        self._console.print(f"Run saved at {self._weights_path}")

    def load_and_merge_from_saved(self):
        # purge VRAM
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()

        # Load from path
        self._console.print("Merging Adapter Weights...")

        dtype = (
            torch.float16
            if self._training_args.fp16
            else torch.bfloat16
            if self._training_args.bf16
            else torch.float32
        )

        self.model = AutoPeftModelForCausalLM.from_pretrained(
            self._weights_path,
            low_cpu_mem_usage=True,
            torch_dtype=dtype,
            device_map=self._model_config.device_map,
        )

        self.model = self.model.merge_and_unload()
        self._console.print("Done Merging")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._weights_path, device_map=self._model_config.device_map
        )
