from os.path import join

from datasets import Dataset
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    ProgressCallback,
    TrainingArguments,
)
from trl import SFTTrainer

from llmtune.finetune.generics import Finetune
from llmtune.pydantic_models.config_model import Config
from llmtune.ui.rich_ui import RichUI
from llmtune.utils.save_utils import DirectoryHelper


class LoRAFinetune(Finetune):
    def __init__(self, config: Config, directory_helper: DirectoryHelper):
        self.config = config

        self._model_config = config.model
        self._training_args = config.training.training_args
        self._sft_args = config.training.sft_args
        self._lora_config = LoraConfig(**config.lora.model_dump())
        self._directory_helper = directory_helper
        self._weights_path = self._directory_helper.save_paths.weights
        self._trainer = None

        self.model = None
        self.tokenizer = None

        self.device_map = self._model_config.device_map

        self._load_model_and_tokenizer()
        self._inject_lora()

    def _load_model_and_tokenizer(self):
        ckpt = self._model_config.hf_model_ckpt
        RichUI.on_basemodel_load(ckpt)
        model = self._get_model()
        tokenizer = self._get_tokenizer()
        RichUI.after_basemodel_load(ckpt)

        self.model = model
        self.tokenizer = tokenizer

    def _get_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            self._model_config.hf_model_ckpt,
            quantization_config=BitsAndBytesConfig(**self._model_config.bitsandbytes.model_dump()),
            use_cache=False,
            device_map=self.device_map,
            torch_dtype=self._model_config.casted_torch_dtype,
            attn_implementation=self._model_config.attn_implementation,
        )

        model.config.pretraining_tp = 1

        return model

    def _get_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self._model_config.hf_model_ckpt)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        return tokenizer

    def _inject_lora(self):
        self.model.gradient_checkpointing_enable()
        self.model = prepare_model_for_kbit_training(self.model)
        self.model = get_peft_model(self.model, self._lora_config)

    def finetune(self, train_dataset: Dataset):
        logging_dir = join(self._weights_path, "/logs")
        training_args = TrainingArguments(
            output_dir=self._weights_path,
            logging_dir=logging_dir,
            report_to="none",
            **self._training_args.model_dump(),
        )

        progress_callback = ProgressCallback()

        self._trainer = SFTTrainer(
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

        self._trainer.train()

    def save_model(self) -> None:
        self._trainer.model.save_pretrained(self._weights_path)
        self.tokenizer.save_pretrained(self._weights_path)
