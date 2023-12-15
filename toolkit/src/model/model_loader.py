import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)


class ModelLoader:
    @classmethod
    def get_model_and_tokenizer(cls, model_ckpt: str, quant_4bit: bool = True):
        model = cls._get_model(model_ckpt, quant_4bit)
        tokenizer = cls._get_tokenizer(model_ckpt)

        return model, tokenizer

    @classmethod
    def _get_model(cls, model_ckpt: str, quant_4bit: bool = True):
        bnb_config = (
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_quant_4bit_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            if quant_4bit
            else None
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_ckpt,
            quantization_config=bnb_config,
            use_cache=False,
            device_map="auto",
        )

        model.config.pretraining_tp = 1

        return model

    @classmethod
    def _get_tokenizer(cls, model_ckpt: str):
        tokenizer = AutoTokenizer.from_pretrained(model_ckpt, device_map="auto")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        return tokenizer
