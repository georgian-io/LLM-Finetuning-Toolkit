import requests
from starlette.requests import Request
from typing import Dict
import torch 
from peft import AutoPeftModelForCausalLM, PeftConfig, PeftModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import sys 
import os 
import json
from ray.serve import Application
from ray import serve
import ray

@serve.deployment(ray_actor_options={"num_gpus": 1})
class TextClassificationDeployment:
    def __init__(self, model_type, task, lora_weights):
        if model_type == "flan":
            config = PeftConfig.from_pretrained(lora_weights)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                config.base_model_name_or_path,
            )
            self._model = PeftModel.from_pretrained(self._model, lora_weights)
            self.tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
        else:
            self._model = AutoPeftModelForCausalLM.from_pretrained(lora_weights,
                                    low_cpu_mem_usage=True,
                                    torch_dtype=torch.float16,
                                    device_map='auto')
            self.tokenizer = AutoTokenizer.from_pretrained(lora_weights)
        self.max_number_of_tokens = 20 if task == "classification" else 100

    def generate(self, text):
        input_ids = self.tokenizer(
            text, return_tensors="pt", truncation=True
        ).input_ids.cuda()
        with torch.inference_mode():
            outputs = self._model.generate(
                    input_ids=input_ids,
                    max_new_tokens=self.max_number_of_tokens,
                    do_sample=True,
                    top_p=0.95,
                    temperature=1e-3,
                )
            result = self.tokenizer.batch_decode(
                    outputs.detach().cpu().numpy(), skip_special_tokens=True
                )[0]
            return [result]
        
    async def __call__(self, http_request: Request):
        json_request = await http_request.json()
        return self.generate(json_request['text'])

def app_builder(args: Dict[str, str]) -> Application:
    return TextClassificationDeployment.bind(args["model_type"], args["task"], args["lora_weights"])
