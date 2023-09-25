import requests
from starlette.requests import Request
from typing import Dict
import torch 
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM

from ray import serve
import ray
@serve.deployment(ray_actor_options={"num_gpus": 1})
class TextSummarizationDeployment:
    def __init__(self):
        peft_model_id = "weights/summarization/assets"
        self._model = AutoPeftModelForCausalLM.from_pretrained(
                            peft_model_id,
                            low_cpu_mem_usage=True,
                            torch_dtype=torch.float16,
                            load_in_4bit=True,
                        )
        self.tokenizer = AutoTokenizer.from_pretrained(peft_model_id)

    def generate(self, text):
        print('text[0]:', text)
        input_ids = self.tokenizer(
            text, return_tensors="pt", truncation=True
        ).input_ids.cuda()
        print('input_ids:', input_ids)
        with torch.inference_mode():
            outputs = self._model.generate(
                    input_ids=input_ids,
                    max_new_tokens=100,
                    do_sample=True,
                    top_p=0.9,
                    temperature=1e-2,
                )
            print('outputs:', outputs)
            result = self.tokenizer.batch_decode(
                    outputs.detach().cpu().numpy(), skip_special_tokens=True
                )[0]
            return result

    async def __call__(self, http_request: Request):
        json_request: str = await http_request.json()
        return self.generate(json_request['text'])

ray.init(
    runtime_env={
        "pip": [
            "accelerate>=0.16.0",
            "transformers>=4.26.0",
            "numpy<1.24",  # remove when mlflow updates beyond 2.2
            "torch",
        ]
    }
)
deployment = TextSummarizationDeployment.bind()
