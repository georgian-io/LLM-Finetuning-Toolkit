import requests
from starlette.requests import Request
from typing import Dict
import torch 
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM

from ray import serve
import ray
@serve.deployment(ray_actor_options={"num_gpus": 1})
class TextClassificationDeployment:
    def __init__(self):
        peft_model_id = "weights/classification/assets"
        self._model = AutoPeftModelForCausalLM.from_pretrained(
                            peft_model_id,
                            low_cpu_mem_usage=True,
                            torch_dtype=torch.float16,
                            device_map='auto',
                        )
        self.tokenizer = AutoTokenizer.from_pretrained(peft_model_id)

    def generate(self, text):
        input_ids = self.tokenizer(
            text, return_tensors="pt", truncation=True
        ).input_ids.cuda()
        with torch.inference_mode():
            outputs = self._model.generate(
                    input_ids=input_ids,
                    max_new_tokens=20,
                    do_sample=True,
                    top_p=0.95,
                    temperature=1e-3,
                )
            result = self.tokenizer.batch_decode(
                    outputs.detach().cpu().numpy(), skip_special_tokens=True
                )[0]
            return result
        
    @serve.batch(max_batch_size=8, batch_wait_timeout_s=0.1)
    async def __call__(self, http_request: Request):
        json_request: str = await http_request.json()
        return self.generate(json_request['text'])

ray.init(
    runtime_env={
        "pip": [
            "accelerate>=0.16.0",
            "transformers>=4.26.0",
            "numpy<1.24",  
            "torch",
        ]
    }
)
deployment = TextClassificationDeployment.bind()
