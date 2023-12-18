import requests
import os 
import re

class ValidationError(Exception):
    pass


def validate_server(config: dict):
    if config.get('server') not in ['vllm', 'tgi', 'ray', 'triton_vllm']:
        raise ValidationError("Incorrect server, can only be: vllm, tgi, ray, triton_vllm.")
    
def validate_model_type(config: dict):
    if config.get('model_type') not in ['red_pajama', 'llama', 'flan', 'falcon']:
        raise ValidationError("Incorrect model type, can only be: red_pajama, llama, flan, falcon.")
    
def validate_huggingface_repo(config: dict):
    repo = config.get('huggingface_repo')
    if repo is None:
        raise ValidationError("The provided HuggingFace repository is not specified.")
    
    url = f"https://huggingface.co/{repo}"
    response = requests.head(url)
    if response.status_code != 200:
        raise ValidationError("The provided HuggingFace repository does not exist.")

def validate_duration(config: dict):
    duration = config.get('duration')
    if duration is None or not re.match(r'^\d+s$', duration):
        raise ValidationError("The provided duration does not follow the format: 30s, 600s, etc.")

def validate_rate(config: dict):
    rate = config.get('rate')
    if rate is None or not str(rate).isdigit():
        raise ValidationError("The rate should be an integer value.")
    
def validate_max_tokens(config: dict):
    max_tokens = config.get('max_tokens')
    if max_tokens is None or not str(max_tokens).isdigit():
        raise ValidationError("The max_tokens should be an integer value.")
    
def validate_inference_config(config: dict):
    validate_server(config)
    validate_model_type(config)
    validate_huggingface_repo(config)
    validate_max_tokens(config)
    
def validate_benchmark_config(config: dict):
    validate_duration(config)
    validate_rate(config)   
