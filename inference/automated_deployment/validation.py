import requests
import os 
import re

class IncorrectModelType(Exception):
    def __init__(self, message="Incorrect model type, can only be: red_pajama, llama, flan, falcon."):
        self.message = message
        super().__init__(self.message)

class IncorrectServer(Exception):
    def __init__(self, message="Incorrect model type, can only be: vllm, tgi, ray, triton_vllm."):
        self.message = message
        super().__init__(self.message)

class IncorrectHuggingFaceRepo(Exception):
    def __init__(self, message="The provided HuggingFace repository does not exist."):
        self.message = message
        super().__init__(self.message)

class IncorrectFolderPath(Exception):
    def __init__(self, message="The provided folder does not exist."):
        self.message = message
        super().__init__(self.message)

class IncorrectDuaration(Exception):
    def __init__(self, message="The provided duration does not follow the next format: 30s, 600s ..."):
        self.message = message
        super().__init__(self.message)

class IncorrectRate(Exception):
    def __init__(self, message="The rate should be an integer value."):
        self.message = message
        super().__init__(self.message)


def validate_server(config: dict):
    if config.get('server') is None:
        raise IncorrectServer()
    if config['server'] not in ['vllm', 'tgi', 'ray', 'triton_vllm']:
        raise IncorrectServer()
    
def validate_model_type(config: dict):
    if config.get('model_type') is None:
        raise IncorrectModelType()
    if config['model_type'] not in ['red_pajama', 'llama', 'flan', 'falcon']:
        raise IncorrectModelType()
    
def validate_huggingface_repo(config: dict):
    if config.get('huggingface_repo') is None:
        raise IncorrectHuggingFaceRepo()
    url = f"https://huggingface.co/{config['huggingface_repo']}"
    response = requests.head(url)
    if response.status_code != 200:
        raise IncorrectHuggingFaceRepo()
    
def validate_folder_exists(config: dict, field_name: str):
    if config.get(field_name) is None:
        raise IncorrectFolderPath()
    if os.path.isdir(config[field_name]) is False:
        raise IncorrectFolderPath()
    
def validate_duration(config: dict):
    if config.get('duration') is None:
        raise IncorrectDuaration()
    pattern = r'^\d+s$'
    if bool(re.match(pattern, config['duration'])) is False:
        raise IncorrectDuaration()
    
def validate_rate(config: dict):
    if config.get('rate') is None:
        raise IncorrectRate()
    if config['rate'].isdigit() is False:
        raise IncorrectRate()
    
def validate_inference_config(config: dict):
    validate_server(config)
    validate_model_type(config)
    validate_huggingface_repo(config)
    if config['server'] == 'ray':
        validate_folder_exists(config, 'path_to_model')
    
def validate_benchmark_config(config: dict):
    validate_duration(config)
    validate_rate(config)   
