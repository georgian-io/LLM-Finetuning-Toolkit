import random
from datasets import load_dataset
from enum_types import Task
import re
import time
import typer
import requests
import json
from utils import load_json
from constants import CONFIG_FILE_PATH
import sys

PROMPTS = {
    "llama": {
        Task.CLASSIFICATION.value: """Classify the following sentence that is delimited with triple backticks. ### Sentence: %s ### Class: """,
        Task.SUMMARIZATION.value: """Summarize the following dialogue that is delimited with triple backticks. ### Dialogue: %s ### Summary: """
    },
    "red_pajama": {
        Task.CLASSIFICATION.value: """Classify the following sentence that is delimited with triple backticks. Sentence: %s Class: """,
        Task.SUMMARIZATION.value: """Summarize the dialogue. Dialogue: %s Summary: """
    },
    "flan": {
        Task.CLASSIFICATION.value: "Classify the following sentence into a category: %s The answer is: ",
        Task.SUMMARIZATION.value: "summarize: %s"
    },
    "falcon": {
        Task.CLASSIFICATION.value: """Classify the following sentence that is delimited with triple backticks. Sentence: %s Class: """,
        Task.SUMMARIZATION.value: """Summarize the following dialogue that is delimited with triple backticks. Dialogue: %s Summary: """
    }
}

DATASETS = {
    Task.CLASSIFICATION.value: "rungalileo/20_Newsgroups_Fixed",
    Task.SUMMARIZATION.value: "samsum"
}

ENDPOINTS = {
    "vllm": 'http://0.0.0.0:8000/v1/completions',
    "tgi": 'http://0.0.0.0:8080/generate',
    "ray": 'http://0.0.0.0:8000/',
    "bentoml": 'http://0.0.0.0:3000/generate',
    "triton_vllm":'http://0.0.0.0:8000/v2/models/vllm_model/generate'
}

def get_promt_huggingface(model_type, task):

    template_prompt = PROMPTS[model_type][task]
    retry_delay = 1 
    number_of_symbols = 600 # equals to ~100 tokens
    while True:
        try:
            dataset = load_dataset(DATASETS[task], split='train')
            random_sample = random.choice(dataset)
            if task == "summarization":
                random_sentence = random_sample['dialogue']
                random_sentence = " ".join(random_sentence.split('\r\n'))[:number_of_symbols]
            else:
                random_sentence = random_sample['text']
                random_sentence = " ".join(random_sentence.split('\n'))[:number_of_symbols]
            break  
        except Exception as e:  
            time.sleep(retry_delay)  
    
    pattern = re.compile(r'[^a-zA-Z0-9 ]')
    random_sentence = pattern.sub('', random_sentence)
    prompt = template_prompt % random_sentence
    return prompt

def create_post_request(server: str, prompt: str, max_tokens: int, huggingface_repo: str):

    POST_BODY = {
        "tgi": '{{"inputs": "{0}"}}'.format(prompt),
        "vllm": '{{"model": "{0}", "prompt": "{1}", "max_tokens": {2}, "temperature": 0}}'.format(huggingface_repo, prompt, max_tokens),
        "ray": '{{"prompt": "{0}"}}'.format(prompt),
        "triton_vllm": '{{"text_input":"{0}", "parameters": {{"stream": false, "temperature": 0}}}}'.format(prompt),
    }

    return POST_BODY[server]

def update_endpoint(server):
    content = f"""POST {ENDPOINTS[server]}
    Content-Type: application/json
    @./input.json"""

    with open("./target.list", "w") as file:
        file.write(content)

def send_to_vegeta(model_type, task: str, server: str, max_tokens: str, huggingface_repo: str):
    prompt = get_promt_huggingface(model_type, task)
    update_endpoint(server)
    post_body = create_post_request(server, prompt, int(max_tokens), huggingface_repo)
    print(post_body, end="")

def inference(max_tokens: str, server: str, huggingface_repo: str):
    prompt = typer.prompt("Input")
    post_body = create_post_request(server, prompt, int(max_tokens), huggingface_repo)
    json_payload = json.loads(post_body)
    headers = {
        'Content-Type': 'application/json'
    }
    response = requests.post(ENDPOINTS[server], json=json_payload, headers=headers)
    print(response.text)

if __name__ == "__main__":

    request_purpose = sys.argv[1]
    config = load_json(CONFIG_FILE_PATH) 
    huggingface_repo = config["huggingface_repo"]
    
    if request_purpose == "benchmark":
        send_to_vegeta(config["model_type"], config["task"], config["server"], config["max_tokens"], huggingface_repo)
    else:
        inference(config["max_tokens"], config["server"], huggingface_repo)