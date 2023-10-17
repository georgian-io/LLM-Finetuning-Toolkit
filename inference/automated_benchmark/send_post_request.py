import random
from datasets import load_dataset
import sys
import re
import time

PROMPTS = {
        "llama": {
            "classification": """Classify the following sentence that is delimited with triple backticks. ### Sentence: %s ### Class: """,
            "summarization": """Summarize the following dialogue that is delimited with triple backticks. ### Dialogue: %s ### Summary: """
        },
        "red_pajama": {
            "classification": """Classify the following sentence that is delimited with triple backticks. Sentence: %s Class: """,
            "summarization": """Summarize the dialogue. Dialogue: %s Summary: """
        },
        "flan": {
            "classification": "Classify the following sentence into a category: %s The answer is: ",
            "summarization": "summarize: %s"
        },
        "falcon": {
            "classification": """Classify the following sentence that is delimited with triple backticks. Sentence: %s Class: """,
            "summarization": """Summarize the following dialogue that is delimited with triple backticks. Dialogue: %s Summary: """
        }
    }

DATASETS = {
        "classification": "rungalileo/20_Newsgroups_Fixed",
        "summarization": "samsum"
    }

ENDPOINTS = {"vllm": 'http://0.0.0.0:8000/v1/completions',
              "tgi": 'http://0.0.0.0:8080/generate',
                "ray": 'http://localhost:8000/'
            }

def get_promt_huggingface(model_type, task):

    template_prompt = PROMPTS[model_type][task]
    retry_delay = 1 

    while True:
        try:
            dataset = load_dataset(DATASETS[task], split='train')
            random_sample = random.choice(dataset)
            if task == "summarization":
                random_sentence = random_sample['dialogue']
                random_sentence = " ".join(random_sentence.split('\r\n'))[:600]
            else:
                random_sentence = random_sample['text']
                random_sentence = " ".join(random_sentence.split('\n'))[:600]
            break  
        except Exception as e:  
            time.sleep(retry_delay)  
    
    pattern = re.compile(r'[^a-zA-Z0-9 ]')
    random_sentence = pattern.sub('', random_sentence)
    prompt = template_prompt % random_sentence
    return prompt

def create_post_request(server, prompt, task, huggingface_repo=None):
    if task == "classification":
        max_tokens = 20
    else:
        max_tokens = 100

    POST_BODY = {
        "tgi": '{{"inputs": "{0}"}}'.format(prompt),
        "vllm": '{{"model": "{0}", "prompt": "{1}", "max_tokens": {2}, "temperature": 0}}'.format(huggingface_repo, prompt, max_tokens),
        "ray": '{{"text": "{0}"}}'.format(prompt),
    }

    return POST_BODY[server]

def update_endpoint(server):
    content = f"""POST {ENDPOINTS[server]}
    Content-Type: application/json
    @./test_text.json"""

    with open("./target.list", "w") as file:
        file.write(content)

if __name__ == "__main__":

    model_type = sys.argv[1]
    task = sys.argv[2]
    server = sys.argv[3]
    huggingface_repo = None
    if len(sys.argv) == 5:
        huggingface_repo = sys.argv[4]
    
    prompt = get_promt_huggingface(model_type, task)
    update_endpoint(server)
    print(create_post_request(server, prompt, task, huggingface_repo), end="")