import random
from datasets import load_dataset
import sys
import re

prompts = {
        "llama": {
            "classification": """Classify the following sentence that is delimited with triple backticks. ### Sentence: %s ### Class: """,
            "summarization": """Summarize the following dialogue that is delimited with triple backticks. ### Dialogue: %s ### Summary: """
        },
        "red_pajama": {
            "classification": """Classify the following sentence that is delimited with triple backticks. Sentence: %s Class: """,
            "summarization": """Summarize the following dialogue that is delimited with triple backticks. Dialogue: %s Summary: """
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

datasets = {
        "classification": "rungalileo/20_Newsgroups_Fixed",
        "summarization": "samsum"
    }

def get_promt_huggingface(model_type, task):

    dataset = load_dataset(datasets[task], split='train')
    template_prompt = prompts[model_type][task]
    random_sample = random.choice(dataset)
    if task == "summarization":
        random_sentence = random_sample['dialogue']
        random_sentence = " ".join(random_sentence.split('\r\n'))[:100]
        pattern = re.compile(r'[^a-zA-Z0-9 ]')
        random_sentence = pattern.sub('', random_sentence)
    else:
        random_sentence = random_sample['text'][:100]
    prompt = template_prompt % random_sentence
    return prompt

if __name__ == "__main__":
    model_type = sys.argv[1]
    task = sys.argv[2]
    
    print(get_promt_huggingface("llama", "summarization"), end="")