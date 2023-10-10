import random
from datasets import load_dataset
import sys

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
    random_sample = random.choice(dataset)
    random_sentence = random_sample['inputs']['text'][:100]
    prompt = prompts[model_type][task] % random_sentence
    return prompt

if __name__ == "__main__":
    model_type = sys.argv[1]
    task = sys.argv[2]
    print(get_promt_huggingface(model_type, task), end="")