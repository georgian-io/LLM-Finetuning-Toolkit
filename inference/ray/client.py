import requests
import argparse

def main(args):
    if args.task == "classification":
        sample_input = {"text": f"Classify the following sentence that is delimited with triple backticks. ### Sentence: {args.text} ### Class: "}
    elif args.task == "summarization":
        sample_input = {"text": f"Summarize the following dialogue that is delimited with triple backticks. ### Dialogue:  {args.text} ### Summary: "}

    response = requests.post("http://127.0.0.1:8000/", json=sample_input)
    print(response.text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task")
    parser.add_argument("--text")
    args = parser.parse_args()

    main(args)