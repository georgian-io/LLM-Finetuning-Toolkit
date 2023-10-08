import openai

import argparse
import evaluate
import os
import ujson
from pathlib import Path
import warnings

from prompts import (
    get_newsgroup_data_for_ft,
    get_samsum_data_for_ft,
)


# Obtain from OpenAI's website 
openai.organization = "org-Qxm4Gb8DM4gPh1hlxL8eCrsO"
openai.api_key = os.getenv("OPENAI_API_KEY")

warnings.filterwarnings("ignore")


def openai_api_call(
    prompt: str, model, custom_model, max_new_tokens: int, top_p: float=0.95, temperature: float=1e-3
):
    
    if custom_model == "":
        response =  openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
    else:
        response =  openai.ChatCompletion.create(
            model=custom_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    generation = response["choices"][0]["message"]["content"]

    return generation


def write_jsonl(file_path, lines):
    """Create a .jsonl file and dump contents.
    file_path (unicode / Path): The path to the output file.
    lines (list): The JSON-serializable contents of each line.
    """
    data = [ujson.dumps(line, escape_forward_slashes=False) for line in lines]
    Path(file_path).open('w', encoding='utf-8').write('\n'.join(data))


def prepare_data_in_openai_format(content, train_x, train_y):

    FINETUNE_FORMAT = {
        "messages": [
            {
                "role": "system",
                "content": content, 
            },

            {
                "role": "user",
                "content": train_x
            },

            {
                "role": "assistant",
                "content": train_y
            }
        ]
    }

    return FINETUNE_FORMAT


def upload_training_file(args):

    if args.task_type == "summarization":
        train_x, train_y, _, _ = get_samsum_data_for_ft()
        content = "You are a helpful assistant who can summarize conversations."
        save_dir = os.path.join("data", args.task_type)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        file_path = os.path.join(save_dir, "train_samsum.jsonl")
    else:
        train_x, train_y, _, _ = get_newsgroup_data_for_ft(
            args.train_sample_fraction
        )
        content = "You are a helpful assistant who can classify newsletters into the right categories."
        save_dir = os.path.join(
            "data",
            f"{args.task_type}_sample-fraction-{args.train_sample_fraction}"
        )
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        file_path = os.path.join(save_dir, "train_newsgroup.jsonl")

    data = []
    for x, y in zip(train_x, train_y):
        example = prepare_data_in_openai_format(content, x, y)
        data.append(example)

    write_jsonl(file_path, data)
    print("JSON file written.....")

    openai.File.create(
        file=open(file_path, "rb"),
        purpose='fine-tune'
    )
    print("File created.....")


def submit_finetuning_job(args):
    openai.FineTuningJob.create(
        training_file=args.training_file_id,
        model=args.model,
        hyperparameters={"n_epochs":args.epochs}
    )
    print("Finetuning job submitted")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--job_type", default="upload_data", choices=["upload_data", "submit_job"])

    parser.add_argument("--task_type", default="summarization", type=str)
    parser.add_argument("--train_sample_fraction", default=0.25, type=float)

    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--model", default="gpt-3.5-turbo")
    parser.add_argument("--training_file_id", default="file-abc123")

    args = parser.parse_args()

    if args.job_type == "upload_data":
        upload_training_file(args)
    elif args.job_type == "submit_job":
        submit_finetuning_job(args)

