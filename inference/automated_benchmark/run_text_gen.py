import subprocess
import argparse
from pathlib import Path
import click 
import sys 

@click.command()
@click.option("--huggingface_token", default="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", prompt=True, help="Your Hugging Face token")
@click.option("--huggingface_username", default="tiiuae", prompt=True, help="Your Hugging Face username")
@click.option("--model_name", default="falcon-7b-instruct", prompt=True, help="Model name (e.g. llama-7b-class): ")
@click.option("--model_type", default="falcon", prompt=True, help="Model type (llama, flan, falcon, red_pajama):")
@click.option("--task", default="classification", prompt=True, help="Task (classification, summarization): ")
@click.option("--compute", default="a10", prompt=True, help="Compute type (a100, a10): ")
def main(huggingface_token, huggingface_username, model_name, model_type, task, compute):

    directory = f"./benchmark_results/raw/{model_name.split('/')[1]}"
    Path(directory).mkdir(parents=True, exist_ok=True)

    results_path = f"{directory}/tgi_{compute}.txt"
    subprocess.run("docker stop $(docker ps -q)".split())
    subprocess.run(["chmod", "+x", "./text_gen.sh"])
    subprocess.run(["./text_gen.sh", f"{huggingface_username}/{model_name}", huggingface_token, model_type, task, results_path])

if __name__ == "__main__":
    main()