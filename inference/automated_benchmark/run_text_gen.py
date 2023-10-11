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
@click.option("--servers", default="a10", prompt=True, help="Compute type (tgi, ray, triton, vllm): list through ', ' ")
def main(huggingface_token, huggingface_username, model_name, model_type, task, compute, servers):

    results_directories = [f"./benchmark_results/{folder}/{model_name}/{compute}" 
                           for folder in ["raw", "processed", "plots"]]
    for directory in results_directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

    servers = servers.split(", ")
    
    for server in servers:
        results_path = f"./benchmark_results/raw/{model_name}/{compute}/{server}.txt"
        subprocess.run("docker stop $(docker ps -q)".split())
        subprocess.run(["chmod", "+x", f"./{server}.sh"])
        subprocess.run([f"./{server}.sh", f"{huggingface_username}/{model_name}", huggingface_token, model_type, task, results_path])

if __name__ == "__main__":
    model_name = "a"
    compute = "b"
    server = "c"
    directory = f"./benchmark_results/raw/{model_name}/{compute}/{server}"
    Path(directory).mkdir(parents=True, exist_ok=True)
    #main()