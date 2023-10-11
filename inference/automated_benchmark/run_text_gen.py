import subprocess
import argparse
from pathlib import Path
import click 
import sys 

@click.command()
@click.option("--huggingface_token", default="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", prompt=True, help="Your Hugging Face token")
@click.option("--huggingface_username", default="mariiaponom", prompt=True, help="Your Hugging Face username")
@click.option("--model_name", default="falcon_summ_merged", prompt=True, help="Model name (e.g. llama-7b-class): ")
@click.option("--model_type", default="falcon", prompt=True, help="Model type (llama, flan, falcon, red_pajama):")
@click.option("--task", default="summarization", prompt=True, help="Task (classification, summarization): ")
@click.option("--compute", default="a10", prompt=True, help="Compute type (a100, a10): ")
@click.option("--servers", default="tgi", prompt=True, help="Compute type (tgi, ray, triton, vllm): list through ', ' ")
def main(huggingface_token, huggingface_username, model_name, model_type, task, compute, servers):

    results_directories = [f"./benchmark_results/{folder}/{model_name}/{compute}" 
                           for folder in ["raw", "processed", "plots"]]
    for directory in results_directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
 # get_metrics(model_name, task, compute, server, instance_cost)
    servers = servers.split(", ")
    for server in servers:
        results_path = f"./benchmark_results/raw/{model_name}/{compute}/{server}.txt"
        subprocess.run("docker stop $(docker ps -q)".split())
        subprocess.run(["chmod", "+x", f"./{server}.sh"])
        subprocess.run([f"./{server}.sh", f"{huggingface_username}/{model_name}", huggingface_token, 
                                            model_type, task, results_path, compute, server, str(1.64)])
        # example: ./tgi.sh mariiaponom/llama_7b_class fkk423k4j22l34kj2 llama classification ./results.txt
if __name__ == "__main__":
    main()
    
    "./benchmark_results/raw/falcon_summ_merged/a10/tgi.txt"