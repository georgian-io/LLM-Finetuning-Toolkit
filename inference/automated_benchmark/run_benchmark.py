import subprocess
import argparse
from pathlib import Path
import click 
import sys 

@click.command()
@click.option("--huggingface_username", default="mariiaponom", prompt=True, help="Your Hugging Face username")
@click.option("--model_name", default="llama_7b_class", prompt=True, help="Model name (e.g. llama-7b-class): ")
@click.option("--model_type", default="llama", prompt=True, help="Model type (llama, flan, falcon, red_pajama):")
@click.option("--task", default="classification", prompt=True, help="Task (classification, summarization): ")
@click.option("--compute", default="a10", prompt=True, help="Compute type (a100, a10): ")
@click.option("--server", default="ray", prompt=True, help="Compute type (tgi, ray, triton, vllm): list through ', ' ")
@click.option("--instance_price", default="1.624", prompt=True, help="Compute type (tgi, ray, triton, vllm): list through ', ' ")
def main(huggingface_username, model_name, model_type, task, compute, server, instance_price):
    Path(f"./benchmark_results/processed").mkdir(parents=True, exist_ok=True)
    Path(f"./benchmark_results/plots/{model_name}/{compute}").mkdir(parents=True, exist_ok=True)
    Path(f"./benchmark_results/raw/{model_name}/{compute}").mkdir(parents=True, exist_ok=True)
    results_path = f"./benchmark_results/raw/{model_name}/{compute}/{server}.txt"
    huggingface_repo = f"{huggingface_username}/{model_name}"
    

    subprocess.run("docker stop $(docker ps -q)".split())
    subprocess.run(["chmod", "+x", f"./script_benchmark.sh"])
    subprocess.run([f"./script_benchmark.sh", huggingface_repo, 
                                        model_type, task, results_path, compute, server, 
                                        instance_price])
    
if __name__ == "__main__":
    main()