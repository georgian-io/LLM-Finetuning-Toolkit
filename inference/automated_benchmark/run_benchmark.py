import subprocess
from pathlib import Path 
import typer

BASE_DIR = "./benchmark_results"
PROCESSED_DIR = f"{BASE_DIR}/processed"
PLOTS_DIR = f"{BASE_DIR}/plots"
RAW_DIR = f"{BASE_DIR}/raw"

def main():
    huggingface_repo = typer.prompt("HuggingFace repository")
    model_type = typer.prompt("Model type")
    task = typer.prompt("Task")
    hardware = typer.prompt("Hardware")
    server = typer.prompt("Server: ")
    instance_price = typer.prompt("Instance price")
    model_name = huggingface_repo.split('/')[1]

    Path(PROCESSED_DIR).mkdir(parents=True, exist_ok=True)
    Path(f"{PLOTS_DIR}/{model_name}/{hardware}").mkdir(parents=True, exist_ok=True)
    Path(f"{RAW_DIR}/{model_name}/{hardware}").mkdir(parents=True, exist_ok=True)

    results_path = f"{RAW_DIR}/{model_name}/{hardware}/{server}.txt"

    subprocess.run("docker stop $(docker ps -q)".split())
    subprocess.run(["chmod", "+x", f"./script_benchmark.sh"])
    subprocess.run([f"./script_benchmark.sh", huggingface_repo, 
                                        model_type, task, results_path, hardware, server, 
                                        instance_price])
    
if __name__ == "__main__":
    main()