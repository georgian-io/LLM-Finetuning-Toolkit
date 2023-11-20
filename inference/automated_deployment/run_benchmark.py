import subprocess
from pathlib import Path 
import typer
import json
from utils import load_json
from constants import PROCESSED_DIR, RAW_DIR, CONFIG_FILE_PATH

def main():

    hardware = typer.prompt("Hardware")
    results_folder = typer.prompt("Name for the folder with benchmark results")
    duration = typer.prompt("Duration of the test in seconds (e.g. 30s)")
    rate = typer.prompt("Number of requests per second")

    config = load_json(CONFIG_FILE_PATH)

    Path(PROCESSED_DIR).mkdir(parents=True, exist_ok=True)
    Path(f"{RAW_DIR}/{results_folder}/{hardware}").mkdir(parents=True, exist_ok=True)

    server = config["server"]
    raw_results_path = f"{RAW_DIR}/{results_folder}/{hardware}/{server}.txt"
    processed_results_path = f"{PROCESSED_DIR}/{results_folder}.csv"

    subprocess.run("docker stop $(docker ps -q)".split())
    subprocess.run(["chmod", "+x", f"./script_benchmark.sh"])
    subprocess.run([f"./script_benchmark.sh", raw_results_path, processed_results_path, duration,
                    rate, hardware])
    
if __name__ == "__main__":
    main()