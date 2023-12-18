import subprocess
from pathlib import Path 
import typer
import json
from utils import load_json
from constants import PROCESSED_DIR, RAW_DIR, CONFIG_FILE_PATH
from validation import validate_benchmark_config
from validation import ValidationError

def main():

    config = load_json(CONFIG_FILE_PATH)
    try:
        validate_benchmark_config(config)
    except ValidationError as e:
        print(f"An error occurred: {e}")
    else:

        Path(PROCESSED_DIR).mkdir(parents=True, exist_ok=True)
        Path(RAW_DIR).mkdir(parents=True, exist_ok=True)

        server = config["server"]
        model_name = config["model_name"]
        raw_results_path = f"{RAW_DIR}/{model_name}_{server}.txt"
        processed_results_path = f"{PROCESSED_DIR}/{model_name}.csv"

        subprocess.run("docker stop $(docker ps -q)".split())
        subprocess.run(["chmod", "+x", f"./script_benchmark.sh"])
        print("Running benchmark...")
        subprocess.run([f"./script_benchmark.sh", raw_results_path, processed_results_path, 
                        config['duration'], config['rate']])
        print("Benchmark is finished.")
        print(f"Raw results are saved at: {raw_results_path}")
        print(f"Processed results are saved at: {processed_results_path}")
    
if __name__ == "__main__":
    main()