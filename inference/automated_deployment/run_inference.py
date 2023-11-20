import subprocess
import argparse
from pathlib import Path
import click
import sys
import typer
import json
from constants import CONFIG_FILE_PATH

def main():

    config_data = {}
    config_data["server"] = typer.prompt("Server")
    if config_data["server"] == "ray":
        config_data["path_to_model"] = typer.prompt("Path to model")
    else:
        config_data["huggingface_token"] = typer.prompt("HuggingFace token")
        config_data["huggingface_repo"] = typer.prompt("HuggingFace repository")
    config_data["model_type"] = typer.prompt("Model type")
    config_data["task"] = typer.prompt("Task")
  
    json_data = json.dumps(config_data, indent=4)

    with open(CONFIG_FILE_PATH, 'w') as file:
        file.write(json_data)

    subprocess.run(["chmod", "+x", f"./script_inference.sh"])
    subprocess.run([f"./script_inference.sh", CONFIG_FILE_PATH])
    
if __name__ == '__main__':
    main()