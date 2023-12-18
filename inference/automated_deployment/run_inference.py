import subprocess
import argparse
from pathlib import Path
import click
import sys
import typer
import json
from constants import CONFIG_FILE_PATH
from utils import load_json
from validation import validate_inference_config

def main():

    config = load_json(CONFIG_FILE_PATH) 
    try:
        validate_inference_config(config)
    except Exception as e:
        print(e)
    else:
        subprocess.run(["chmod", "+x", f"./script_inference.sh"])
        subprocess.run([f"./script_inference.sh", config["server"], config["huggingface_repo"],
                        config["huggingface_token"]])
    
if __name__ == '__main__':
    main()