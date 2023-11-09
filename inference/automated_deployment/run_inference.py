import subprocess
import argparse
from pathlib import Path
import click
import sys
import typer

def main():

    path_to_model = ""
    huggingface_token = ""
    huggingface_repo = ""
 
    server = typer.prompt("Server")
    if server == "ray":
        answer = typer.prompt("Do you want to use local folder with model files? [Y/n]")
        if answer.lower() == "y":
            path_to_model = typer.prompt("Path to model")
        else:
            huggingface_token = typer.prompt("HuggingFace token")
            huggingface_repo = typer.prompt("HuggingFace repository")
    else:
        huggingface_token = typer.prompt("HuggingFace token")
        huggingface_repo = typer.prompt("HuggingFace repository")
    model_type = typer.prompt("Model type")
    task = typer.prompt("Task")

    subprocess.run(["chmod", "+x", f"./script_inference.sh"])
    subprocess.run([f"./script_inference.sh", huggingface_repo, huggingface_token,
                                        model_type, task, server, path_to_model])
    
if __name__ == '__main__':
    main()