import yaml
from rich.panel import Panel
from rich.layout import Layout
from rich.console import Console
from rich.markdown import Markdown


from src.data.dataset_generator import DatasetGenerator
from src.model.model_loader import ModelLoader


if __name__ == "__main__":
    console = Console()
    console.rule("[bold green]Loading Data")

    with open("./config.yml", "r") as file:
        config = yaml.safe_load(file)

    dataset_generator = DatasetGenerator(**config["data"])

    with console.status("Injecting Columns into Prompt...", spinner="monkey"):
        train, test = dataset_generator.get_dataset()

    # console.print(train[0])

    console.rule("[bold yellow]Loading Model")
    model, tokenizer = ModelLoader.get_model_and_tokenizer(**config["model"])

    console.rule("[bold red]Injecting LoRA Adapters")

    console.rule("[bold green]:smiley:Training")
