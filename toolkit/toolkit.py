from os import listdir
from os.path import join, exists
import yaml

from rich.console import Console

from transformers.utils import logging

from pydantic import ValidationError

from src.pydantic_models.config_model import Config
from src.data.dataset_generator import DatasetGenerator
from src.utils.save_utils import DirectoryHelper
from src.utils.ablation_utils import generate_permutations
from src.finetune.lora import LoRAFinetune
from src.inference.lora import LoRAInference

logging.set_verbosity_error()


def run_one_experiment(config: Config) -> None:
    dir_helper = DirectoryHelper(config_path, config)

    # Loading Data -------------------------------
    console = Console()
    console.rule("[bold green]Loading Data")

    dataset_generator = DatasetGenerator(console=console, **config.data.model_dump())
    train_columns = dataset_generator.train_columns
    test_column = dataset_generator.test_column

    dataset_path = dir_helper.save_paths.dataset
    if not exists(dataset_path):
        train, test = dataset_generator.get_dataset()
        dataset_generator.save_dataset(dataset_path)
    else:
        train, test = dataset_generator.load_dataset_from_pickle(dataset_path)

    dataset_generator.print_one_example()

    # Loading Model -------------------------------
    console.rule("[bold yellow]:smiley: Finetuning")

    weights_path = dir_helper.save_paths.weights

    # model_loader = ModelLoader(config, console, dir_helper)
    if not exists(weights_path) or not listdir(weights_path):
        finetuner = LoRAFinetune(config, console, dir_helper)
        finetuner.finetune(train)
        finetuner.save_model()
    else:
        console.print(f"Fine-Tuned Model Found at {weights_path}... skipping training")

    # Inference -------------------------------
    console.rule("[bold pink]:face_with_monocle: Testing")
    results_path = dir_helper.save_paths.results
    results_file_path = join(dir_helper.save_paths.results, "results.csv")
    if not exists(results_path) or exists(results_file_path):
        inference_runner = LoRAInference(
            test, test_column, config, console, dir_helper
        ).infer_all()

    # QA -------------------------------
    console.rule("[bold blue]:thinking_face: Running LLM Unit Tests")
    qa_path = dir_helper.save_paths.qa
    if not exists(qa_path) or not listdir(qa_path):
        # TODO: Instantiate unit test classes
        # TODO: Load results.csv
        # TODO: Run Unit Tests
        # TODO: Save Unit Test Results
        pass


if __name__ == "__main__":
    config_path = "./config.yml"  # TODO: parameterize this

    # Load YAML config
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        configs = (
            generate_permutations(config, Config)
            if config.get("ablation", {}).get("use_ablate", False)
            else [config]
        )
    for config in configs:
        try:
            config = Config(**config)
        # validate data with pydantic
        except ValidationError as e:
            print(e.json())

        dir_helper = DirectoryHelper(config_path, config)

        # Reload config from saved config
        with open(join(dir_helper.save_paths.config, "config.yml"), "r") as file:
            config = yaml.safe_load(file)
            config = Config(**config)

        run_one_experiment(config)
