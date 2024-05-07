import logging
import shutil
from pathlib import Path

import torch
import transformers
import typer
import yaml
from pydantic import ValidationError
from typing_extensions import Annotated

import llmtune
from llmtune.constants.files import EXAMPLE_CONFIG_FNAME
from llmtune.data.dataset_generator import DatasetGenerator
from llmtune.finetune.lora import LoRAFinetune
from llmtune.inference.lora import LoRAInference
from llmtune.pydantic_models.config_model import Config
from llmtune.qa.generics import LLMTestSuite
from llmtune.qa.qa_tests import QaTestRegistry
from llmtune.ui.rich_ui import RichUI
from llmtune.utils.ablation_utils import generate_permutations
from llmtune.utils.save_utils import DirectoryHelper


transformers.logging.set_verbosity(transformers.logging.CRITICAL)
torch._logging.set_logs(all=logging.CRITICAL)
logging.captureWarnings(True)


app = typer.Typer()
generate_app = typer.Typer()

app.add_typer(
    generate_app,
    name="generate",
    help="Generate various artefacts, such as config files",
)


def run_one_experiment(config: Config, config_path: Path) -> None:
    dir_helper = DirectoryHelper(config_path, config)

    # Loading Data -------------------------------
    RichUI.before_dataset_creation()

    with RichUI.during_dataset_creation("Injecting Values into Prompt", "monkey"):
        dataset_generator = DatasetGenerator(**config.data.model_dump())

    _ = dataset_generator.train_columns
    test_column = dataset_generator.test_column

    dataset_path = dir_helper.save_paths.dataset
    if not dataset_path.exists():
        train, test = dataset_generator.get_dataset()
        dataset_generator.save_dataset(dataset_path)
    else:
        RichUI.dataset_found(dataset_path)
        train, test = dataset_generator.load_dataset_from_pickle(dataset_path)

    RichUI.dataset_display_one_example(train[0], test[0])
    RichUI.after_dataset_creation(dataset_path, train, test)

    # Loading Model -------------------------------
    RichUI.before_finetune()

    weights_path = dir_helper.save_paths.weights

    # model_loader = ModelLoader(config, console, dir_helper)
    if not weights_path.exists() or not any(weights_path.iterdir()):
        finetuner = LoRAFinetune(config, dir_helper)
        with RichUI.during_finetune():
            finetuner.finetune(train)
        finetuner.save_model()
        RichUI.after_finetune()
    else:
        RichUI.finetune_found(weights_path)

    # Inference -------------------------------
    RichUI.before_inference()
    results_path = dir_helper.save_paths.results
    results_file_path = dir_helper.save_paths.results_file
    if not results_file_path.exists():
        inference_runner = LoRAInference(test, test_column, config, dir_helper)
        inference_runner.infer_all()
        RichUI.after_inference(results_path)
    else:
        RichUI.results_found(results_path)

    RichUI.before_qa()
    qa_file_path = dir_helper.save_paths.qa_file
    if not qa_file_path.exists():
        llm_tests = config.qa.llm_tests
        tests = QaTestRegistry.create_tests_from_list(llm_tests)
        test_suite = LLMTestSuite.from_csv(results_file_path, tests)
        test_suite.save_test_results(qa_file_path)
        test_suite.print_test_results()


@app.command("run")
def run(config_path: Annotated[str, typer.Argument(help="Path of the config yaml file")] = "./config.yml") -> None:
    """Run the entire exmperiment pipeline"""
    # Load YAML config
    with Path(config_path).open("r") as file:
        config = yaml.safe_load(file)
        configs = (
            generate_permutations(config, Config) if config.get("ablation", {}).get("use_ablate", False) else [config]
        )
    for config in configs:
        # validate data with pydantic
        try:
            config = Config(**config)
        except ValidationError as e:
            print(e.json())

        dir_helper = DirectoryHelper(config_path, config)

        # Reload config from saved config
        with dir_helper.save_paths.config_file.open("r") as file:
            config = yaml.safe_load(file)
            config = Config(**config)

        run_one_experiment(config, config_path)


@generate_app.command("config")
def generate_config():
    """
    Generate an example `config.yml` file in current directory
    """
    module_path = Path(llmtune.__file__)
    example_config_path = module_path.parent / EXAMPLE_CONFIG_FNAME
    destination = Path.cwd()
    shutil.copy(example_config_path, destination)
    RichUI.generate_config(EXAMPLE_CONFIG_FNAME)


def cli():
    app()
