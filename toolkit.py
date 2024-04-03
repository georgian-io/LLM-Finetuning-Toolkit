from os import listdir
from os.path import join, exists
import yaml
import logging

from transformers import utils as hf_utils
from pydantic import ValidationError
import torch

from src.pydantic_models.config_model import Config
from src.data.dataset_generator import DatasetGenerator
from src.utils.save_utils import DirectoryHelper
from src.utils.ablation_utils import generate_permutations
from src.finetune.lora import LoRAFinetune
from src.inference.lora import LoRAInference
from src.ui.rich_ui import RichUI

hf_utils.logging.set_verbosity_error()
torch._logging.set_logs(all=logging.CRITICAL)


def run_one_experiment(config: Config) -> None:
    dir_helper = DirectoryHelper(config_path, config)

    # Loading Data -------------------------------
    RichUI.before_dataset_creation()

    with RichUI.during_dataset_creation("Injecting Values into Prompt", "monkey"):
        dataset_generator = DatasetGenerator(**config.data.model_dump())

    train_columns = dataset_generator.train_columns
    test_column = dataset_generator.test_column

    dataset_path = dir_helper.save_paths.dataset
    if not exists(dataset_path):
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
    if not exists(weights_path) or not listdir(weights_path):
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
    results_file_path = join(dir_helper.save_paths.results, "results.csv")
    if not exists(results_path) or exists(results_file_path):
        inference_runner = LoRAInference(
            test, test_column, config, dir_helper
        ).infer_all()
        RichUI.after_inference(results_path)
    else:
        RichUI.inference_found(results_path)

    # QA -------------------------------
    # console.rule("[bold blue]:thinking_face: Running LLM Unit Tests")
    qa_path = dir_helper.save_paths.qa
    if not exists(qa_path) or not listdir(qa_path):
        # TODO: Instantiate unit test classes
        llm_tests = config.get('qa', {}).get('llm_tests', [])
        tests = TestRegistry.create_tests_from_list(llm_tests)
         # TODO: Load results.csv
        results_df = pd.read_csv(results_file_path)
        prompts = results_df['prompt'].tolist()
        ground_truths = results_df['ground_truth'].tolist()
        model_preds = results_df['model_prediction'].tolist()
        # TODO: Run Unit Tests
        test_suite = LLMTestSuite(tests, prompts, ground_truths, model_preds)
        # TODO: Save Unit Test Results
        test_suite.save_test_results('unit_test_results.csv')
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
