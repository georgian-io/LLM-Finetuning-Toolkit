import pytest
from unittest.mock import MagicMock
from datasets import Dataset

from llmtune.inference.lora import LoRAInference
from llmtune.utils.save_utils import DirectoryHelper
from test_utils.test_config import get_sample_config  # Adjust import path as needed

from transformers import BitsAndBytesConfig


def test_lora_inference_initialization(mocker):
    # Mock dependencies
    mock_model = mocker.patch(
        "llmtune.inference.lora.AutoPeftModelForCausalLM.from_pretrained",
        return_value=MagicMock(),
    )
    mock_tokenizer = mocker.patch(
        "llmtune.inference.lora.AutoTokenizer.from_pretrained", return_value=MagicMock()
    )

    # Mock configuration and directory helper
    config = get_sample_config()
    dir_helper = MagicMock(
        save_paths=MagicMock(results="results_dir", weights="weights_dir")
    )
    test_dataset = Dataset.from_dict(
        {
            "formatted_prompt": ["prompt1", "prompt2"],
            "label_column_name": ["label1", "label2"],
        }
    )

    inference = LoRAInference(
        test_dataset=test_dataset,
        label_column_name="label_column_name",
        config=config,
        dir_helper=dir_helper,
    )

    mock_model.assert_called_once_with(
        "weights_dir",
        torch_dtype=config.model.casted_torch_dtype,
        quantization_config=BitsAndBytesConfig(),
        device_map=config.model.device_map,
        attn_implementation=config.model.attn_implementation,
    )
    mock_tokenizer.assert_called_once_with(
        "weights_dir", device_map=config.model.device_map
    )


def test_infer_all(mocker):
    mocker.patch(
        "llmtune.inference.lora.AutoPeftModelForCausalLM.from_pretrained",
        return_value=MagicMock(),
    )
    mocker.patch(
        "llmtune.inference.lora.AutoTokenizer.from_pretrained", return_value=MagicMock()
    )
    mocker.patch("os.makedirs")
    mock_open = mocker.patch("builtins.open", mocker.mock_open())
    mock_csv_writer = mocker.patch("csv.writer")

    mock_infer_one = mocker.patch.object(
        LoRAInference, "infer_one", return_value="predicted"
    )

    config = get_sample_config()
    dir_helper = MagicMock(
        save_paths=MagicMock(results="results_dir", weights="weights_dir")
    )
    test_dataset = Dataset.from_dict(
        {"formatted_prompt": ["prompt1"], "label_column_name": ["label1"]}
    )

    inference = LoRAInference(
        test_dataset=test_dataset,
        label_column_name="label_column_name",
        config=config,
        dir_helper=dir_helper,
    )
    inference.infer_all()

    mock_infer_one.assert_called_once_with("prompt1")
    mock_open.assert_called_once_with("results_dir/results.csv", "w", newline="")
    mock_csv_writer.assert_called()  # You might want to add more specific assertions based on your CSV structure
