import pytest
from unittest.mock import patch, MagicMock, mock_open

from llmtune.data.ingestor import (
    CsvIngestor,
    HuggingfaceIngestor,
    JsonIngestor,
    JsonlIngestor,
    get_ingestor,
)

from datasets import Dataset


def test_get_ingestor():
    assert isinstance(get_ingestor("json")(""), JsonIngestor)
    assert isinstance(get_ingestor("jsonl")(""), JsonlIngestor)
    assert isinstance(get_ingestor("csv")(""), CsvIngestor)
    assert isinstance(get_ingestor("huggingface")(""), HuggingfaceIngestor)

    with pytest.raises(ValueError):
        get_ingestor("unsupported_type")


def test_json_ingestor_to_dataset(mocker):
    mock_generator = mocker.patch("llmtune.data.ingestor.JsonIngestor._json_generator")
    mock_dataset = mocker.patch("llmtune.data.ingestor.Dataset")
    JsonIngestor("").to_dataset()

    mock_dataset.from_generator.assert_called_once_with(mock_generator)


def test_jsonl_ingestor_to_dataset(mocker):
    mock_generator = mocker.patch(
        "llmtune.data.ingestor.JsonlIngestor._jsonl_generator"
    )
    mock_dataset = mocker.patch("llmtune.data.ingestor.Dataset")
    JsonlIngestor("").to_dataset()

    mock_dataset.from_generator.assert_called_once_with(mock_generator)


def test_csv_ingestor_to_dataset(mocker):
    mock_generator = mocker.patch("llmtune.data.ingestor.CsvIngestor._csv_generator")
    mock_dataset = mocker.patch("llmtune.data.ingestor.Dataset")
    CsvIngestor("").to_dataset()

    mock_dataset.from_generator.assert_called_once_with(mock_generator)


def test_huggingface_to_dataset(mocker):
    # Setup
    path = "some_path"
    ingestor = HuggingfaceIngestor(path)
    mock_concatenate_datasets = mocker.patch(
        "llmtune.data.ingestor.concatenate_datasets"
    )
    mock_load_dataset = mocker.patch("llmtune.data.ingestor.load_dataset")
    mock_dataset = mocker.patch("llmtune.data.ingestor.Dataset")

    # Configure the mock objects
    mock_dataset = MagicMock(spec=Dataset)
    mock_load_dataset.return_value = {"train": mock_dataset, "test": mock_dataset}
    mock_concatenate_datasets.return_value = mock_dataset

    # Execute
    result = ingestor.to_dataset()

    # Assert
    assert isinstance(result, Dataset)
    mock_load_dataset.assert_called_once_with(path)
    mock_concatenate_datasets.assert_called_once()


@pytest.mark.parametrize(
    "file_content,expected_output",
    [
        (
            '[{"column1": "value1", "column2": "value2"}, {"column1": "value3", "column2": "value4"}]',
            [
                {"column1": "value1", "column2": "value2"},
                {"column1": "value3", "column2": "value4"},
            ],
        )
    ],
)
def test_json_ingestor_generator(file_content, expected_output, mocker):
    mocker.patch("builtins.open", mock_open(read_data=file_content))
    mocker.patch("ijson.items", side_effect=lambda f, prefix: iter(expected_output))
    ingestor = JsonIngestor("dummy_path.json")

    assert list(ingestor._json_generator()) == expected_output


@pytest.mark.parametrize(
    "file_content,expected_output",
    [
        (
            '{"column1": "value1", "column2": "value2"}\n{"column1": "value3", "column2": "value4"}',
            [
                {"column1": "value1", "column2": "value2"},
                {"column1": "value3", "column2": "value4"},
            ],
        )
    ],
)
def test_jsonl_ingestor_generator(file_content, expected_output, mocker):
    mocker.patch("builtins.open", mock_open(read_data=file_content))
    mocker.patch(
        "ijson.items",
        side_effect=lambda f, prefix, multiple_values: (
            iter(expected_output) if multiple_values else iter([])
        ),
    )
    ingestor = JsonlIngestor("dummy_path.jsonl")

    assert list(ingestor._jsonl_generator()) == expected_output


@pytest.mark.parametrize(
    "file_content,expected_output",
    [
        (
            "column1,column2\nvalue1,value2\nvalue3,value4",
            [
                {"column1": "value1", "column2": "value2"},
                {"column1": "value3", "column2": "value4"},
            ],
        )
    ],
)
def test_csv_ingestor_generator(file_content, expected_output, mocker):
    mocker.patch("builtins.open", mock_open(read_data=file_content))
    ingestor = CsvIngestor("dummy_path.csv")

    assert list(ingestor._csv_generator()) == expected_output
