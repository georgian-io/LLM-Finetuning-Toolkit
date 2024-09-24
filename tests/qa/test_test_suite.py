from pathlib import Path

import pytest
from pandas import DataFrame

from llmtune.qa.qa_tests import LLMQaTest
from llmtune.qa.test_suite import LLMTestSuite, TestBank, all_same


@pytest.fixture
def mock_rich_ui(mocker):
    return mocker.patch("llmtune.ui.rich_ui.RichUI")


@pytest.fixture
def example_data():
    data = {
        "Prompt": ["What is 2+2?", "What is the capital of France?"],
        "Ground Truth": ["4", "Paris"],
        "Predicted": ["3", "Paris"],
    }
    return DataFrame(data)


@pytest.fixture
def mock_csv(mocker, example_data):
    mocker.patch("pandas.read_csv", return_value=example_data)


# mock a LoRAInference object that returns a value when .infer_one() is called
class MockLoRAInference:
    def infer_one(self, prompt: str) -> str:
        return "Paris"


@pytest.mark.parametrize(
    "data, expected",
    [
        (["a", "a", "a"], True),
        (["a", "b", "a"], False),
        ([], False),
    ],
)
def test_all_same(data, expected):
    assert all_same(data) == expected


@pytest.fixture
def mock_cases():
    return [
        {"prompt": "What is the capital of France?"},
        {"prompt": "What is the capital of Germany?"},
    ]


class MockQaTest(LLMQaTest):
    @property
    def test_name(self):
        return "Mock Accuracy"

    def test(self, model_pred) -> bool:
        return model_pred == "Paris"


@pytest.fixture
def mock_test_banks(mock_cases):
    return [
        TestBank(MockQaTest(), mock_cases, "mock_file_name_stem"),
        TestBank(MockQaTest(), mock_cases, "mock_file_name_stem"),
    ]


def test_test_bank_save_test_results(mocker, mock_cases):
    mocker.patch("pandas.DataFrame.to_csv")
    test_bank = TestBank(MockQaTest(), mock_cases, "mock_file_name_stem")
    test_bank.generate_results(MockLoRAInference())
    test_bank.save_test_results(Path("mock/dir/path"))
    assert DataFrame.to_csv.called  # Check if pandas DataFrame to_csv was called


def test_test_suite_save_test_results(mocker, mock_test_banks):
    mocker.patch("pandas.DataFrame.to_csv")
    ts = LLMTestSuite(mock_test_banks)
    ts.run_inference(MockLoRAInference())
    ts.save_test_results(Path("mock/dir/path/doesnt/exist"))
    assert DataFrame.to_csv.called  # Check if pandas DataFrame to_csv was called


def test_test_suite_from_dir():
    ts = LLMTestSuite.from_dir("examples/test_suite")
    ts.run_inference(MockLoRAInference())


def test_test_suite_print_results(capfd, mock_test_banks):
    ts = LLMTestSuite(mock_test_banks)
    ts.run_inference(MockLoRAInference())
    ts.print_test_results()
    out, _ = capfd.readouterr()
    assert "Mock Accuracy" in out
