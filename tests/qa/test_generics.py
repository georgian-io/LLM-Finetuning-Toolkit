import pytest
from pandas import DataFrame

from llmtune.qa.generics import LLMQaTest, LLMTestSuite


@pytest.fixture
def mock_rich_ui(mocker):
    return mocker.patch("llmtune.ui.rich_ui.RichUI")


@pytest.fixture
def example_data():
    data = {
        "prompt": ["What is 2+2?", "What is the capital of France?"],
        "ground_truth": ["4", "Paris"],
        "model_prediction": ["3", "Paris"],
    }
    return DataFrame(data)


@pytest.fixture
def mock_csv(mocker, example_data):
    mocker.patch("pandas.read_csv", return_value=example_data)


class MockQaTest(LLMQaTest):
    @property
    def test_name(self):
        return "Mock Accuracy"

    def get_metric(self, prompt, ground_truth, model_pred):
        return ground_truth == model_pred


@pytest.fixture
def mock_tests():
    return [MockQaTest()]


def test_from_csv(mock_csv, mock_tests, example_data):
    test_suite = LLMTestSuite.from_csv("dummy_path.csv", mock_tests)
    assert len(test_suite.tests) == 1
    assert test_suite.prompts[0] == "What is 2+2?"


def test_run_tests(mock_csv, mock_tests):
    test_suite = LLMTestSuite.from_csv("dummy_path.csv", mock_tests)
    results = test_suite.run_tests()
    assert results["Mock Accuracy"] == [False, True]  # Expected results from the mock test


def test_save_test_results(mock_csv, mock_tests, mocker):
    mocker.patch("pandas.DataFrame.to_csv")
    test_suite = LLMTestSuite.from_csv("dummy_path.csv", mock_tests)
    test_suite.save_test_results("dummy_save_path.csv")
    assert DataFrame.to_csv.called  # Check if pandas DataFrame to_csv was called


# def test_print_test_results(mock_csv, mock_tests, mock_rich_ui):
#     test_suite = LLMTestSuite.from_csv("dummy_path.csv", mock_tests)
#     test_suite.print_test_results()
#     assert mock_rich_ui.qa_display_table.called


def test_print_test_results(capfd, example_data):
    tests = [MockQaTest()]
    test_suite = LLMTestSuite(
        tests, example_data["prompt"], example_data["ground_truth"], example_data["model_prediction"]
    )
    test_suite.print_test_results()
    out, err = capfd.readouterr()

    assert "0.5000" in out
    assert "0.5000" in out
    assert "0.7071" in out
