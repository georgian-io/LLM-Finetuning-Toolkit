import pytest
from pandas import DataFrame

from llmtune.qa.metric_suite import LLMMetricSuite
from llmtune.qa.qa_metrics import LLMQaMetric


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


class MockQaMetric(LLMQaMetric):
    @property
    def metric_name(self):
        return "Mock Accuracy"

    def get_metric(self, prompt, ground_truth, model_pred) -> int:
        return int(ground_truth == model_pred)


@pytest.fixture
def mock_metrics():
    return [MockQaMetric()]


def test_from_csv(mock_metrics, mock_csv):
    suite = LLMMetricSuite.from_csv("dummy_path.csv", mock_metrics)
    assert len(suite.metrics) == 1
    assert suite.prompts[0] == "What is 2+2?"


def test_compute_metrics(mock_metrics, mock_csv):
    suite = LLMMetricSuite.from_csv("dummy_path.csv", mock_metrics)
    results = suite.compute_metrics()
    assert results["Mock Accuracy"] == [0, 1]  # Expected results from the mock test


def test_save_metric_results(mock_metrics, mocker, mock_csv):
    mocker.patch("pandas.DataFrame.to_csv")
    test_suite = LLMMetricSuite.from_csv("dummy_path.csv", mock_metrics)
    test_suite.save_metric_results("dummy_save_path.csv")
    assert DataFrame.to_csv.called  # Check if pandas DataFrame to_csv was called


def test_print_metric_results(capfd, example_data):
    metrics = [MockQaMetric()]
    suite = LLMMetricSuite(metrics, example_data["Prompt"], example_data["Ground Truth"], example_data["Predicted"])
    suite.print_metric_results()
    out, err = capfd.readouterr()

    assert "0.5000" in out
    assert "0.5000" in out
    assert "0.7071" in out
