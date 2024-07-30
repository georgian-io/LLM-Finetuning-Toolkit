import statistics
from pathlib import Path
from typing import Dict, List, Union

import pandas as pd

from llmtune.qa.qa_metrics import LLMQaMetric
from llmtune.ui.rich_ui import RichUI


class LLMMetricSuite:
    """
    Represents and runs a suite of metrics on a set of prompts,
    golden responses, and model predictions.
    """

    def __init__(
        self,
        metrics: List[LLMQaMetric],
        prompts: List[str],
        ground_truths: List[str],
        model_preds: List[str],
    ) -> None:
        self.metrics = metrics
        self.prompts = prompts
        self.ground_truths = ground_truths
        self.model_preds = model_preds

        self._results: Dict[str, List[Union[float, int]]] = {}

    @staticmethod
    def from_csv(
        file_path: str,
        metrics: List[LLMQaMetric],
        prompt_col: str = "Prompt",
        gold_col: str = "Ground Truth",
        pred_col="Predicted",
    ) -> "LLMMetricSuite":
        results_df = pd.read_csv(file_path)
        prompts = results_df[prompt_col].tolist()
        ground_truths = results_df[gold_col].tolist()
        model_preds = results_df[pred_col].tolist()
        return LLMMetricSuite(metrics, prompts, ground_truths, model_preds)

    def compute_metrics(self) -> Dict[str, List[Union[float, int]]]:
        results = {}
        for metric in self.metrics:
            metric_results = []
            for prompt, ground_truth, model_pred in zip(self.prompts, self.ground_truths, self.model_preds):
                metric_results.append(metric.get_metric(prompt, ground_truth, model_pred))
            results[metric.metric_name] = metric_results

        self._results = results
        return results

    @property
    def metric_results(self) -> Dict[str, List[Union[float, int]]]:
        return self._results if self._results else self.compute_metrics()

    def print_metric_results(self):
        result_dictionary = self.metric_results
        column_data = {key: list(result_dictionary[key]) for key in result_dictionary}
        mean_values = {key: statistics.mean(column_data[key]) for key in column_data}
        median_values = {key: statistics.median(column_data[key]) for key in column_data}
        stdev_values = {key: statistics.stdev(column_data[key]) for key in column_data}
        # Use the RichUI class to display the table
        RichUI.qa_display_metric_table(result_dictionary, mean_values, median_values, stdev_values)

    def save_metric_results(self, path: str):
        # TODO: save these!
        path = Path(path)
        dir = path.parent

        if not dir.exists():
            dir.mkdir(parents=True, exist_ok=True)

        resultant_dataframe = pd.DataFrame(self.metric_results)
        resultant_dataframe.to_csv(path, index=False)
