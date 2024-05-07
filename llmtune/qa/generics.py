import statistics
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Union

import pandas as pd

from llmtune.ui.rich_ui import RichUI


class LLMQaTest(ABC):
    @property
    @abstractmethod
    def test_name(self) -> str:
        pass

    @abstractmethod
    def get_metric(self, prompt: str, grount_truth: str, model_pred: str) -> Union[float, int, bool]:
        pass


class LLMTestSuite:
    def __init__(
        self,
        tests: List[LLMQaTest],
        prompts: List[str],
        ground_truths: List[str],
        model_preds: List[str],
    ) -> None:
        self.tests = tests
        self.prompts = prompts
        self.ground_truths = ground_truths
        self.model_preds = model_preds

        self._results = {}

    @staticmethod
    def from_csv(
        file_path: str,
        tests: List[LLMQaTest],
        prompt_col: str = "Prompt",
        gold_col: str = "Ground Truth",
        pred_col="Predicted",
    ) -> "LLMTestSuite":
        results_df = pd.read_csv(file_path)
        prompts = results_df[prompt_col].tolist()
        ground_truths = results_df[gold_col].tolist()
        model_preds = results_df[pred_col].tolist()
        return LLMTestSuite(tests, prompts, ground_truths, model_preds)

    def run_tests(self) -> Dict[str, List[Union[float, int, bool]]]:
        test_results = {}
        for test in self.tests:
            metrics = []
            for prompt, ground_truth, model_pred in zip(self.prompts, self.ground_truths, self.model_preds):
                metrics.append(test.get_metric(prompt, ground_truth, model_pred))
            test_results[test.test_name] = metrics

        self._results = test_results
        return test_results

    @property
    def test_results(self):
        return self._results if self._results else self.run_tests()

    def print_test_results(self):
        result_dictionary = self.test_results
        column_data = {key: list(result_dictionary[key]) for key in result_dictionary}
        mean_values = {key: statistics.mean(column_data[key]) for key in column_data}
        median_values = {key: statistics.median(column_data[key]) for key in column_data}
        stdev_values = {key: statistics.stdev(column_data[key]) for key in column_data}
        # Use the RichUI class to display the table
        RichUI.qa_display_table(result_dictionary, mean_values, median_values, stdev_values)

    def save_test_results(self, path: str):
        # TODO: save these!
        path = Path(path)
        dir = path.parent

        if not dir.exists():
            dir.mkdir(parents=True, exist_ok=True)

        resultant_dataframe = pd.DataFrame(self.test_results)
        resultant_dataframe.to_csv(path, index=False)
