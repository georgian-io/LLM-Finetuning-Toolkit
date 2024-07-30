import statistics
from pathlib import Path
from typing import Dict, List, Union
from dataclasses import dataclass

import pandas as pd

from llmtune.qa.qa_tests import LLMQaTest, QaTestRegistry
from llmtune.ui.rich_ui import RichUI
from llmtune.inference.lora import LoRAInference


def assert_all_same(items, filename: str) -> None:
    assert len(items) > 0
    for item in items:
        assert item == items[0], f"Tests in {filename} are not all the same: {items}"

@dataclass
class TestBank:
    test_type: str
    test: LLMQaTest
    cases: List[Dict[str, str]]  # list of parameters to feed the test
    results: List[bool] = []


class LLMTestSuite:
    """
    Represents and runs a suite of metrics on a set of prompts,
    golden responses, and model predictions.
    """

    def __init__(
        self,
        test_banks: List[TestBank],
    ) -> None:
        self.test_banks = test_banks

    @staticmethod
    def from_dir(
        dir_path: str,
        prompt_col: str = "prompt",
        test_type_col: str = "Test Type",
    ) -> "LLMTestSuite":
        # walk the directory and get all the csv files:
        # for each csv file, load the data into a pandas dataframe
        # then extract the prompts, ground truths, and model predictions

        csv_files = Path(dir_path).rglob("*.csv")

        test_banks = []
        for file_name in csv_files:
            df = pd.read_csv(file_name)
            test_type_column = df[test_type_col].tolist()
            params = df.columns - [test_type_col]
            assert_all_same(test_type_column, file_name)
            test_type = test_type_column[0]
            # TODO validate columns
            # TODO instantiate test and add to bank 
            test = QaTestRegistry.from_name(test_type)
            cases = []
            # all rows are a test case, encode them all
            for _, row in df.iterrows():
                case = {}
                for param in params:
                    case[param] = row[param]
                cases.append(case)
            test_banks.append(TestBank(test_type, test, cases))
        return LLMTestSuite(test_banks)




    def run_inference(self, model: LoRAInference):


    def print_test_results(self):
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
