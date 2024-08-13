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

class TestBank:
    def __init__(self, test: LLMQaTest, cases: List[Dict[str, str]]):
        self.test = test
        self.cases = cases
        self.results: List[bool] = []

    def generate_results(self, model: LoRAInference) -> None:
        for case in self.cases:
            prompt = case["prompt"]
            model_pred = model.infer_one(prompt)
            # run the test with the model prediction and additional args
            test_args = {k: v for k, v in case.items() if k != "prompt"}
            result = self.test.test(model_pred, **test_args)
            self.results.append(result)


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
            params = list(set(df.columns.tolist()) - set([test_type_col]))
            assert_all_same(test_type_column, file_name)
            test_type = test_type_column[0]
            # TODO validate columns
            test = QaTestRegistry.from_name(test_type)
            cases = []
            # all rows are a test case, encode them all
            for _, row in df.iterrows():
                case = {}
                for param in params:
                    case[param] = row[param]
                cases.append(case)
            test_banks.append(TestBank(test, cases))
        return LLMTestSuite(test_banks)


    def run_inference(self, model: LoRAInference) -> None:
        for test_bank in self.test_banks:
            test_bank.generate_results(model)


    def print_test_results(self):
        # Use the RichUI class to display the table
        test_names, num_passed, num_instances = [], [], []
        for test_bank in self.test_banks:
            test_name = test_bank.test.test_name
            test_results = test_bank.results
            passed = sum(test_results)
            instances = len(test_results)
            test_names.append(test_name)
            num_passed.append(passed)
            num_instances.append(instances)

        RichUI.qa_display_test_table(test_names, num_passed, num_instances)
