from pathlib import Path
from typing import Dict, List

import pandas as pd

from llmtune.inference.lora import LoRAInference
from llmtune.qa.qa_tests import LLMQaTest, QaTestRegistry
from llmtune.ui.rich_ui import RichUI


def assert_all_same(items, filename: Path) -> None:
    assert len(items) > 0
    for item in items:
        assert item == items[0], f"Tests in {filename} are not all the same: {items}"


class TestBank:
    """A test bank is a collection of test cases for a single test type.
    Test banks can be specified using CSV files, and also save their results to CSV files.
    """

    def __init__(self, test: LLMQaTest, cases: List[Dict[str, str]], file_name_stem: str) -> None:
        self.test = test
        self.cases = cases
        self.results: List[bool] = []
        self.file_name = file_name_stem + "_results.csv"

    def generate_results(self, model: LoRAInference) -> None:
        """Generates pass/fail results for each test case, based on the model's predictions."""
        self.results = []  # reset results
        for case in self.cases:
            prompt = case["prompt"]
            model_pred = model.infer_one(prompt)
            # run the test with the model prediction and additional args
            test_args = {k: v for k, v in case.items() if k != "prompt"}
            result = self.test.test(model_pred, **test_args)
            self.results.append(result)

    def save_test_results(self, output_dir: Path, result_col: str = "result") -> None:
        """
        Re-saves the test results in a CSV file, with a results column.
        """
        df = pd.DataFrame(self.cases)
        df[result_col] = self.results
        df.to_csv(output_dir / self.file_name, index=False)


class LLMTestSuite:
    """
    Represents and runs a suite of different tests for LLMs.
    """

    def __init__(
        self,
        test_banks: List[TestBank],
    ) -> None:
        self.test_banks = test_banks

    @staticmethod
    def from_dir(
        dir_path: str,
        test_type_col: str = "Test Type",
    ) -> "LLMTestSuite":
        """Creates an LLMTestSuite from a directory of CSV files.
        Each CSV file is a test bank, which encodes test cases for a certain
        test type.
        """

        csv_files = Path(dir_path).rglob("*.csv")

        test_banks = []
        for file_name in csv_files:
            df = pd.read_csv(file_name)
            test_type_column = df[test_type_col].tolist()
            # everything that isn't the test type column is a test parameter
            params = list(set(df.columns.tolist()) - set(test_type_col))
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
            # get file name stub without extension or path
            test_banks.append(TestBank(test, cases, file_name.stem))
        return LLMTestSuite(test_banks)

    def run_inference(self, model: LoRAInference) -> None:
        """Runs inference on all test cases in all the test banks."""
        for test_bank in self.test_banks:
            test_bank.generate_results(model)

    def print_test_results(self) -> None:
        """Prints the results of the tests in the suite."""
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

    def save_test_results(self, output_dir: Path) -> None:
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
        for test_bank in self.test_banks:
            test_bank.save_test_results(output_dir)
