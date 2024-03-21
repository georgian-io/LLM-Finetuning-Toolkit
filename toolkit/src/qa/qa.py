from abc import ABC, abstractmethod
from typing import Union, List, Tuple, Dict
import pandas as pd
from toolkit.src.ui.rich_ui import RichUI
import statistics
from toolkit.src.qa.qa_tests import *


class LLMQaTest(ABC):
    @property
    @abstractmethod
    def test_name(self) -> str:
        pass

    @abstractmethod
    def get_metric(
        self, prompt: str, grount_truth: str, model_pred: str
    ) -> Union[float, int, bool]:
        pass

class LLMQaTestFactory:
    @classmethod
    def from_string_list(cls, string_list: List[str]) -> List[LLMQaTest]:
        tests = []
        for test_name in string_list:
            tests.append(cls.create_test(test_name))
        return tests

    @classmethod 
    def create_test(cls, test_name: str) -> LLMQaTest:
        if test_name == "Summary Length Test":
            return LengthTest()
        elif test_name == "Jaccard Similarity":
            return JaccardSimilarityTest()
        elif test_name == "Semantic Similarity":
            return DotProductSimilarityTest()
        elif test_name == "Rouge Score":
            return RougeScoreTest()
        elif test_name == "Word Overlap Test":
            return WordOverlapTest()
        elif test_name == "Verb Composition":
            return VerbPercent()
        elif test_name == "Adjective Composition":
            return AdjectivePercent()
        elif test_name == "Noun Composition":
            return NounPercent()

class LLMTestSuite():
    def __init__(self, 
                 tests:List[LLMQaTest],
                 prompts:List[str],
                 ground_truths:List[str],
                 model_preds:List[str]) -> None:

        self.tests = tests
        self.prompts = prompts
        self.ground_truths = ground_truths
        self.model_preds = model_preds

        self.test_results = {}

    def run_tests(self) -> Dict[str, List[Union[float, int, bool]]]:
        test_results = {}
        for test in zip(self.tests):
            metrics = []
            for prompt, ground_truth, model_pred in zip(
                self.prompts, self.ground_truths, self.model_preds
            ):
                metrics.append(test.get_metric(prompt, ground_truth, model_pred))
            test_results[test.test_name] = metrics

        self.test_results = test_results
        return test_results

    @property
    def test_results(self):
        return self.test_results if self.test_results else self.run_tests()

    def print_test_results(self):
        result_dictionary = self.test_results()
        column_data = {
            key: [value for value in result_dictionary[key]]
            for key in result_dictionary
        }
        mean_values = {key: statistics.mean(column_data[key]) for key in column_data}
        median_values = {
            key: statistics.median(column_data[key]) for key in column_data
        }
        stdev_values = {key: statistics.stdev(column_data[key]) for key in column_data}
        # Use the RichUI class to display the table
        RichUI.display_table(
            result_dictionary, mean_values, median_values, stdev_values
        )

    def save_test_results(self, path: str):
        # TODO: save these!
        resultant_dataframe = pd.DataFrame(self.test_results())
        resultant_dataframe.to_csv(path, index=False)
