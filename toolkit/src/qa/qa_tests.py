from src.qa.qa import LLMQaTest
from typing import Union, List, Tuple, Dict

class DummyLLMQaTest(LLMQaTest):
    @property
    def test_name(self) -> str:
        return "DummyLLMQaTest"

    def get_metric(self, prompt: str, grount_truth: str, model_pred: str) -> Union[float, int, bool]:
        return 0.5
    

class AccuracyTest(LLMQaTest):
    @property
    def test_name(self) -> str:
        return "Accuracy"

    def get_metric(self, prompt: str, grount_truth: str, model_pred: str) -> Union[float, int, bool]:
        #TODO: Compute!
        pass