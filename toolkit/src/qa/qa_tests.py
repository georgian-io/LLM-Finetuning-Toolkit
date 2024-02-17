from src.qa.qa import LLMQaTest
from typing import Union, List, Tuple, Dict

class DummyLLMQaTest(LLMQaTest):
    @property
    def test_name(self) -> str:
        return "DummyLLMQaTest"

    def get_metric(self, prompt: str, grount_truth: str, model_pred: str, *args, **kwargs) -> Union[float, int, bool]:
        return 0.5