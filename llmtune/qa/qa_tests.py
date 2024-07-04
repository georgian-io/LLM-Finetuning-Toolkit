from abc import ABC, abstractmethod

from langchain.evaluation import JsonValidityEvaluator


class LLMQaTest(ABC):
    """
    Abstract base class for a test. A test can be computed over a single
    data instance/llm response, and outputs a boolean value (pass or fail).
    """

    @property
    @abstractmethod
    def test_name(self) -> str:
        pass

    @abstractmethod
    def test(self, prompt: str, grount_truth: str, model_pred: str) -> bool:
        pass


class JSONValidityTest(LLMQaTest):
    """
    Checks to see if valid json can be parsed from the model output, according
    to langchain_core.utils.json.parse_json_markdown
    The JSON can be wrapped in markdown and this test will still pass
    """

    def __init__(self):
        self.json_validity_evaluator = JsonValidityEvaluator()

    @property
    def test_name(self) -> str:
        return "json_valid"

    def test(self, prompt: str, grount_truth: str, model_pred: str) -> bool:
        result = self.json_validity_evaluator.evaluate_strings(prediction=model_pred)
        binary_res = result["score"]
        return bool(binary_res)
