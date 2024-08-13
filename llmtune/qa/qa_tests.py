from abc import ABC, abstractmethod
from typing import List, Union

import torch
import numpy as np
from langchain.evaluation import JsonValidityEvaluator
from transformers import DistilBertModel, DistilBertTokenizer


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

# TODO this is the same as QaMetricRegistry, could be combined?
class QaTestRegistry:
    """Provides a registry that maps metric names to metric classes.
    A user can provide a list of metrics by name, and the registry will convert
    that into a list of metric objects.
    """
    registry = {}

    @classmethod
    def register(cls, *names):
        def inner_wrapper(wrapped_class):
            for name in names:
                cls.registry[name] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def create_tests_from_list(cls, test_names: List[str]) -> List[LLMQaTest]:
        return [cls.registry[test]() for test in test_names]

    @classmethod
    def from_name(cls, name: str) -> LLMQaTest:
        """Return a Test object from a given name."""
        return cls.registry[name]()


@QaTestRegistry.register("json_valid")
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

    def test(self, model_pred: str) -> bool:
        result = self.json_validity_evaluator.evaluate_strings(prediction=model_pred)
        binary_res = result["score"]
        return bool(binary_res)


@QaTestRegistry.register("cosine_similarity")
class CosineSimilarityTest(LLMQaTest):
    """
    Checks to see if the response of the LLM is within a certain cosine
    similarity to the gold-standard response. Uses a DistilBERT model to encode
    the responses into vectors.
    """

    def __init__(self):
        model_name = "distilbert-base-uncased"
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertModel.from_pretrained(model_name)

    @property
    def test_name(self) -> str:
        return "cosine_similarity"

    def _encode_sentence(self, sentence):
        tokens = self.tokenizer(sentence, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**tokens)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    def test(self, model_pred: str, ground_truth: str, threshold: float=0.8) -> bool:
        embedding_ground_truth = self._encode_sentence(ground_truth)
        embedding_model_prediction = self._encode_sentence(model_pred)
        dot_product_similarity = np.dot(embedding_ground_truth, embedding_model_prediction)
        norm_ground_truth = np.linalg.norm(embedding_ground_truth)
        norm_model_prediction = np.linalg.norm(embedding_model_prediction)
        cosine_similarity = dot_product_similarity / (norm_ground_truth * norm_model_prediction)
        return cosine_similarity >= threshold
