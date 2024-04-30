import pytest

from llmtune.inference.generics import Inference


class MockInference(Inference):
    def infer_one(self, prompt: str):
        return "inferred one"

    def infer_all(self):
        return "inferred all"


def test_infer_one():
    mock_inference = MockInference()
    result = mock_inference.infer_one("")
    assert result == "inferred one"


def test_infer_all():
    mock_inference = MockInference()
    result = mock_inference.infer_all()
    assert result == "inferred all"


def test_inference_abstract_class_instantiation():
    with pytest.raises(TypeError):
        _ = Inference()
