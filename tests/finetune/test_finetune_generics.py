import pytest

from llmtune.finetune.generics import Finetune


class MockFinetune(Finetune):
    def finetune(self):
        return "finetuning complete"

    def save_model(self):
        return "model saved"


def test_finetune_method():
    mock_finetuner = MockFinetune()
    result = mock_finetuner.finetune()
    assert result == "finetuning complete"


def test_save_model_method():
    mock_finetuner = MockFinetune()
    result = mock_finetuner.save_model()
    assert result == "model saved"


def test_finetune_abstract_class_instantiation():
    with pytest.raises(TypeError):
        _ = Finetune()
