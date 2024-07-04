import pytest

from llmtune.qa.qa_metrics import (
    AdjectivePercentMetric,
    DotProductSimilarityMetric,
    JaccardSimilarityMetric,
    JSONValidityMetric,
    LengthMetric,
    NounPercentMetric,
    RougeScoreMetric,
    VerbPercentMetric,
    WordOverlapMetric,
)


@pytest.mark.parametrize(
    "test_class,expected_type",
    [
        (LengthMetric, int),
        (JaccardSimilarityMetric, float),
        (DotProductSimilarityMetric, float),
        (RougeScoreMetric, float),
        (WordOverlapMetric, float),
        (VerbPercentMetric, float),
        (AdjectivePercentMetric, float),
        (NounPercentMetric, float),
        (JSONValidityMetric, float),
    ],
)
def test_metric_return_type(test_class, expected_type):
    test_instance = test_class()
    prompt = "This is a test prompt."
    ground_truth = "This is a ground truth sentence."
    model_prediction = "This is a model predicted sentence."

    # Depending on the test class, the output could be different.
    metric_result = test_instance.get_metric(prompt, ground_truth, model_prediction)
    assert isinstance(
        metric_result, expected_type
    ), f"Expected return type {expected_type}, but got {type(metric_result)}."


def test_length_metric():
    test = LengthMetric()
    result = test.get_metric("prompt", "short text", "longer text")
    assert result == 1, "Length difference should be 1."


def test_jaccard_similarity_metric():
    test = JaccardSimilarityMetric()
    result = test.get_metric("prompt", "hello world", "world hello")
    assert result == 1.0, "Jaccard similarity should be 1.0 for the same words in different orders."


def test_dot_product_similarity_metric():
    test = DotProductSimilarityMetric()
    result = test.get_metric("prompt", "data", "data")
    assert result >= 0, "Dot product similarity should be non-negative."


def test_rouge_score_metric():
    test = RougeScoreMetric()
    result = test.get_metric("prompt", "the quick brown fox", "the quick brown fox jumps over the lazy dog")
    assert result >= 0, "ROUGE precision should be non-negative."


def test_word_overlap_metric():
    test = WordOverlapMetric()
    result = test.get_metric("prompt", "jump over the moon", "jump around the sun")
    assert result >= 0, "Word overlap percentage should be non-negative."


def test_verb_percent_metric():
    test = VerbPercentMetric()
    result = test.get_metric("prompt", "He eats", "He is eating")
    assert result >= 0, "Verb percentage should be non-negative."


def test_adjective_percent_metric():
    test = AdjectivePercentMetric()
    result = test.get_metric("prompt", "It is beautiful", "It is extremely beautiful")
    assert result >= 0, "Adjective percentage should be non-negative."


def test_noun_percent_metric():
    test = NounPercentMetric()
    result = test.get_metric("prompt", "The cat", "The cat and the dog")
    assert result >= 0, "Noun percentage should be non-negative."


@pytest.mark.parametrize(
    "input_string,expected_value",
    [
        ('{"Answer": "The cat"}', 1),
        ("{'Answer': 'The cat'}", 0),  # Double quotes are required in json
        ('{"Answer": "The cat",}', 0),
        ('{"Answer": "The cat", "test": "case"}', 1),
        ('```json\n{"Answer": "The cat"}\n```', 1),  # this json block can still be processed
        ('Here is an example of a JSON block: {"Answer": "The cat"}', 0),
    ],
)
def test_json_valid_metric(input_string: str, expected_value: float):
    test = JSONValidityMetric()
    result = test.get_metric("prompt", "The cat", input_string)
    assert result == expected_value, f"JSON validity should be {expected_value} but got {result}."
