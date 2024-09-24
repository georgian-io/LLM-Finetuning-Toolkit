import pytest

from llmtune.qa.qa_tests import (
    JSONValidityTest,
)


@pytest.mark.parametrize(
    "test_class",
    [
        JSONValidityTest,
    ],
)
def test_test_return_bool(test_class):
    """Test to ensure that all tests return pass/fail boolean value."""
    test_instance = test_class()
    model_prediction = "This is a model predicted sentence."

    metric_result = test_instance.test(model_prediction)
    assert isinstance(metric_result, bool), f"Expected return type bool, but got {type(metric_result)}."


@pytest.mark.parametrize(
    "input_string,expected_value",
    [
        ('{"Answer": "The cat"}', True),
        ("{'Answer': 'The cat'}", False),  # Double quotes are required in json
        ('{"Answer": "The cat",}', False),  # Trailing comma is not allowed
        ('{"Answer": "The cat", "test": "case"}', True),
        ('```json\n{"Answer": "The cat"}\n```', True),  # this json block can still be processed
        ('Here is an example of a JSON block: {"Answer": "The cat"}', False),
    ],
)
def test_json_valid_metric(input_string: str, expected_value: bool):
    test = JSONValidityTest()
    result = test.test(input_string)
    assert result == expected_value, f"JSON validity should be {expected_value} but got {result}."
