import pytest
from pydantic import BaseModel
from llmtune.utils.ablation_utils import (
    get_types_from_dict,
    get_annotation,
    get_model_field_type,
    validate_and_get_ablations,
    patch_with_permutation,
    generate_permutations,
)


# Mocks or fixtures for models and data if necessary
class BarModel(BaseModel):
    baz: list
    qux: str


class ConfigModel(BaseModel):
    foo: int
    bar: BarModel


@pytest.fixture
def example_yaml():
    return {"foo": 10, "bar": {"baz": [[1, 2, 3], [4, 5]], "qux": ["hello", "world"]}}


def test_get_types_from_dict(example_yaml):
    expected = {"foo": (int, None), "bar.baz": (list, list), "bar.qux": (list, str)}
    assert get_types_from_dict(example_yaml) == expected


def test_get_annotation():
    key = "foo"
    assert get_annotation(key, ConfigModel) == int

    key_nested = "bar.qux"
    # Assuming you adjust your FakeModel or real implementation for nested annotations correctly
    assert get_annotation(key_nested, ConfigModel) == str


def test_get_model_field_type_from_typing_list():
    from typing import List

    annotation = List[int]
    assert get_model_field_type(annotation) == list


def test_get_model_field_type_from_union():
    from typing import Union

    annotation = Union[int, str]
    # Assuming the first type is picked from Union for simplicity
    assert get_model_field_type(annotation) == int


def test_patch_with_permutation():
    old_dict = {"foo": {"bar": 10, "baz": 20}}
    permutation_dict = {"foo.bar": 100}
    expected = {"foo": {"bar": 100, "baz": 20}}
    assert patch_with_permutation(old_dict, permutation_dict) == expected


def test_generate_permutations(example_yaml):
    results = generate_permutations(example_yaml, ConfigModel)
    assert isinstance(results, list)

    # Calculate expected permutations
    expected_permutation_count = len(example_yaml["bar"]["baz"]) * len(example_yaml["bar"]["qux"])
    assert len(results) == expected_permutation_count

    for _, result_dict in enumerate(results):
        assert result_dict["foo"] == example_yaml["foo"]  # 'foo' should remain unchanged
        assert result_dict["bar"]["baz"] in example_yaml["bar"]["baz"]
        assert result_dict["bar"]["qux"] in example_yaml["bar"]["qux"]
