import copy
import itertools
from typing import List, Type, Any, Dict, Optional, Union, Tuple
from typing import get_args, get_origin, get_type_hints

import yaml


# TODO: organize this a little bit. It's a bit of a mess rn.

"""
Helper functions to create multiple valid configs based on ablation (i.e. list of values)
fron a single config yaml
"""


def get_types_from_dict(
    source_dict: dict, root="", type_dict={}
) -> Dict[str, Tuple[type, type]]:
    for key, val in source_dict.items():
        if type(val) is not dict:
            attr = f"{root}.{key}" if root else key
            tp = (
                (type(val), None)
                if type(val) is not list
                else (type(val), type(val[0]))
            )
            type_dict[attr] = tp
        else:
            join_array = [root, key] if root else [key]
            new_root = ".".join(join_array)
            get_types_from_dict(val, new_root, type_dict)

    return type_dict


def get_annotation(key: str, base_model):
    keys = key.split(".")
    model = base_model
    for key in keys:
        model = model.__annotations__[key]

    return model


def get_model_field_type(annotation):
    origin = get_origin(annotation)
    if not origin:
        return annotation
    if origin is Union:
        annotations = get_args(annotation)[0]
        return get_model_field_type(annotations)
    if origin is list:
        return list


def get_data_with_key(key, data):
    keys = key.split(".")
    for key in keys:
        data = data[key]
    return data


def validate_and_get_ablations(type_dict, data, base_model):
    ablations = {}
    for key, (tp, subtype) in type_dict.items():
        annotation = get_annotation(key, base_model)
        model_field_type = get_model_field_type(annotation)
        if (model_field_type is list) and (tp is list) and (subtype is list):
            # Handle both list and list of lists
            ablations[key] = get_data_with_key(key, data)
        elif model_field_type is not list and tp is list:
            # Handle single-level lists
            ablations[key] = get_data_with_key(key, data)

    return ablations


def patch_with_permutation(old_dict, permutation_dict):
    # Create a deep copy of the old dictionary to avoid modifying the original
    updated_dict = copy.deepcopy(old_dict)

    # Iterate over each item in the permutation dictionary
    for dot_key, new_value in permutation_dict.items():
        # Split the dot-joined key into individual keys
        keys = dot_key.split(".")

        # Start from the root of the updated dictionary
        current_level = updated_dict

        # Traverse to the second-to-last key in the nested dictionary
        for key in keys[:-1]:
            current_level = current_level[key]

        # Update the value at the final key
        current_level[keys[-1]] = new_value

    return updated_dict


def generate_permutations(yaml_dict, model):
    type_dict = get_types_from_dict(yaml_dict)

    ablations = validate_and_get_ablations(type_dict, yaml_dict, model)

    # get permutations
    lists = list(ablations.values())
    permutations = list(itertools.product(*lists))

    permutation_dicts = []
    for perm in permutations:
        new_dict = dict(zip(ablations.keys(), perm))
        permutation_dicts.append(new_dict)

    new_dicts = []
    for perm in permutation_dicts:
        new_dicts.append(patch_with_permutation(yaml_dict, perm))

    return new_dicts
