---
sidebar_label: Utilities
sidebar_position: 6
---

# Utilities

## Save Utils

### DirectoryList

```python
class src.utils.save_utils.DirectoryList()
```

> The `DirectoryList` class represents a list of directories used for saving experiment results.

### DirectoryHelper

```python
class src.utils.save_utils.DirectoryHelper(config_path: str, config: Config)
"""
config_path: The path to the configuration file.
config: The configuration object.
"""
```

> The `DirectoryHelper` class provides helper methods for managing directories and saving configurations.
>
> #### Methods
>
> ```python
> save_config(self) -> None
> ```
>
> > Saves the configuration to a file.

## Ablation Utils

```python
src.utils.ablation_utils
```

> This module contains utility functions for ablation studies.
>
> #### Functions
>
> ```python
> get_types_from_dict(source_dict: dict, root="", type_dict={}) -> Dict[str, Tuple[type, type]]
> """
> source_dict: The source dictionary.
> root: The current root key (used for recursion).
> type_dict: The dictionary to store the types (used for recursion).
> """
> ```
>
> > Recursively retrieves the types of values in a nested dictionary.
> >
> > **Returns:** A dictionary mapping keys to their corresponding types.
>
> ```python
> get_annotation(key: str, base_model)
> """
> key: The key to retrieve the annotation for.
> base_model: The base model object.
> """
> ```
>
> > Retrieves the annotation for a given key in the base model.
> >
> > **Returns:** The annotation for the key.
>
> ```python
> get_model_field_type(annotation)
> """
> annotation: The annotation to retrieve the field type from.
> """
> ```
>
> > Retrieves the field type of a model based on the annotation.
> >
> > **Returns:** The field type.
>
> ```python
> get_data_with_key(key, data)
> """
> key: The key to retrieve the value for.
> data: The nested dictionary.
> """
> ```
>
> > Retrieves the value associated with a given key in a nested dictionary.
> >
> > **Returns:** The value associated with the key.
>
> ```python
> validate_and_get_ablations(type_dict, data, base_model)
> """
> type_dict: The dictionary mapping keys to their corresponding types.
> data: The data dictionary.
> base_model: The base model object.
> """
> ```
>
> > Validates and retrieves the ablations from the data based on the type dictionary and base model.
> >
> > **Returns:** A dictionary containing the validated ablations.
>
> ```python
> patch_with_permutation(old_dict, permutation_dict)
> """
> old_dict: The old dictionary to be patched.
> permutation_dict: The permutation dictionary containing the new values.
> """
> ```
>
> > Patches an old dictionary with values from a permutation dictionary.
> >
> > **Returns:** The patched dictionary.
>
> ```python
> generate_permutations(yaml_dict, model)
> """
> yaml_dict: The YAML dictionary containing the ablations.
> model: The model object.
> """
> ```
>
> > Generates permutations of the YAML dictionary based on the specified ablations.
> >
> > **Returns:** A list of permuted dictionaries.
