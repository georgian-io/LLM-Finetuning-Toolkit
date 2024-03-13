---
sidebar_label: Quality Assurance
sidebar_position: 4
---

# Quality Assurance

## Quality Assurance Tests

### LLMQaTest

```python
class src.qa.qa.LLMQaTest()
```

> The `LLMQaTest` class is an abstract base class for defining quality assurance tests for language models.
>
> #### Methods
>
> ```python
> test_name(self) -> str
> ```
>
> > An abstract property to be implemented by subclasses. Returns the name of the test.
>
> ```python
> get_metric(self, prompt: str, ground_truth: str, model_pred: str) -> Union[float, int, bool]
> """
> prompt: The input prompt.
> ground_truth: The ground truth output.
> model_pred: The model's predicted output.
> """
> ```
>
> > An abstract method to be implemented by subclasses. Computes the metric for the test.
> >
> > **Returns:** The computed metric.

## Test Runner

### LLMTestSuite

```python
class src.qa.qa.LLMTestSuite(tests: List[LLMQaTest], prompts: List[str], ground_truths: List[str], model_preds: List[str])
"""
tests: A list of LLMQaTest objects representing the tests to run.
prompts: A list of input prompts.
ground_truths: A list of ground truth outputs.
model_preds: A list of model's predicted outputs.
"""
```

> The `LLMTestSuite` class represents a suite of quality assurance tests for language models.
>
> #### Methods
>
> ```python
> run_tests(self) -> Dict[str, List[Union[float, int, bool]]]
> ```
>
> > Runs all the tests in the suite and returns the results.
> >
> > **Returns:** A dictionary mapping test names to their corresponding metrics.
>
> ```python
> print_test_results(self) -> None
> ```
>
> > Prints the test results in a tabular format.
>
> ```python
> save_test_results(self, path: str) -> None
> """
> path: The path to save the CSV file.
> """
> ```
>
> > Saves the test results to a CSV file.
