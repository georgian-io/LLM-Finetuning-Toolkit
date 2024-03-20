---
sidebar_position: 4
---

# Quality Assurance

:::warning
We are still in the process of finalizing the interface for QA. Please check back for updates.
:::

To run QA tests on a custom `results.csv` file, follow these steps:

1. Create instances of the desired QA test classes from `src/qa/qa_tests.py`, such as `LengthTest`, `JaccardSimilarityTest`, etc.
2. Create an instance of the `LLMTestSuite` class from src/qa/qa.py, passing the list of QA test instances, along with the prompts, ground truths, and model predictions obtained from the `results.csv` file.
3. Call the run_tests method on the LLMTestSuite instance to execute the QA tests.
4. Optionally, call the `print_test_results` method to display the test results or the `save_test_results` method to save the results to a file.

```python title="Example"
from src.qa.qa import LLMTestSuite
from src.qa.qa_tests import LengthTest, JaccardSimilarityTest

# Load prompts, ground_truths, and model_preds from the results.csv file

tests = [LengthTest(), JaccardSimilarityTest()]
test_suite = LLMTestSuite(tests, prompts, ground_truths, model_preds)
test_suite.run_tests()
test_suite.print_test_results()
test_suite.save_test_results("path/to/save/test_results.csv")
```

The test results will be displayed or saved based on the chosen method.
