---
sidebar_position: 1
---

import arch from "./img/arch.png"

# Extending Modules

The toolkit provides a modular and extensible architecture that allows developers to customize and enhance its functionality to suit their specific needs. Each component of the toolkit, such as data ingestion, finetuning, inference, and quality assurance testing, is designed to be easily extendable.

## General Guidelines

<img src={arch} width="1000"/>

There are various scenarios where you might want to extend a particular module of the toolkit. For example:

1. **Data Ingestion**: If you have a custom data format or source that is not supported out of the box, you can extend the `Ingestor` class to handle your specific data format. For instance, if you have data stored in a proprietary binary format, you can create a new subclass of `Ingestor` that reads and processes your binary data and converts it into a compatible format for the toolkit.
2. **Finetuning**: If you want to experiment with different finetuning techniques or modify the finetuning process, you can extend the `Finetune` class. For example, if you want to incorporate a custom loss function or implement a new finetuning algorithm, you can create a subclass of `Finetune` and override the necessary methods to include your custom logic.
3. **Inference**: If you need to modify the inference process or add custom post-processing steps, you can extend the `Inference` class. For instance, if you want to apply domain-specific post-processing to the generated text or integrate the inference process with an external API, you can create a subclass of `Inference` and implement your custom functionality.
4. **Quality Assurance (QA) Testing**: If you have specific quality metrics or evaluation criteria that are not included in the existing QA tests, you can extend the `LLMQaTest` class to define your own custom tests. For example, if you want to evaluate the generated text based on domain-specific metrics or compare it against a custom benchmark, you can create a new subclass of `LLMQaTest` and implement your custom testing logic.

By extending the toolkit's components, you can tailor it to your specific requirements and incorporate custom functionality that is not provided by default. This flexibility allows you to adapt the toolkit to various domains, data formats, and evaluation criteria.

In the following sections, we will provide detailed guidance on how to extend each component of the toolkit, along with code examples and best practices. Whether you are a researcher exploring new finetuning techniques or a developer integrating the toolkit into a larger pipeline, the ability to extend and customize the modules will empower you to achieve your goals effectively.

## Extending Data Ingestor

To extend the data ingestor component, follow these steps:

1. Open the file `src/data/ingestor.py`.
2. Define a new class that inherits from the abstract base class `Ingestor`.
3. Implement the required abstract method `to_dataset` in your custom ingestor class. This method should load and preprocess the data from the specified source and return a `Dataset` object.
4. Update the `get_ingestor`` function to include your custom ingestor class based on a new file type or data source.

```python title="Example"
from src.data.ingestor import Ingestor

class CustomIngestor(Ingestor):
    def __init__(self, path):
        self.path = path

    def to_dataset(self):
        # Implement the logic to load and preprocess data from the specified path
        ...

def get_ingestor(data_type):
    if data_type == "custom":
        return CustomIngestor
    ...
```

## Extending Finetuning

To extend the finetuning component, follow these steps:

1. Create a new file in the `src/finetune` directory, e.g., `custom_finetune.py`.
2. In this file, define a new class that inherits from the abstract base class `Finetune` from `src/finetune/finetune.py`.
3. Implement the required abstract methods `finetune` and `save_model` in your custom finetuning class.
4. The `finetune` method should take the training dataset and perform the finetuning process using the provided configuration.
5. The `save_model` method should save the finetuned model to the specified directory.
6. Modify the `toolkit.py` file to import your custom finetuning class and use it instead of the default `LoRAFinetune` class if needed.

```python title="Example"
from src.finetune.finetune import Finetune

class CustomFinetune(Finetune):
    def finetune(self, train_dataset: Dataset):
        # Implement your custom finetuning logic here
        ...

    def save_model(self):
        # Implement the logic to save the finetuned model
        ...

```

## Extending Inference

To extend the inference component, follow these steps:

1. Create a new file in the `src/inference` directory, e.g., `custom_inference.py`.
2. In this file, define a new class that inherits from the abstract base class `Inference` from `src/inference/inference.py`.
3. Implement the required abstract methods `infer_one` and `infer_all` in your custom inference class.
4. The `infer_one` method should take a single prompt and generate the model's prediction.
5. The `infer_all` method should iterate over the test dataset and generate predictions for each example.
6. Modify the `toolkit.py` file to import your custom inference class and use it instead of the default `LoRAInference` class if needed.

```python title="Example"
from src.inference.inference import Inference

class CustomInference(Inference):
    def infer_one(self, prompt: str):
        # Implement the logic to generate a prediction for a single prompt
        ...

    def infer_all(self):
        # Implement the logic to generate predictions for the entire test dataset
        ...
```

## Extending QA Test

To extend the quality assurance (QA) tests, follow these steps:

1. Open the file `src/qa/qa_tests.py`.
2. Define a new class that inherits from the abstract base class `LLMQaTest` from `src/qa/qa.py`.
3. Implement the required abstract property `test_name` and the abstract method `get_metric` in your custom QA test class.
4. The `test_name` property should return a string representing the name of the test.
5. The `get_metric` method should take the prompt, ground truth, and model prediction, and return a metric value (e.g., `float`, `int`, or `bool`) indicating the test result.
6. Include instance of new `CustomQATest` when instantiating `LLMTestSuite` object.

```python title="Example"
from src.qa.qa import LLMQaTest

class CustomQATest(LLMQaTest):
    @property
    def test_name(self):
        return "Custom QA Test"

    def get_metric(self, prompt, ground_truth, model_prediction):
        # Implement the logic to calculate the metric for the custom QA test
        ...


test_suite = LLMTestSuite([JaccardSimilarityTest(), CustomQATest()], prompts, ground_truths, model_preds)
```
