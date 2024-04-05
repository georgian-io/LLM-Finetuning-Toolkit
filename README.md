# LLM Finetuning Toolkit

<p align="center">
  <img src="https://github.com/georgian-io/LLM-Finetuning-Toolkit/blob/main/assets/toolkit-animation.gif?raw=true" width="900" />
</p>

## Overview

LLM Finetuning toolkit is a config-based CLI tool for launching a series of LLM finetuning experiments on your data and gathering their results. From one single `yaml` config file, control all elements of a typical experimentation pipeline - **prompts**, **open-source LLMs**, **optimization strategy** and **LLM testing**.

<p align="center">
<img src="https://github.com/georgian-io/LLM-Finetuning-Toolkit/blob/main/assets/overview_diagram.png?raw=true" width="900" />
</p>

## Installation
### pipx (recommended)
pipx installs the package and depdencies in a seperate virtual environment
```shell
pipx install llm-toolkit
```

### pip
```shell
pip install llm-toolkit
```


## Quick Start

This guide contains 3 stages that will enable you to get the most out of this toolkit!

- **Basic**: Run your first LLM fine-tuning experiment
- **Intermediate**: Run a custom experiment by changing the componenets of the YAML configuration file
- **Advanced**: Launch series of fine-tuning experiments across different prompt templates, LLMs, optimization techniques -- all through **one** YAML configuration file

### Basic

```python
   llmtune --config-path ./config.yml
```

This command initiates the fine-tuning process using the settings specified in the default YAML configuration file `config.yaml`.

### Intermediate

The configuration file is the central piece that defines the behavior of the toolkit. It is written in YAML format and consists of several sections that control different aspects of the process, such as data ingestion, model definition, training, inference, and quality assurance. We highlight some of the critical sections.

#### Data Ingestion

An example of what the data ingestion may look like:

```yaml
data:
  file_type: "huggingface"
  path: "yahma/alpaca-cleaned"
  prompt:
    ### Instruction: {instruction}
    ### Input: {input}
    ### Output:
  prompt_stub: { output }
  test_size: 0.1 # Proportion of test as % of total; if integer then # of samples
  train_size: 0.9 # Proportion of train as % of total; if integer then # of samples
  train_test_split_seed: 42
```

- While the above example illustrates using a public dataset from Hugging Face, the config file can also ingest your own data.

```yaml
   file_type: "json"
   path: "<path to your data file>
```

```yaml
   file_type: "csv"
   path: "<path to your data file>
```

- The prompt fields help create instructions to fine-tune the LLM on. It reads data from specific columns, mentioned in {} brackets, that are present in your dataset. In the example provided, it is expected for the data file to have column names: `instruction`, `input` and `output`.

- The prompt fields use both `prompt` and `prompt_stub` during fine-tuning. However, during testing, **only** the `prompt` section is used as input to the fine-tuned LLM.

#### LLM Definition

```yaml
model:
  hf_model_ckpt: "NousResearch/Llama-2-7b-hf"
  quantize: true
  bitsandbytes:
    load_in_4bit: true
    bnb_4bit_compute_dtype: "bf16"
    bnb_4bit_quant_type: "nf4"

# LoRA Params -------------------
lora:
  task_type: "CAUSAL_LM"
  r: 32
  lora_dropout: 0.1
  target_modules:
    - q_proj
    - v_proj
    - k_proj
    - o_proj
    - up_proj
    - down_proj
    - gate_proj
```

- While the above example showcases using Llama2 7B, in theory, any open-source LLM supported by Hugging Face can be used in this toolkit.

```yaml
hf_model_ckpt: "mistralai/Mistral-7B-v0.1"
```

```yaml
hf_model_ckpt: "tiiuae/falcon-7b"
```

- The parameters for LoRA, such as the rank `r` and dropout, can be altered.

```yaml
lora:
  r: 64
  lora_dropout: 0.25
```

#### Quality Assurance

```yaml
qa:
  llm_tests:
    - length_test
    - word_overlap_test
```

- To ensure that the fine-tuned LLM behaves as expected, you can add tests that check if the desired behaviour is being attained. Example: for an LLM fine-tuned for a summarization task, we may want to check if the generated summary is indeed smaller in length than the input text. We would also like to learn the overlap between words in the original text and generated summary.

#### Artifact Outputs

This config will run finetuning and save the results under directory `./experiment/[unique_hash]`. Each unique configuration will generate a unique hash, so that our tool can automatically pick up where it left off. For example, if you need to exit in the middle of the training, by relaunching the script, the program will automatically load the existing dataset that has been generated under the directory, instead of doing it all over again.

After the script finishes running you will see these distinct artifacts:

```shell
  /dataset # generated pkl file in hf datasets format
  /model # peft model weights in hf format
  /results # csv of prompt, ground truth, and predicted values
  /qa # csv of test results: e.g. vector similarity between ground truth and prediction
```

Once all the changes have been incorporated in the YAML file, you can simply use it to run a custom fine-tuning experiment!

```python
   python toolkit.py --config-path <path to custom YAML file>
```

### Advanced

Fine-tuning workflows typically involve running ablation studies across various LLMs, prompt designs and optimization techniques. The configuration file can be altered to support running ablation studies.

- Specify different prompt templates to experiment with while fine-tuning.

```yaml
data:
  file_type: "huggingface"
  path: "yahma/alpaca-cleaned"
  prompt:
    - >-
      This is the first prompt template to iterate over
      ### Input: {input}
      ### Output:
    - >-
      This is the second prompt template
      ### Instruction: {instruction}
      ### Input: {input}
      ### Output:
  prompt_stub: { output }
  test_size: 0.1 # Proportion of test as % of total; if integer then # of samples
  train_size: 0.9 # Proportion of train as % of total; if integer then # of samples
  train_test_split_seed: 42
```

- Specify various LLMs that you would like to experiment with.

```yaml
model:
  hf_model_ckpt:
    [
      "NousResearch/Llama-2-7b-hf",
      mistralai/Mistral-7B-v0.1",
      "tiiuae/falcon-7b",
    ]
  quantize: true
  bitsandbytes:
    load_in_4bit: true
    bnb_4bit_compute_dtype: "bf16"
    bnb_4bit_quant_type: "nf4"
```

- Specify different configurations of LoRA that you would like to ablate over.

```yaml
lora:
  r: [16, 32, 64]
  lora_dropout: [0.25, 0.50]
```

## Extending

The toolkit provides a modular and extensible architecture that allows developers to customize and enhance its functionality to suit their specific needs. Each component of the toolkit, such as data ingestion, finetuning, inference, and quality assurance testing, is designed to be easily extendable.

## Contributing

If you would like to contribute to this project, we recommend following the "fork-and-pull" Git workflow.

1.  **Fork** the repo on GitHub
2.  **Clone** the project to your own machine
3.  **Commit** changes to your own branch
4.  **Push** your work back up to your fork
5.  Submit a **Pull request** so that we can review your changes

NOTE: Be sure to merge the latest from "upstream" before making a pull request!

### Set Up Dev Environment

<details>
<summary>1. Clone Repo</summary>
  
```shell
   git clone https://github.com/georgian-io/LLM-Finetuning-Toolkit.git
   cd LLM-Finetuning-Toolkit/
```

</details>

<details>
<summary>2. Install Dependencies</summary>
<details>
<summary>Install with Docker [Recommended]</summary>

```shell
   docker build -t llm-toolkit
```

```shell
   # CPU
   docker run -it llm-toolkit
   # GPU
   docker run -it --gpus all llm-toolkit
```
</details>

<details>
<summary>Poetry (recommended)</summary>

See poetry documentation page for poetry [installation instructions](https://python-poetry.org/docs/#installation)

```shell
   poetry install
```
</details>
<details>
<summary>pip</summary>
We recommend using a virtual environment like `venv` or `conda` for installation

```shell
   pip install -e .
```
</details>
</details>



### Checklist Before Pull Request (Optional)

1. Use `ruff check --fix` to check and fix lint errors
2. Use `ruff format` to apply formatting

NOTE: Ruff linting and formatting checks are done when PR is raised via Git Action. Before raising a PR, it is a good practice to check and fix lint errors, as well as apply formatting.


### Releasing


To manually release a PyPI package, please run: 

```shell
   make build-release
```

Note: Make sure you have pypi token for this [PyPI repo](https://pypi.org/project/llm-toolkit/).

