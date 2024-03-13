---
sidebar_label: "Getting Started"
sidebar_position: 1
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

import features from "./img/features.png"

# Getting Started

LLM Finetuning toolkit is a config-based CLI tool for launching a series of finetuning experiments and gathering their results. From one single `yaml` config file, you can define the following:

<img src={features} width="800" />

- **Data**
  - Bring your own dataset in any of `json`, `csv`, and `huggingface` formats
  - Define your own prompt format and inject desired columns into the prompt
- **Fine Tuning**
  - Configure desired hyperparameters for quantization and LoRA fine-tune.
- **Ablation**
  - Intuitively define multiple hyperparameter settings to iterate through
- **Inference**
  - Configure desired sampling algorithm and parameters
- **Testing**
  - Test desired properties such as length and similarity against reference text

## Content

This documentation page is organized in the following sections:

- **[Quick Start](category/quick-start)** provides a quick overview of the toolkit and helps you get started running your own experiments
- **[Configuration](category/configuration)** walks you through all the changes that can be made to customize your experiments
- **[Tutorials](category/tutorials)** guides you through each component of this toolkit, what they are doing, and artifacts that you may expect it to produce
- **[Developer Guides](category/developer-guides)** goes over how to extend each component for custom use-cases and for contributing to this toolkit
- **[API Reference](category/api-reference)** details the underlying modules of this toolkit
- **[Research](category/research)** documents our fine-tune benchmarks for various commercial/open models

## Installation

### Clone Repository

```bash
git clone https://github.com/georgian-io/LLM-Finetuning-Hub.git
cd LLM-Finetuning-Hub/
```

### Install CLI

<Tabs>
<TabItem value="docker" label="docker (recommended)" default>
```bash
# build image
docker build -t llm-toolkit .
# launch container
docker run -it llm-toolkit              # with CPU
docker run -it --gpus all llm-toolkit   # with GPU
```

</TabItem>
<TabItem value="poetry" label="poetry (recommended)">
First, [make sure poetry is installed](https://python-poetry.org/docs/)

Then run:

```bash
poetry install
```

</TabItem>
<TabItem value="pip" label="pip">
```bash
pip install -r requirements.txt
```
</TabItem>
<TabItem value="conda" label="conda">
```bash
conda create --name llm-toolkit python=3.11
conda activate llm-toolkit
pip install -r requirements.txt
```
</TabItem>
</Tabs>

## Quick Start

The toolkit has everything you need to get started. This guide will walk you through the initial setup, explain the key components of the configuration, and offer advice on customizing your fine-tuning job. Let's dive in!

First, make sure you have read the installation guide above and installed all the dependencies. Then, To launch a LoRA fine-tuning job, run the following command in your terminal:

```bash
python3 toolkit.py
```

This command initiates the fine-tuning process using the settings specified in the default YAML configuration file `config.yaml`.

```yaml
save_dir: "./experiment/"

ablation:
  use_ablate: false

# Data Ingestion -------------------
data:
  file_type: "huggingface" # one of 'json', 'csv', 'huggingface'
  path: "yahma/alpaca-cleaned"
  prompt:
    >- # prompt, make sure column inputs are enclosed in {} brackets and that they match your data
    Below is an instruction that describes a task. 
    Write a response that appropriately completes the request. 
    ### Instruction: {instruction}
    ### Input: {input}
    ### Output:
  prompt_stub:
    >- # Stub to add for training at the end of prompt, for test set or inference, this is omitted; make sure only one variable is present
    {output}
  test_size: 0.1 # Proportion of test as % of total; if integer then # of samples
  train_size: 0.9 # Proportion of train as % of total; if integer then # of samples
  train_test_split_seed: 42

# Model Definition -------------------
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
  lora_alpha: 16
  lora_dropout: 0.1
  target_modules:
    - q_proj
    - v_proj
    - k_proj
    - o_proj
    - up_proj
    - down_proj
    - gate_proj

# Training -------------------
training:
  training_args:
    num_train_epochs: 5
    per_device_train_batch_size: 4
    optim: "paged_adamw_32bit"
    learning_rate: 2.0e-4
    bf16: true # Set to true for mixed precision training on Newer GPUs
    tf32: true
  sft_args:
    max_seq_length: 1024

inference:
  max_new_tokens: 1024
  do_sample: True
  top_p: 0.9
  temperature: 0.8
```
