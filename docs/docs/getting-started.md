---
sidebar_label: "Getting Started"
sidebar_position: 1
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Getting Started

LLM Finetuning toolkit is a config-based CLI tool for launching a series of finetuning experiments and gathering their results. From one single `yaml` config file, you can define the following:

**Data**

- Bring your own dataset in any of `json`, `csv`, and `huggingface` formats
- Define your own prompt format and inject desired columns into the prompt

**Fine Tuning**

- Configure desired hyperparameters for quantization and LoRA fine-tune.

**Ablation**

- Intuitively define multiple hyperparameter settings to iterate through

**Inference**

- Configure desired sampling algorithm and parameters

**Testing**

- Test desired properties such as length and similarity against reference text

## Content

This documentation page is organized in the following sections:

- **Quick Start** provides a quick overview of the toolkit and helps you get started running your own experiments
- **Configuration** walks you through all the changes that can be made to customize your experiments
- **Tutorials** guides you through each component of this toolkit, what they are doing, and artifacts that you may expect it to produce
- **Developer Guides** goes over how to extend each component for custom use-cases and for contributing to this toolkit
- **API Reference** details the underlying modules of this toolkit
- **Research** details our findings / tables thus far & blog posts

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
