Important: Zephyr has the option of using flash attention to speed up inference (since it uses Mistral as its base model). In order to use flash attention, please do:

```shell
pip install -U flash-attn --no-build-isolation
```

```shell
git clone https://github.com/huggingface/peft
cd peft
pip install .
```

Important: To enable neftune, must install `trl` from source:
```shell
git clone https://github.com/huggingface/trl.git
cd trl/
pip install .
```

# Contents:

- [Contents:](#contents)
	- [What is Zephyr?](#what-is-zephyr)
	- [Variations of Zephyr and Parameters](#variations-of-zephyr-and-parameters)
	- [What does this folder contain?](#what-does-this-folder-contain)
	- [Evaluation Framework](#evaluation-framework)
		- [ Performance ](#-performance-)
			- [Classification](#classification)
			- [Summarization](#summarization)
	

## What is Zephyr? 

TODO

## Variations of Zephyr and Parameters

Zephyr models come in two variations, and can be leveraged depending on the task at hand.

| Zephyr Variant   | Parameters  |
|:----------------:|:-----------:|
|alpha (α)         |7B           |
|beta (β)          |7B           |           
In this repository, we have experimented with the 7B-β variation. 

## What does this folder contain? 

This folder contains ready-to-use scripts, using which you can do the following:
	

* Finetuning Zephyr using PeFT methodology QLoRA:
	* ```zephyr_classification.py```: Finetune on News Group classification dataset
	* ```zephyr_summarization.py```: Finetune on Samsum summarization dataset
* Prompts used:
	* ```prompts.py```: Zero-shot, Few-shot and instruction tuning for classification and summarization
* Perform hyperparameter optimization over a well-constrained search space:
	* ```run_lora.sh```: Ablation study on LoRA's parameters 
	* ```sample_ablate.sh```: Ablation study over sample complexities
* Infer Zephyr using trained checkpoints:
	* ```zephyr_baseline_inference.py```: Infer in zero-shot and few-shot settings using Zephyr-7B-α
	* ```zephyr_classification_inference.py```: Infer on News Group classification dataset
	* ```zephyr_summarization_inference.py```: Infer on Samsum summarization dataset
* Infer across a different settings:
	* ```baseline_inference.sh```: Loop over all settings to perform zero-shot and few-shot prompting across classification and summarization tasks

## Evaluation Framework


### <img src="../assets/rocket.gif" width="32" height="32"/> Performance <img src="../assets/rocket.gif" width="32" height="32"/>

We evaluated Zephyr under the following conditions:

* Tasks & Datasets:
	* Classification: News Group dataset, which is a 20-way classification task.
	* Summarization: Samsum dataset. 
* Experiments:
	* Sample Efficiency vs Accuracy
	* Zero-Shot prompting vs Few-Shot prompting vs PeFT QLoRA (for summarization)
* Training config:
	* Epochs: 5 (for classification)
	* Epochs: 1 (for summarization)
	* Zephyr-7B-β:
		* PeFT technique: QLoRA
		* Learning rate: 2e-4
* Hardware:
	* Cloud provider: AWC EC2
	* Instance: g5.2xlarge
	
#### Classification ####

<u> Table 1: Sample Efficiency vs Accuracy </u>

|Training samples (fraction) | Zephyr-7B-β     | Zephyr-7B-β w/ NEFTune  |
|:--------------------------:|:---------------:|:-----------------------:|
|266   (2.5%)                |~~49.30~~        | -                       |
|533   (5%)                  |~~48.14~~        | -                       |
|1066  (10%)                 |~~58.41~~        | -                       |
|2666  (25%)                 |~~64.89~~        | -                       |
|5332  (50%)                 |~~73.10~~        | -                       |
|10664 (100%)                |~~74.36~~        | -                       |

TODO: Commentary


#### Summarization ####

<u> Table 2: Zero-Shot prompting vs Few-Shot prompting vs Fine-Tuning QLoRA </u>

|Method         | Zephyr-7B-β Zero-Shot | Zephyr-7B-β Few-Shot | Fine-Tuning + QLoRA | Fine-Tuning + QLoRA + NEFTune  |
|:-------------:|:---------------------:|:--------------------:|:-------------------:|:------------------------------:|
|ROUGE-1 (in %) |33.93                  |35.99                 |~~53.61~~            | -                              |
|ROUGE-2 (in %) |11.21                  |12.97                 |~~29.28~~            | -                              |

TODO: Commentary


<u> Table 3: Zephyr vs Other LLMs </u>

|Model          | Flan-T5-Base Full Fine-Tune | Flan-T5-Large | Falcon-7B | RP-3B | RP-7B | Llama2-7B | Llama2-13B | Mistral-7B | Zephyr-7B-β  |
|:-------------:|:---------------------------:|:-------------:|:---------:|:-----:|:-----:|:---------:|:----------:|:----------:|:------------:|
|ROUGE-1 (in %) |47.23                        |49.21          |52.18      |47.75  |49.96  |51.71      |52.97       |53.61       | -            |
|ROUGE-2 (in %) |21.01                        |23.39          |27.84      |23.53  |25.94  |26.86      |28.32       |29.28       | -	           |	

TODO: Commentary
