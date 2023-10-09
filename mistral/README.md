Important: Mistral has the option of using flash attention to speed up inference. In order to use flash attention, please do:

```shell
pip install -U flash-attn --no-build-isolation
```

```shell
git clone https://github.com/huggingface/peft
cd peft
pip install .
```

# Contents:

- [Contents:](#contents)
	- [What is Mistral?](#what-is-mistral)
	- [Variations of Mistral and Parameters](#variations-of-mistral-and-parameters)
	- [What does this folder contain?](#what-does-this-folder-contain)
	- [Evaluation Framework](#evaluation-framework)
		- [ Performance ](#-performance-)
			- [Classification](#classification)
			- [Summarization](#summarization)
	

## What is Mistral? 

Mistral-7B-v0.1 is Mistral AI’s first Large Language Model (LLM). Mistral-7B-v0.1 is a decoder-based LM with the following architectural choices: (i) Sliding Window Attention, (ii) GQA (Grouped Query Attention) and (iii) Byte-fallback BPE tokenizer.


## Variations of Mistral and Parameters

Mistral models come in two sizes, and can be leveraged depending on the task at hand.

| Mistral variation | Parameters  |
|:----------------:|:-----------:|
|Base              |7B           |
|Instruct          |7B           |           

In this repository, we have experimented with the Base 7B variation. 

## What does this folder contain? 

This folder contains ready-to-use scripts, using which you can do the following:
	
* Finetuning Mistral using PeFT methodology QLoRA:
	* ```mistral_classification.py```: Finetune on News Group classification dataset
	* ```mistral_summarization.py```: Finetune on Samsum summarization dataset
* Prompts used:
	* ```prompts.py```: Zero-shot, Few-shot and instruction tuning for classification and summarization
* Perform hyperparameter optimization over a well-constrained search space:
	* ```run_lora.sh```: Ablation study on LoRA's parameters 
	* ```sample_ablate.sh```: Ablation study over sample complexities
* Infer Mistral using trained checkpoints:
	* ```mistral_baseline_inference.py```: Infer in zero-shot and few-shot settings using Mistral-7B
	* ```mistral_classification_inference.py```: Infer on News Group classification dataset
	* ```mistral_summarization_inference.py```: Infer on Samsum summarization dataset
* Infer across a different settings:
	* ```baseline_inference.sh```: Loop over all settings to perform zero-shot and few-shot prompting across classification and summarization tasks

## Evaluation Framework


### <img src="../assets/rocket.gif" width="32" height="32"/> Performance <img src="../assets/rocket.gif" width="32" height="32"/>

We evaluated Mistral under the following conditions:

* Tasks & Datasets:
	* Classification: News Group dataset, which is a 20-way classification task.
	* Summarization: Samsum dataset. 
* Experiments:
	* Sample Efficiency vs Accuracy
	* Zero-Shot prompting vs Few-Shot prompting vs PeFT QLoRA (for summarization)
* Training config:
	* Epochs: 5 (for classification)
	* Epochs: 1 (for summarization)
	* Mistral-7B:
		* PeFT technique: QLoRA
		* Learning rate: 2e-4
* Hardware:
	* Cloud provider: AWC EC2
	* Instance: g5.2xlarge
	
#### Classification ####

<u> Table 1: Sample Efficiency vs Accuracy </u>

|Training samples (fraction) | Mistral Base-7B |
|:--------------------------:|:---------------:|
|266   (2.5%)                |49.30            |
|533   (5%)                  |48.14            |
|1066  (10%)                 |58.41            |
|2666  (25%)                 |64.89            |
|5332  (50%)                 |73.10            |
|10664 (100%)                |74.36            |


The above table shows how performance of Mistral-7B track with the number of training samples. The last row of the table demonstrates the performance when the entire dataset is used.  



#### Summarization ####

<u> Table 2: Zero-Shot prompting vs Few-Shot prompting vs Fine-Tuning QLoRA </u>

|Method         | Mistral-7B Zero-Shot  | Mistral-7B Few-Shot  | Fine-Tuning + QLoRA |
|:-------------:|:---------------------:|:--------------------:|:-------------------:|
|ROUGE-1 (in %) |32.77                  |38.87                 |53.61                |
|ROUGE-2 (in %) |10.64                  |16.71                 |29.28                |


Looking at the ROUGE-1 and ROUGE-2 scores, we see that Mistral-7B’s performance increases from zero-shot to few-shot to fine-tuning settings. 


<u> Table 3: Mistral vs Other LLMs </u>

|Model          | Flan-T5-Base Full Fine-Tune | Flan-T5-Large | Falcon-7B | RP-3B | RP-7B | Llama2-7B | Llama2-13B | Mistral-7B |
|:-------------:|:---------------------------:|:-------------:|:---------:|:-----:|:-----:|:---------:|:----------:|:----------:|
|ROUGE-1 (in %) |47.23                        |49.21          |52.18      |47.75  |49.96  |51.71      |52.97       |53.61       |
|ROUGE-2 (in %) |21.01                        |23.39          |27.84      |23.53  |25.94  |26.86      |28.32       |29.28       |


Mistral-7B achieves the best results, even when compared with Falcon-7B and Llama2-7B. This makes Mistral-7B, in our opinion, the best model to leverage in the 7B parameter space.


