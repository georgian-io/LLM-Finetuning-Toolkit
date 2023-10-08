# Contents:

- [Contents:](#contents)
	- [What is MPT?](#what-is-mpt)
	- [Variations of MPT and Parameters](#variations-of-mpt-and-parameters)
	- [What does this folder contain?](#what-does-this-folder-contain)
	- [Evaluation Framework](#evaluation-framework)
		- [ Performance ](#-performance-)
			- [Classification](#classification)
			- [Summarization](#summarization)
	

## What is MPT? 

MPT-7B is a decoder-style transformer pretrained from scratch on 1T tokens of English text and code. This model was trained by MosaicML. MPT-7B is part of the family of MosaicPretrainedTransformer (MPT) models, which use a modified transformer architecture optimized for efficient training and inference. [Source](https://huggingface.co/mosaicml/mpt-7b)


## Variations of MPT and Parameters

MPT models come in two sizes, and can be leveraged depending on the task at hand.

| MPT variation    | Parameters  |
|:----------------:|:-----------:|
|Base              |7B           |
|Instruct          |7B           | 
|StoryWriter       |7B           | 
|Chat              |7B           | 

In this repository, we have experimented with the Base 7B variation. 

## What does this folder contain? 

This folder contains ready-to-use scripts, using which you can do the following:
	
* Finetuning MPT using PeFT methodology QLoRA:
	* ```mpt_classification.py```: Finetune on News Group classification dataset
	* ```mpt_summarization.py```: Finetune on Samsum summarization dataset
* Prompts used:
	* ```prompts.py```: Zero-shot, Few-shot and instruction tuning for classification and summarization
* Perform hyperparameter optimization over a well-constrained search space:
	* ```run_lora.sh```: Ablation study on LoRA's parameters 
	* ```sample_ablate.sh```: Ablation study over sample complexities
* Infer MPT using trained checkpoints:
	* ```mpt_baseline_inference.py```: Infer in zero-shot and few-shot settings using MPT-7B
	* ```mpt_classification_inference.py```: Infer on News Group classification dataset
	* ```mpt_summarization_inference.py```: Infer on Samsum summarization dataset
* Infer across a different settings:
	* ```baseline_inference.sh```: Loop over all settings to perform zero-shot and few-shot prompting across classification and summarization tasks

## Evaluation Framework


### <img src="../assets/rocket.gif" width="32" height="32"/> Performance <img src="../assets/rocket.gif" width="32" height="32"/>

We evaluated MPT under the following conditions:

* Tasks & Datasets:
	* Classification: News Group dataset, which is a 20-way classification task.
	* Summarization: Samsum dataset. 
* Experiments:
	* PeFT QLoRA (for classification)
	* Zero-Shot prompting vs Few-Shot prompting vs PeFT QLoRA (for summarization)
* Training config:
	* Epochs: 5 (for classification)
	* Epochs: 1 (for summarization)
	* MPT-7B:
		* PeFT technique: QLoRA
		* Learning rate: 2e-4
* Hardware:
	* Cloud provider: AWC EC2
	* Instance: g5.2xlarge
	
#### Classification ####

<u> Table 1: Sample Efficiency vs Accuracy </u>

|Training samples (fraction) | MPT-7B |
|:--------------------------:|:------:|
|10664 (100%)                |0.0     |


MPT-7B does not produce the right labels. It tends to also generate a lot of additional text, which causes accuracy to suffer. Since MPT-7B is unable to learn despite being trained on the entire training set, we do not perform the ablation study on sample complexity vs performance. 


#### Summarization ####

<u> Table 2: Zero-Shot prompting vs Few-Shot prompting vs Fine-Tuning QLoRA </u>

|Method         | MPT-7B Zero-Shot  | MPT-7B Few-Shot  | Fine-Tuning + QLoRA |
|:-------------:|:-----------------:|:----------------:|:-------------------:|
|ROUGE-1 (in %) |32.86              |34.71             |23.5                 |
|ROUGE-2 (in %) |10.41              |12.26             |9.67                 |


Looking at the ROUGE-1 and ROUGE-2 scores, we see that MPT-7B’s performance increases from zero-shot to few-shot. However, the performance drops when MPT-7B is fine-tuned on the Samsum dataset. This finding is aligned with what we see on the classification task. After being fine-tuned, MPT-7B's performance drops across both tasks, which is a peculiar behaviour.


<u> Table 3: MPT vs Other LLMs </u>

|Model          | Flan-T5-Base Full Fine-Tune | Flan-T5-Large | Falcon-7B | RP-3B | RP-7B | Llama2-7B | Llama2-13B | Mistral-7B | MPT-7B |
|:-------------:|:---------------------------:|:-------------:|:---------:|:-----:|:-----:|:---------:|:----------:|:----------:|:------:|
|ROUGE-1 (in %) |47.23                        |49.21          |52.18      |47.75  |49.96  |51.71      |52.97       |53.61       |23.5    |
|ROUGE-2 (in %) |21.01                        |23.39          |27.84      |23.53  |25.94  |26.86      |28.32       |29.28       |9.67    |


MPT-7B gets the lowest results, even when compared with models that are smaller than itself, i.e., Flan-T5-Large and  RP-3B. In our opinion, other 7B models are better choices to consider for fine-tuning purposes.


