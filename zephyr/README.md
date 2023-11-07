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
[Zephyr](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta) models are specifically tailored to function as a helpful assistant. It is an enhanced iteration of Mistral-7B, refined using Direct Preference Optimization (DPO) on a combination of public and synthetic datasets. Notably, the model demonstrates improved performance on MT Bench, resulting in a more helpful output. The authors [report](https://arxiv.org/abs/2310.16944) SOTA results on MT-Bench even compared with models that have much higher parameter counts (40B-70B).


## Variations of Zephyr and Parameters

Zephyr models come in two variations, and can be leveraged depending on the task at hand.

| Zephyr Variant   | Parameters  |
|:----------------:|:-----------:|
|alpha (α)         |7B           |
|beta (β)          |7B           |

Beta variant is newer and more performant. In this repository, we have experimented with the 7B-β variation. 

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
	* Training with [NEFTune](https://arxiv.org/abs/2310.05914) vs without
	* Tuning only attention modules (default for `peft` library) vs all modules
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

|Training samples (fraction) | Zephyr-7B-β     | Zephyr-7B-β w/ NEFTune  | Zephyr-7B-β w/ Full Module Tuning | Zephyr-7B-β w/ NEFTune + Full Module Tuning |
|:--------------------------:|:---------------:|:-----------------------:|:---------------------------------:|:-------------------------------------------:|
|266   (2.5%)                |46.05            |49.61                    |65.36                              |67.23                                        |
|533   (5%)                  |55.66            |60.33                    |72.26                              |72.94                                        |
|1066  (10%)                 |66.48            |64.65                    |73.29                              |72.82                                        |
|2666  (25%)                 |66.73            |68.04                    |74.27                              |75.85                                        |
|5332  (50%)                 |69.54            |72.10                    |74.83                              |74.40                                        |
|10664 (100%)                |74.90            |72.93                    |77.76                              |77.86                                        |

- Zephyr performance is roughly in-line with that of its base model, Mistral; however, we note that the performance tends to converge faster
- NEFTune tends to help model training when there is few examples; however as training set size increases, the performance is the same as non-NEFTune
- Tuning on all modules (attention + linear) makes the model converge much faster


#### Summarization ####

<u> Table 2: Zero-Shot prompting vs Few-Shot prompting vs Fine-Tuning QLoRA </u>

|Method         | Zephyr-7B-β Zero-Shot | Zephyr-7B-β Few-Shot | Fine-Tuning + QLoRA | Fine-Tuning + QLoRA + NEFTune  | Fine-Tuning + QLoRA + Full Module Tuning | Fine-Tuning + QLoRA + NEFTune + Full Module Tuning | 
|:-------------:|:---------------------:|:--------------------:|:-------------------:|:------------------------------:|:----------------------------------------:|:--------------------------------------------------:|
|ROUGE-1 (in %) |33.93                  |35.99                 |52.84                |52.97                           | 53.50                                    | 53.05                                              |
|ROUGE-2 (in %) |11.21                  |12.97                 |27.75                |28.44                           | 29.66                                    | 29.23                                              |

- Zephyr performance is roughly in-line with Mistral but slightly underperforms
- Few-shot approach only yields slight improvement in ROUGE metrics over zero-shot
- Fine-tuning works the best, but we note that using NEFTune and tuning on all modules only yield marginal performance improvements


<u> Table 3: Zephyr vs Other LLMs </u>

|Model          | Flan-T5-Base Full Fine-Tune | Flan-T5-Large | Falcon-7B | RP-3B | RP-7B | Llama2-7B | Llama2-13B | Mistral-7B | Zephyr-7B-β  |
|:-------------:|:---------------------------:|:-------------:|:---------:|:-----:|:-----:|:---------:|:----------:|:----------:|:------------:|
|ROUGE-1 (in %) |47.23                        |49.21          |52.18      |47.75  |49.96  |51.71      |52.97       |53.61       |52.84         |
|ROUGE-2 (in %) |21.01                        |23.39          |27.84      |23.53  |25.94  |26.86      |28.32       |29.28       |28.44	       |	

- Zephyr achieves results comparable to Mistral, which is the best among 7B parameter models
