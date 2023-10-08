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
	- [What is Mistral?](#what-is-llama2)
	- [Variations of Mistral and Parameters](#variations-of-llama2-and-parameters)
	- [What does this folder contain?](#what-does-this-folder-contain)
	- [Evaluation Framework](#evaluation-framework)
		- [ Performance ](#-performance-)
			- [Classification](#classification)
			- [Summarization](#summarization)
		- [  Time \& Cost to Train  ](#--time--cost-to-train--)
		- [ Inference ](#-inference-)
			- [Llama-7B Classification](#llama-7b-classification)
			- [Llama-13B Classification](#llama-13b-classification)
			- [Llama-7B Summarization](#llama-7b-summarization)
			- [Llama-13B Summarization](#llama-13b-summarization)

## What is Mistral? 

Llama 2 is the latest addition to the open-source large language model that is released by Meta. Based on Meta’s benchmarking results, it is the best available open-source large language model that can also be used for commercial purposes. Llama 2 comes in 3 different versions: 7B, 13B, and 70B.

## Variations of Mistral and Parameters

Mistral models come in two sizes, and can be leveraged depending on the task at hand.

| Mistral variation | Parameters  |
|:----------------:|:-----------:|
|Base-7B           |7B           |
|Base-13B          |13B          |           
|Base-70B          |70B          |

In this repository, we have experimented with the 7B and 13B variations. 

## What does this folder contain? 

This folder contains ready-to-use scripts, using which you can do the following:
	
* Finetuning Mistral using PeFT methodology QLoRA:
	* ```llama2_classification.py```: Finetune on News Group classification dataset
	* ```llama2_summarization.py```: Finetune on Samsum summarization dataset
* Prompts used:
	* ```prompts.py```: Zero-shot, Few-shot and instruction tuning for classification and summarization
* Perform hyperparameter optimization over a well-constrained search space:
	* ```run_lora.sh```: Ablation study on LoRA's parameters 
	* ```sample_ablate.sh```: Ablation study over sample complexities
* Infer Mistral using trained checkpoints:
	* ```llama2_baseline_inference.py```: Infer in zero-shot and few-shot settings using Mistral-3B or 7B Instruct versions
	* ```llama2_classification_inference.py```: Infer on News Group classification dataset
	* ```llama2_summarization_inference.py```: Infer on Samsum summarization dataset
* Infer across a different settings:
	* ```baseline_inference.sh```: Loop over all settings to perform zero-shot and few-shot prompting across classification and summarization tasks

## Evaluation Framework

In this section, we bring to you our insights after extensively experimenting with Mistral-7B and 13B across different tasks. For a thorough evaluation, we need to evaluate the __four pillars__:

* Performance
* Cost to Train
* Time to Train
* Inference Costs


### <img src="../assets/rocket.gif" width="32" height="32"/> Performance <img src="../assets/rocket.gif" width="32" height="32"/>

We evaluated Mistral under the following conditions:

* Tasks & Datasets:
	* Classification: News Group dataset, which is a 20-way classification task.
	* Summarization: Samsum dataset. 
* Competing Models:
	* BERT-Base (110M parameters)
	* Distilbert (66M parameters)
	* Flan-T5 Large (780M parameters)
	* Falcon-7B (7B parameters)
	* RedPajama (3B / 7B parameters)
* Experiments:
	* Sample Efficiency vs Accuracy
	* Zero-Shot prompting vs Few-Shot prompting vs PeFT QLoRA
* Training config:
	* Epochs: 5
	* Mistral-7B/13B:
		* PeFT technique: QLoRA
		* Learning rate: 2e-4
	* RedPajama 3B/7B:
		* PeFT technique: QLoRA
		* Learning rate: 2e-4
	* Falcon-7B:
		* PeFT technique: QLoRA
		* Learning rate: 2e-4
	* Flan-T5 Large:
		* PeFT technique: LoRA
		* Learning rate: 1e-3
	* BERT/Distilbert:
		* Learning rate: 2e-5
* Hardware:
	* Cloud provider: AWC EC2
	* Instance: g5.2xlarge
	
#### Classification ####

<u> Table 1: Sample Efficiency vs Accuracy </u>

|Training samples (fraction) | Distilbert | Bert | Flan-T5 Large | Falcon-7B | RP-3B | RP-7B | Mistral-7B | Mistral-13B |
|:--------------------------:|:----------:|:----:|:-------------:|:---------:|:-----:|:-----:|:---------:|:----------:|
|266   (2.5%)                |36.24       |16.91 |59.86          |61.85      |55.32  |58.17  |52.10      |66.23       |
|533   (5%)                  |46.65       |30.75 |68.84          |64.02      |57.49  |60.31  |54.72      |67.45       |
|1066  (10%)                 |54.15       |53.73 |73.38          |67.52      |65.45  |67.22  |55.97      |71.69       |
|2666  (25%)                 |67.07       |68.41 |75.45          |70.32      |67.18  |69.53  |69.20      |73.50       |
|5332  (50%)                 |72.00       |72.46 |75.43          |72.42      |70.58  |70.96  |69.09      |77.87       |
|10664 (100%)                |71.91       |74.15 |72.31          |76.37      |72.34  |75.52  |75.30      |77.93       |

<u> Insight: </u>

The above table shows how performance of different LLMs track with sample efficiency. The last row of the table demonstrates the performance when the entire dataset is used. We can see that Mistral-13B outperforms all other models in terms of accuracy. Moreover, the first row of the table corresponding to the lowest fraction of training samples show a similar trend. Mistral-13B achieves the best performance in a low data situation across all models.

Mistral-7B, the smallest version of Llama, is unable to achieve competitive results when compared with other models across different sample fractions. This shows the significance of the extra parameters contained in Mistral-13B in comparison to Mistral-7B.



#### Summarization ####

<u> Table 2: Zero-Shot prompting vs Few-Shot prompting vs Fine-Tuning QLoRA </u>

|Method         | Mistral-7B Zero-Shot  | Mistral-7B Few-Shot  | Fine-Tuning + QLoRA |
|:-------------:|:--------------------:|:-------------------:|:-------------------:|
|ROUGE-1 (in %) |30.06                 |35.57                |51.71                |
|ROUGE-2 (in %) |8.61                  |14.23                |26.86                |


|Method         | Mistral-13B Zero-Shot  | Mistral-13B Few-Shot  | Fine-Tuning + QLoRA |
|:-------------:|:---------------------:|:--------------------:|:-------------------:|
|ROUGE-1 (in %) |11.02                  |22.50                 |52.97                |
|ROUGE-2 (in %) |3.38                   |9.25                  |28.32                |


<u> Insight: </u>

The Mistral-7B version performs significantly better than Mistral-13B in a zero-shot and few-shot setting. Looking at the ROUGE-1 and ROUGE-2 scores, we see that Mistral-7B’s performance consistently shines in comparison to Mistral-13B. However, post fine-tuning with QLoRA, Mistral-13B comes out ahead by a small margin. In our opinion, Mistral-7B can be a great candidate to consider for summarization and QnA tasks as it delivers strong results despite being smaller than Mistral-13B.


<u> Table 3: Mistral vs Other LLMs </u>

|Model          | Flan-T5-Base Full Fine-Tune | Flan-T5-Large | Falcon-7B | RP-3B | RP-7B | Mistral-7B | Mistral-13B |
|:-------------:|:---------------------------:|:-------------:|:---------:|:-----:|:-----:|:---------:|:----------:|
|ROUGE-1 (in %) |47.23                        |49.21          |52.18      |47.75  |49.96  |51.71      |52.97       | 
|ROUGE-2 (in %) |21.01                        |23.39          |27.84      |23.53  |25.94  |26.86      |28.32       |

<u> Insight: </u>

Both versions of Mistral achieve competitive results, with Mistral-13B taking the lead once again. In our opinion, Mistral and Falcon are good candidates to consider for summarization tasks. The 7B versions of both Mistral and Falcon can deliver good performance at potentially lower latencies.


