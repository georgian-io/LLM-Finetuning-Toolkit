# Contents:

- [Contents:](#contents)
	- [What is RedPajama?](#what-is-redpajama)
	- [Variations of RedPajama and Parameters](#variations-of-redpajama-and-parameters)
	- [What does this folder contain?](#what-does-this-folder-contain)
	- [Evaluation Framework](#evaluation-framework)
		- [ Performance ](#-performance-)
			- [Classification](#classification)
			- [Summarization](#summarization)
		- [  Time \& Cost to Train  ](#--time--cost-to-train--)
		- [ Inference ](#-inference-)
			- [Classification](#classification-1)
			- [Summarization](#summarization-1)

## What is RedPajama? 

RedPajama-INCITE (RP) combines Together.ai’s RedPajama dataset and EleutherAI’s Pythia model architecture to form a fully open source LLM. One interesting aspect is that there is a 3B parameter size model, which is unusual in our experience. They reason that this will allow for wider adoption due to smaller hardware requirements and easier experimentation. 

## Variations of RedPajama and Parameters

RedPajama models come in two sizes, and can be leveraged depending on the task at hand.

| RedPajama variation | Parameters  |
|:-------------------:|:-----------:|
|Base-3B              |3B           |
|Instruct-3B          |3B           |           
|Base-7B              |7B           |
|Instruct-7B          |7B           |

In this repository, we have experimented with all of the above variations. 

## What does this folder contain? 

This folder contains ready-to-use scripts, using which you can do the following:
	
* Finetuning RedPajama using PeFT methodology QLoRA:
	* ```redpajama_classification.py```: Finetune on News Group classification dataset
	* ```redpajama_summarization.py```: Finetune on Samsum summarization dataset
* Prompts used:
	* ```prompts.py```: Zero-shot, Few-shot and instruction tuning for classification and summarization
* Perform hyperparameter optimization over a well-constrained search space:
	* ```run_lora.sh```: Ablation study on LoRA's parameters 
	* ```sample_ablate.sh```: Ablation study over sample complexities
* Infer RedPajama using trained checkpoints:
	* ```redpajama_baseline_inference.py```: Infer in zero-shot and few-shot settings using RedPajama-3B or 7B Instruct versions
	* ```redpajama_classification_inference.py```: Infer on News Group classification dataset
	* ```redpajama_summarization_inference.py```: Infer on Samsum summarization dataset
* Infer across a different settings:
	* ```baseline_inference.sh```: Loop over all settings to perform zero-shot and few-shot prompting across classification and summarization tasks

## Evaluation Framework

In this section, we bring to you our insights after extensively experimenting with RedPajama-3B and 7B across different tasks. For a thorough evaluation, we need to evaluate the __four pillars__:

* Performance
* Cost to Train
* Time to Train
* Inference Costs


### <img src="../assets/rocket.gif" width="32" height="32"/> Performance <img src="../assets/rocket.gif" width="32" height="32"/>

We evaluated RedPajama (RP) under the following conditions:

* Tasks & Datasets:
	* Classification: News Group dataset, which is a 20-way classification task.
	* Summarization: Samsum dataset. 
* Competing Models:
	* BERT-Base (110M parameters)
	* Distilbert (66M parameters)
	* Flan-T5 Large (780M parameters)
	* Falcon-7B (7B parameters)
* Experiments:
	* Zero-Shot prompting vs Few-Shot prompting vs PeFT QLoRA
	* Sample Efficiency vs Accuracy
* Training config:
	* Epochs: 5
	* RedPajama-3B/7B:
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

<u> Table 1: Zero-Shot prompting vs Few-Shot prompting vs Fine-Tuning QLoRA </u>

|Method          | RP-3B Zero-Shot  | RP-3B Few-Shot | Fine-Tuning + QLoRA |
|:--------------:|:----------------:|:--------------:|:-------------------:|
|Accuracy (in %) |0.0               |:x:             |72.34                |


|Method          | RP-7B Zero-Shot  | RP-7B Few-Shot | Fine-Tuning + QLoRA |
|:--------------:|:----------------:|:--------------:|:-------------------:|
|Accuracy (in %) |0.0               |:x:             |75.52                |



NOTE: 

* ```prompts.py``` contains the prompts used for zero-shot prompting, few-shot prompting and instruction tuning.
* For zero-shot and few-shot experiments, we used the Instruct versions. For instruction tuning, we used the Base versions as per recommendations.


<u> Table 2: Sample Efficiency vs Accuracy </u>

|Training samples (fraction) | Distilbert | Bert | Flan-T5 Large + LoRA | Falcon-7B + QLoRA | RP-3B + QLoRA | RP-7B + QLoRA |
|:--------------------------:|:----------:|:----:|:--------------------:|:-----------------:|:-------------:|:-------------:|
|266   (2.5%)                |36.24       |16.91 |59.86                 |61.85              |55.32          |58.17          |
|533   (5%)                  |46.65       |30.75 |68.84                 |64.02              |57.49          |60.31          |
|1066  (10%)                 |54.15       |53.73 |73.38                 |67.52              |65.45          |67.22          |
|2666  (25%)                 |67.07       |68.41 |75.45                 |70.32              |67.18          |69.53          |
|5332  (50%)                 |72.00       |72.46 |75.43                 |72.42              |70.58          |70.96          |
|10664 (100%)                |71.91       |74.15 |72.31                 |76.37              |72.34          |75.52          |

<u> Insight: </u>

We can see that RedPajama-7B does a better job when compared to RedPajam-3B across all sample sizes! 

#### Summarization ####

<u> Table 3: Zero-Shot prompting vs Few-Shot prompting vs Fine-Tuning QLoRA </u>

|Method         | RP-3B Zero-Shot  | RP-3B Few-Shot  | Fine-Tuning + QLoRA |
|:-------------:|:----------------:|:---------------:|:-------------------:|
|ROUGE-1 (in %) |30.09             |29.16            |47.75                |
|ROUGE-2 (in %) |10.48             |10.05            |23.53                |


|Method         | RP-7B Zero-Shot  | RP-7B Few-Shot  | Fine-Tuning + QLoRA |
|:-------------:|:----------------:|:---------------:|:-------------------:|
|ROUGE-1 (in %) |30.85             |23.22            |49.96                |
|ROUGE-2 (in %) |11.30             |8.24             |25.94                |


<u> Insight: </u>

RedPajama does a much better job at summarizing dialogues than classifying news documents in zero-shot and few-shot settings. But Fine-Tuning is still most effective as it helps RedPajama learn the summarization style specific to the dataset as opposed to creating a generic summary. It is, however, surprising that Few-Shot prompting yields lower ROUGE scores than zero-shot prompting.

<u> Table 4: RedPajama vs Other LLMs </u>

|Model          | Flan-T5-Base Full Fine-Tune | Flan-T5-Large + LoRA | Falcon-7B + QLoRA | RP-3B + QLoRA | RP-7B + QLoRA |
|:-------------:|:---------------------------:|:--------------------:|:-----------------:|:-------------:|:-------------:|
|ROUGE-1 (in %) |47.23                        |49.21                 |52.18              |47.75          |49.96          |
|ROUGE-2 (in %) |21.01                        |23.39                 |27.84              |23.53          |25.94          |

<u> Insight: </u>

RedPajama-7B outperforms Flan-T5-Large and Flan-T5-Base versions, showcasing its superiority for summarization tasks. Furthermore, these results prove the merits of fine-tuning RedPajama-7B on target datasets as opposed to just using it out-of-the-box. 

### <img src="../assets/time.gif" width="32" height="32"/> <img src="../assets/money.gif" width="32" height="32"/> Time & Cost to Train <img src="../assets/money.gif" width="32" height="32"/> <img src="../assets/time.gif" width="32" height="32"/>

Conditions:

* AWS EC2 instance : g5.2xlarge
* Cost             : $1.212 / hour
* GPU included     : NVIDIA A10G: 24GB
* Epochs           : 5

<u> Table 5: Time & Cost to Train </u>

|Task           |Training Time | Training Cost |
|:-------------:|:------------:|:-------------:|
|Classification |2 hours       |$2.424         |
|Summarization  |3 hours       |$3.636         |


### <img src="../assets/progress.gif" width="32" height="32"/> Inference <img src="../assets/progress.gif" width="32" height="32"/>

As with previous assessments in this series, we used a load testing tool called Vegeta to test how effectively the system handles a large number of requests. We selected the HuggingFace Text Generation Inference Server and FastAPI as our deployment options. We aimed to determine the maximum RPS each model can handle, as well as the throughput, latency and cost per 1,000 tokens. We built a collection of example sentences each consisting of ~100 tokens in order to produce the requests. Then we randomly selected one of these sentences for each request during the load testing experiment. This approach helped us to ensure that our testing results are consistent. Through experimentation we determined the typical ranges of RPS that each model and service could handle for each task.  

To effectively evaluate RedPajama models through FastAPI, we decided to run the service with a single worker. This way GPU memory could be allocated for just one model instance at any given time. By doing so, we successfully averted the occurrence of "Out of Memory" errors, as the memory demands of RedPajama models surpassed the available GPU memory when multiple instances were executed concurrently. Since the TGI requires a standalone model, we have to merge the base model with the LoRA layers.

All load testing experiments have been performed on an AWS g5.4xlarge instance that costs US$1.624 per hour.  

#### Classification

For the classification task we tested the FastAPI service for RPS ranging from 1 to 4 with a step size of 1. Then, we evaluated Text Generation Inference (TGI) for RPS ranging from 10 to 150, using a step size of 15. We then calculated the average throughput and latency for the maximum possible RPS. The tables and plots demonstrate significant differences in the response speed and load capacity when deploying the RedPajama model through FastAPI versus TGI.

<u> Table 6: RedPajama-3B + LoRA </u>

|     Server   | Inference cost     | Requests per second (rps) | Throughput | Latency 90% |
|:------------:|:------------------:|:-------------------------:|:----------:|:-----------:|
|FastAPI (no optimization)| US$0,001 / 1K tokens |			4				|	0.15	 |	26.4 s.    |
|text-generation| US$0,00003 / 1K tokens|			135				|	57.3	 |	1.44 s.    |

The inference cost for FastAPI (US$0,001 / 1K tokens) is much higher than for Text Generation Inference (US$0,00003 / 1K tokens). Moreover, looking at the latency value we can say that it would cost US$0.0006 to get responses on 135 requests in 1.44 seconds using TGI, which is not possible to achieve with FastAPI. 

<img src="../assets/readme_images/redpajama_results/RedPajama-3B%20Classification.png" width="830" height="332"/>

<p></p>
<u> Table 7: RedPajama-3B + LoRA </u>

|     Server   | Inference cost     | Requests per second (rps) | Throughput | Latency 90% |
|:------------:|:------------------:|:-------------------------:|:----------:|:-----------:|
|FastAPI (no optimization)| US$0,001 / 1K tokens |			4				|	0.14	 |	28.1 s.    |
|text-generation| US$0,00003 / 1K tokens|			125				|	26.13	 |	3.98 s.    |
<p></p>
RedPajama-7B performs similarly to RedPajama-3B, but with a slightly lower RPS for text-generation. This is expected due to the larger model size.

Taking into account the latency value, it will cost US$0.001 to get responses on 125 requests with the RedPajama-7B model deployed on Text Generation Inference. 

<img src="../assets/readme_images/redpajama_results/RedPajama-7B%20Classification.png" width="830" height="332"/>

#### Summarization

For the summarization task we were testing the models for the RPS in range from 10 to 200 with a step equal to 15. You can see our results in the tables below. 

<u> Table 8: RedPajama-3B + LoRA </u>

|     Server   | Inference cost     | Requests per second (rps) | Throughput | Latency 90% |
|:------------:|:------------------:|:-------------------------:|:----------:|:-----------:|
|FastAPI (no optimization)| US$0.00002 / 1K tokens  |			160				|	5.46	 |	28.4     |
|text-generation| US$0.00001 / 1K tokens|			195				|	96.06	 |	0.7139    |

Even though the maximum RPS is quite similar for FastAPI and Text Generation using RedPajama-3B, the throughput and latency differ a lot which influences price per number of responses. According to our results, it will cost US$0.0003 to get responses for 195 requests using TGI and US$0.01 for 160 requests using FastAPI. 

<img src="../assets/readme_images/redpajama_results/RedPajama-3B%20Summarization.png" width="830" height="332"/>

<u> Table 9: RedPajama-7B + LoRA </u>

|     Server   | Inference cost     | Requests per second (rps) | Throughput | Latency 90% |
|:------------:|:------------------:|:-------------------------:|:----------:|:-----------:|
|FastAPI (no optimization)| US$0.00002 / 1K tokens   |			160				|	5.27	 |	29.527     |
|text-generation| US$0.00002 / 1K tokens|			145				|	41.5	 |	2.5    |

As we saw with the classification task, the load testing performance for summarization of  RedPajama-7B is similar to RedPajama-3B. The maximum RPS for FastAPI stays the same and for TGI it is a bit lower. 

<img src="../assets/readme_images/redpajama_results/RedPajama-7B%20Summarization.png" width="830" height="332"/>
<p> </p>

Our load testing experiments highlight a key finding: the difference in model size does not have a big impact on inference. Instead, the factor influencing performance is the choice of deployment platform. In our opinion, when dealing with LLMs, it's better never to use FastAPI as a deployment solution, and instead opt to much more effective options like Text Generation Inference.  