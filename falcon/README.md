# Contents:

- [Contents:](#contents)
	- [What is Falcon?](#what-is-falcon)
	- [Variations of Falcon and Parameters](#variations-of-falcon-and-parameters)
	- [What does this folder contain?](#what-does-this-folder-contain)
	- [Evaluation Framework](#evaluation-framework)
		- [ Performance ](#-performance-)
		- [  Time \& Cost to Train  ](#--time--cost-to-train--)
		- [ Inference ](#-inference-)
			- [FastApi](#fastapi)
			- [HuggingFace Text Generation Inference](#huggingface-text-generation-inference)

## What is Falcon? 

Falcon is a causal decoder-only model, i.e, given a sequence of words, it can predict the most-likely next word. Falcon comes in two sizes – 7 billion and 40 billion parameters. Furthermore, each of the two sizes has two versions: (i) base, which has been pre-trained on large corpuses of text and can be fine-tuned on downstream tasks, and (ii) instruct, which has already been fine-tuned on instructions, making it favorable for out-of-the-box chatbot and Q&A applications!

## Variations of Falcon and Parameters

Falcon models come in two sizes, and can be leveraged depending on the task at hand.

| Falcon variation | Parameters  |
|:----------------:|:-----------:|
|Base-3B            |3B          |
|Instruct-3B        |3B          |           
|Base-7B            |7B          |
|Instruct-7B        |7B          |

In this repository, we have used Falcon-7B for our experiments.

## What does this folder contain? 

This folder contains ready-to-use scripts, using which you can do the following:
	
* Finetuning Falcon-7B using PeFT methodology QLoRA:
	* ```falcon_classification.py```: Finetune on News Group classification dataset
	* ```falcon_summarization.py```: Finetune on Samsum summarization dataset
* Prompts used:
	* ```prompts.py```: Zero-shot, Few-shot and instruction tuning for classification and summarization
* Perform hyperparameter optimization over a well-constrained search space:
	* ```run_lora.sh```: Ablation study on LoRA's parameters 
	* ```sample_ablate.sh```: Ablation study over sample complexities
* Infer Falcon-7B using trained checkpoints:
	* ```falcon_baseline_inference.py```: Infer in zero-shot and few-shot settings using Falcon-7B Instruct version
	* ```falcon_classification_inference.py```: Infer on News Group classification dataset
	* ```falcon_summarization_inference.py```: Infer on Samsum summarization dataset
* Infer across a bunch of checkpoints:
	* ```baseline_inference.sh```: Loop over all settings to perform zero-shot and few-shot prompting across classification and summarization tasks

## Evaluation Framework

In this section, we bring to you our insights after extensively experimenting with Falcon-7B across different tasks. For a thorough evaluation, we need to evaluate the __four pillars__:

* Performance
* Cost to Train
* Time to Train
* Inference Costs


### <img src="../assets/rocket.gif" width="32" height="32"/> Performance <img src="../assets/rocket.gif" width="32" height="32"/>

We evaluated Falcon-7B under the following conditions:

* Tasks & Datasets:
	* Classification: News Group dataset, which is a 20-way classification task.
	* Summarization: Samsum dataset. 
* Competing Models:
	* BERT-Base (110M parameters)
	* Distilbert (66M parameters)
	* Flan-T5 Large (780M parameters)
* Experiments:
	* Zero-Shot prompting vs Few-Shot prompting vs PeFT QLoRA
	* Sample Efficiency vs Accuracy
* Training config:
	* Epochs: 5
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

|Method          | Zero-Shot  | Few-Shot | Fine-Tuning + QLoRA |
|:--------------:|:----------:|:--------:|:-------------------:|
|Accuracy (in %) |1.08        |:x:       |76.37                |


NOTE: 

* ```prompts.py``` contains the prompts used for zero-shot prompting, few-shot prompting and instruction tuning.
* For zero-shot and few-shot experiments, we used Falcon-7B-Instruct version. For instruction tuning, we used Faclon-7B-Base as per recommendations.


<u> Table 2: Sample Efficiency vs Accuracy </u>

|Training samples (fraction) | Distilbert | Bert | Flan-T5 Large + LoRA | Falcon-7B + QLoRA |
|:--------------------------:|:----------:|:----:|:--------------------:|:-----------------:|
|266   (2.5%)                |36.24       |16.91 |59.86                 |61.85              |
|533   (5%)                  |46.65       |30.75 |68.84                 |64.02              |
|1066  (10%)                 |54.15       |53.73 |73.38                 |67.52              |
|2666  (25%)                 |67.07       |68.41 |75.45                 |70.32              |
|5332  (50%)                 |72.00       |72.46 |75.43                 |72.42              |
|10664 (100%)                |71.91       |74.15 |72.31                 |76.37              |

<u> Insight: </u>

We can see that Falcon-7B does a significantly better job when compared to other models on a sample size as low as ~250! At roughly 50% of training samples, Distilbert and Bert finally catch-up to Falcon-7B, making Falcon-7B a great candidate to consider in low-data situations. 

#### Summarization ####

<u> Table 3: Zero-Shot prompting vs Few-Shot prompting vs Fine-Tuning QLoRA </u>

|Method         | Zero-Shot  | Few-Shot  | Fine-Tuning + QLoRA |
|:-------------:|:----------:|:---------:|:-------------------:|
|ROUGE-1 (in %) |32.21       |34.12      |52.18                |
|ROUGE-2 (in %) |10.08       |11.9       |27.84                |

<u> Insight: </u>

Falcon-7B does a much better job at summarizing dialogues than classifying news documents in zero-shot and few-shot settings. But Fine-Tuning is still most effective as it helps Falcon-7B learn the summarization style specific to the dataset as opposed to creating a generic summary. It is, however, surprising that Few-Shot prompting yields lower ROUGE scores than zero-shot prompting.

<u> Table 4: Falcon-7B vs Other LLMs </u>

|Model          | Flan-T5-Base Full Fine-Tune | Flan-T5-Large + LoRA | Falcon-7B + QLoRA |
|:-------------:|:---------------------------:|:--------------------:|:-----------------:|
|ROUGE-1 (in %) |47.23                        |49.21                 |52.18              |
|ROUGE-2 (in %) |21.01                        |23.39                 |27.84              |

<u> Insight: </u>

Falcon-7B outperforms Flan-T5-Large and Flan-T5-Base versions, showcasing its superiority for summarization tasks. Furthermore, these results prove the merits of fine-tuning Falcon-7B on target datasets as opposed to just using it out-of-the-box. 

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

We use a basic setup with FastAPI server without any optimization under the hood. 

All benchmarks were conducted on a g5.4xlarge AWS instance costing $1.624 (on-demand price as of June 2023). For stress-testing purposes, we used a load-testing tool called Vegeta and loaded the web servers with an increasing number of requests per second (RPS) until latency started to degrade significantly, or we started getting timeout errors. We conducted experiments for each RPS value multiple times (3-6 times) and calculated the average latency and throughput.

It is worth mentioning that when we perform a load test with a tool like Vegeta and set the request rate to n requests per second, it means that the tool attempts to simulate an average of n requests per second over the duration of the test. It doesn't guarantee that exactly n requests will be served and completed within each second.

Inference costs are derived from:
- Total tokens server can process in 1 hour = (rps * average number of tokens (input + output) * 60 seconds * 60 minutes)
- Price per hour = from AWS 
- Inference cost = Price per hour / (Total tokens in 1 hour / 1000)

This is a specific calculation, but using it makes it easy to compare other LLMs, including closed LLM APIs (such as GPT-4 and Writer Parmila). 


<u> Table 3: Cost estimation of deploying Falcon-7B + LoRA for summarization task </u>

|     Server   | Inference cost     | Requests per second (rps) | Throughput | Latency 90% |
|:------------:|:------------------:|:-------------------------:|:----------:|:-----------:|
|	FastAPI    |$0.00007 / 1K tokens|			30 			    |	1.5		 |	18.27 s.   | 
|text-generation| $0.00001 /1K tokens|			120				|	45.5	 |	2.03 s.    |

<p></p>
<u> Table 4: Cost estimation of deploying Falcon-7B + LoRA for classification task </u>
<p></p>

|     Server   | Inference cost        | Requests per second (rps)  | Throughput | Latency 90% |
|:------------:|:---------------------:|:--------------------------:|:----------:|:-----------:|
|	FastAPI    |$0.00001 / 1K tokens|			180 			|	5.84	 |	28.01 s.   | 
|text-generation|$0.00001 /1K tokens   |        145				|   78.5 	 |  1.5 s.     |



#### FastApi ####

For the summarization task we varied the RPS from 5 to 30, and examined the system's responsiveness across different load levels. We discovered that 90% of all requests had a response time equal to or less than 18.27 seconds (for 30 RPS). The plot also shows that as RPS increases, the 90th percentile latency rises gradually, signaling potential performance limitations. We found out that 35 requests per second is a critical threshold where the system fails.



#### HuggingFace Text Generation Inference ####

Text Generation Inference server developed by HuggingFace allows faster text generation by using advanced techniques like Tensor Parallelism and dynamic batching with popular open-source Language Model Libraries (LLMs) such as StarCoder, BLOOM, GPT-NeoX, Llama, and T5.

This time for the summarization task we varied the RPS value from 5 to 120. 90% of all requests had a response time equal to or less than 2.03 seconds (for 120 RPS). 

The Throughput value was reported as 45.5, which is much greater than the value we were able to get using FastApi. 

