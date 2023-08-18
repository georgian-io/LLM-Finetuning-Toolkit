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

With inference, we used the same approach for deployment and cost estimation for the Flan model. 

Following the same process we used to test Flan-T5-Large, we are using the load testing tool, Vegeta, on RedPajama. We created a script that sent varying numbers of requests (ranging from 5 to 185) in three sets, with a three-second interval to give the server time to recover. Afterward, we examined the results, excluding instances where a "too many requests" error occurred. We calculated the average throughput and latency (90%) for the maximum possible requests per second (RPS) and used this data to calculate the cost. Again, following the same process we used to test Flan-T5-Large,  all of the load testing experiments have been executed on a g5.4xlarge instance.

For the summarization task, we varied the RPS from five to 180. 90% of all requests had a response time equal to or less than 1.82 seconds for 145 RPS (which is the maximum number of requests the server was able to handle). 

As for the cost, taking into account that the throughput value was reported as 53.8, to get this number of responses in one second (or to get 145 responses in ~1.82 seconds) will cost you $0.0008.

<u> Table 6: Cost estimation of deploying RedPajama-7B + LoRA for summarization task </u>

|     Server   | Inference cost     | Requests per second (rps) | Throughput | Latency 90% |
|:------------:|:------------------:|:-------------------------:|:----------:|:-----------:|
|text-generation| $0.00004 / 1K tokens|			145				|	53.8	 |	1.82 s.    |

<img src="../inference/load_testing/vegeta/text_gen/plots/falcon/redpajama_summ_exp1.png" width="430" height="332"/>

The performance of the classification model during inference is quite similar to the summarization. The maximum RPS that TGI was able to handle equals to 125. 

Taking into account the latency value, it will cost $0.001 to get responses for 125 requests in 2.7s. 

<p></p>
<u> Table 7: Cost estimation of deploying RedPajama-7B + LoRA for classification task </u>
<p></p>

|     Server   | Inference cost        | Requests per second (rps)  | Throughput | Latency 90% |
|:------------:|:---------------------:|:--------------------------:|:----------:|:-----------:|
|text-generation|$0.00005 / 1K tokens   |        125				|   30.3 	 |  2.7 s.     |
<p></p>
<img src="../inference/load_testing/vegeta/text_gen/plots/falcon/redpajama_class_exp1.png" width="430" height="332"/>



