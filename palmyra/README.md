# Contents:

- [Contents:](#contents)
	- [What is Palmyra?](#what-is-palmyra)
	- [Variations of Palmyra and Parameters](#variations-of-palmyra-and-parameters)
	- [What does this folder contain?](#what-does-this-folder-contain)
	- [Evaluation Framework](#evaluation-framework)
		- [ Performance ](#-performance-)
			- [Classification](#classification)
			- [Summarization](#summarization)
		- [  Time \& Cost to Train  ](#--time--cost-to-train--)
		- [ Inference ](#-inference-)

## What is Palmyra? 

Palmyra is a causal decoder-only model, i.e, given a sequence of words, it can predict the most-likely next word. Created by the [Writer](https://writer.com/) team, Palmyra was trained on a custom dataset created by Writer. Several of the models in this family are publicly available on [HuggingFace](https://huggingface.co/Writer), with the largest models being exclusive to Writer's product and API.

## Variations of Palmyra and Parameters

Palmyra models come in a variety of sizes, and can be leveraged depending on the task at hand.

| Palmyra variation     | Parameters  	|
|:---------------------:|:-------------:|
| Palmyra Small			| 128M		  	|
| Palmyra 3B			| 3B			|
| Palmyra Base			| 5B			|
| Palmyra Large			| 20B			|
| InstructPalmyra-20B	| 20B			|
| InstructPalmyra-30B	| 30B			|

There are also a couple of other more specialized models which can be viewed in their [API documentation](https://dev.writer.com/docs/models). In this repository, we have used InstructPalmyra-30B for our experiments.

## What does this folder contain? 

This folder contains ready-to-use scripts, using which you can do the following:

* Prompts used:
	* ```prompts.py```: Zero-shot, Few-shot and instruction tuning for classification and summarization
* Infer Palmyra-30B using trained checkpoints:
	* ```palmyra_baseline_inference.py```: Infer in zero-shot and few-shot settings using Palmyra-30B Instruct version

## Evaluation Framework

In this section, we bring to you our insights after extensively experimenting with InstructPalmyra-30B across different tasks. For a thorough evaluation, we need to evaluate the __four pillars__:

* Performance
* Cost to Train
* Time to Train
* Inference Costs


### <img src="../assets/rocket.gif" width="32" height="32"/> Performance <img src="../assets/rocket.gif" width="32" height="32"/>

We evaluated InstructPalmyra-30B under the following conditions:

* Tasks & Datasets:
	* Classification: News Group dataset, which is a 20-way classification task.
	* Summarization: Samsum dataset. 
* Experiments:
	* Zero-Shot prompting vs Few-Shot prompting
* Hardware:
	* No requirements as we used the [Writer API](https://dev.writer.com/docs/quickstart)
	
#### Classification ####

<u> Table 1: Zero-Shot prompting vs Few-Shot prompting </u>

|Method          | Zero-Shot  | Few-Shot |
|:--------------:|:----------:|:--------:|
|Accuracy (in %) | 15.236     | :x:      |

NOTE: 

* ```prompts.py``` contains the prompts used for zero-shot prompting, and few-shot prompting.
* We used InstructPalmyra-30B version. 
* The Palmyra API has a moderation tool built-in, thus some prompts failed to return a response due to the moderation tool being triggered. In such cases, we count it as an incorrect response.

<u> Insight: </u>

The Few-Shot prompting approach failed due to the prompt surpassing the token limit of 2050 used by InstructPalmyra-30B. However, even in the Zero-Shot setting, the model did not work very well, achieving only a 15.236% accuracy on the classification task. 

#### Summarization ####

<u> Table 2: Zero-Shot prompting vs Few-Shot prompting </u>

|Method         | Zero-Shot  | Few-Shot  |
|:-------------:|:----------:|:---------:|
|ROUGE-1 (in %) | 33.680     | 39.285    |
|ROUGE-2 (in %) | 12.184     | 16.199    |

NOTE: 

* ```prompts.py``` contains the prompts used for zero-shot prompting, and few-shot prompting.
* We used InstructPalmyra-30B version. 
* The Palmyra API has a moderation tool built-in, thus some prompts failed to return a response due to the moderation tool being triggered. In such cases, we count it as an incorrect response.

<u> Insight: </u>

InstructPalmyra-30B does a much better job at summarizing dialogues than classifying news documents in zero-shot and few-shot settings. Understandably, few-shot prompting does noticeably better than zero-shot prompting on both ROUGE metrics. However, the overall score is still relatively low.


### <img src="../assets/time.gif" width="32" height="32"/> <img src="../assets/money.gif" width="32" height="32"/> Time & Cost to Train <img src="../assets/money.gif" width="32" height="32"/> <img src="../assets/time.gif" width="32" height="32"/>

Palmyra was trained on a private dataset from Writer. Thus we do not have information on the dataset, the time it took to train or the cost of training the model.

### <img src="../assets/progress.gif" width="32" height="32"/> Inference <img src="../assets/progress.gif" width="32" height="32"/>

In terms of costs, Palmyra has two different [plans](https://writer.com/plans/). The first is for small teams and costs USD 18/user/month. The second is the enterprise plan which can be negotiated with their sales team. For the purposes of these experiments, we utilized the free trial available on the website. Thus there is no cost per-API call but rather a single monthly rate. 

| Task           	| Time/Call 	| Total Time 	|
|----------------	|-----------	|------------	|
| Classification 	| ~1s       	| ~2.5 hours 	|
| Summarization  	| ~2s       	| ~1 hour    	|

Note: We observed no significant difference in the time taken for zero-shot vs few-shot for summarization. For classification, few-shot experiments were faster since they failed due to hitting the maximum token length. We ignore this speed up due to the failure to generate a response.