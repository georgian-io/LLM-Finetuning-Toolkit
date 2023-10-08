# Contents:

- [Contents:](#contents)
	- [What is GPT-3.5?](#what-is-gpt-3.5)
	- [Variations of GPT-3.5 and Parameters](#variations-of-gpt-3.5-and-parameters)
	- [What does this folder contain?](#what-does-this-folder-contain)
	- [Evaluation Framework](#evaluation-framework)
		- [ Performance ](#-performance-)
			- [Classification](#classification)
			- [Summarization](#summarization)
		- [  Time \& Cost to Train  ](#--time--cost-to-train--)
		- [ Inference ](#-inference-)

## What is GPT-3.5? 

OpenAI's GPT (generative pre-trained transformer) models have been trained to understand natural language and code. GPTs provide text outputs in response to their inputs. The inputs to GPTs are also referred to as "prompts". Designing a prompt is essentially how you "program" a GPT model, usually by providing instructions or some examples of how to successfully complete a task. GPTs can be used across a great variety of tasks including content or code generation, summarization, conversation, creative writing, and more. [Source](https://platform.openai.com/docs/introduction/key-concepts)

## Variations of GPT-3.5 and Parameters

GPT-3.5 models come in a variety of sizes, and can be leveraged depending on the task at hand. An extensive list of GPT-3.5 models can be found [here](https://platform.openai.com/docs/models/gpt-3-5).

 We have used GPT-3.5-turbo for our experiments.

## What does this folder contain? 

This folder contains ready-to-use scripts, using which you can do the following:

* Prompts used:
	* ```prompts.py```: Zero-shot, Few-shot and instruction tuning for classification and summarization
* Infer GPT-3.5-turbo for out-of-the-box classification and summarization:
	* ```gpt_baseline_inference.py```: Infer in zero-shot and few-shot settings using GPT-3.5-turbo version
	* ```baseline_inference.sh```: Set of experiments to run
* Finetune and infer GPT-3.5 for classification and summarization:
	* ```gpt_finetune.py```: Prepare data for finetuning + Submit finetuning job + Infer via finetuned model
	* ```finetune_instructions.sh```: Set of commands to run for data upload + finetune + infer

## Evaluation Framework

In this section, we bring to you our insights after extensively experimenting with GPT-3.5-turbo across different tasks. For a thorough evaluation, we need to evaluate the __four pillars__:

* Performance
* Cost to Train
* Time to Train
* Inference Costs


### <img src="../assets/rocket.gif" width="32" height="32"/> Performance <img src="../assets/rocket.gif" width="32" height="32"/>

We evaluated GPT-3.5-turbo under the following conditions:

* Tasks & Datasets:
	* Classification: News Group dataset, which is a 20-way classification task.
	* Summarization: Samsum dataset. 
* Experiments:
	* Zero-Shot prompting vs Few-Shot prompting vs Fine-Tuning
* Hardware:
	* No requirements as we used the [OpenAI API](https://platform.openai.com/docs/api-reference)
	
#### Classification ####

<u> Table 1: Zero-Shot prompting vs Few-Shot prompting </u>

|Method          | Zero-Shot  | Few-Shot |
|:--------------:|:----------:|:--------:|
|Accuracy (in %) | 60.22      | :x:      |

NOTE: 

* ```prompts.py``` contains the prompts used for zero-shot prompting, and few-shot prompting.
* We used GPT-3.5-turbo version. 

<u> Insight: </u>

The Few-Shot prompting approach failed due to the prompt surpassing the token limit of 4097 used by GPT-3.5-turbo. However, in the Zero-Shot setting, the model performed relatively well, achieving a 60.22% accuracy on the classification task. 

<u> Table 2: Sample Efficiency vs Accuracy </u>

|Training samples (fraction) | GPT-3.5-turbo |
|:--------------------------:|:-------------:|
|266   (2.5%)                |73.81          |
|533   (5%)                  |56.17          |
|1066  (10%)                 |47.32          |
|2666  (25%)                 |49.15          |
|5332  (50%)                 |78.84          |
|10664 (100%)                |79.41          |

<u> Insight: </u>

* We can see that GPT-3.5-turbo does a good job on a sample size as low as ~250!
* It is surprising that the accuracy drops when the sample fraction is increased from 2.5% to 5%, 10% and 25%. Ideally, the performance should have increased with more training samples. It is challenging to understand why this happened as OpenAI does not give control to hyperparameters aside from n\_epochs.
* However, the performance goes back up when half, and the entirety of training data is used. 



#### Summarization ####

<u> Table 3: Zero-Shot prompting vs Few-Shot prompting vs Fine-Tuning </u>

|Method         | Zero-Shot  | Few-Shot  | Fine-Tuning |
|:-------------:|:----------:|:---------:|:-----------:|
|ROUGE-1 (in %) | 36.419     | 39.089    | 55.915      |
|ROUGE-2 (in %) | 13.316     | 15.835    | 31.887      |

NOTE: 

* ```prompts.py``` contains the prompts used for zero-shot prompting, and few-shot prompting.
* We used GPT-3.5-turbo version. 

<u> Insight: </u>

GPT-3.5-turbo achieves better performance in few-shot setting than zero-shot. However, fine-tuned GPT-3.5-turbo achieves the best ROUGE-1 and ROUGE-2 scores, attesting the effectiveness of fine-tuning on proprietary data.


### <img src="../assets/time.gif" width="32" height="32"/> <img src="../assets/money.gif" width="32" height="32"/> Time & Cost to Train <img src="../assets/money.gif" width="32" height="32"/> <img src="../assets/time.gif" width="32" height="32"/>

GPT-3.5-turbo was trained on a large corpus of Internet data, curated upto Sept 2021. More information can be found on OpenAI's [Models](https://platform.openai.com/docs/models/gpt-3-5) page.

### <img src="../assets/progress.gif" width="32" height="32"/> Inference <img src="../assets/progress.gif" width="32" height="32"/>


| Task           	| Time/Call 	| Total Time 	|
|----------------	|-----------	|------------	|
| Classification 	| ~1s       	| ~2.5 hours 	|
| Summarization  	| ~1s       	| ~0.5 hours  	|

Note: We observed no significant difference in the time taken for zero-shot vs few-shot for summarization. For classification, few-shot experiments were faster since they failed due to hitting the maximum token length. We ignore this speed up due to the failure to generate a response.
