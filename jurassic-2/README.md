# Contents:

- [Contents:](#contents)
	- [What is Jurassic-2?](#what-is-jurassic-2)
	- [Variations of Jurassic-2 and Parameters](#variations-of-jurassic-2-and-parameters)
	- [What does this folder contain?](#what-does-this-folder-contain)
	- [Evaluation Framework](#evaluation-framework)
		- [ Performance ](#-performance-)
			- [Classification](#classification)
			- [Summarization](#summarization)
		- [  Time \& Cost to Train  ](#--time--cost-to-train--)
		- [ Inference ](#-inference-)

## What is Jurassic-2? 

Jurassic-2, created by [AI21 Labs](https://www.ai21.com/), are large language models that give maximum flexibility to provide AI-first reading and writing experiences. Jurassic-2 line of models are available to developers via the [AI21 Studio](https://www.ai21.com/studio).

## Variations of Jurassic-2 and Parameters

Jurassic-2 models, for API use, come in a variety of sizes, and can be leveraged depending on the task at hand.

* Jurassic-2 Ultra: Unmatched quality
* Jurassic-2 Mid: Optimal balance of quality, speed, and cost
* Jurassic-2 Light: Fast and cost-effective

More information about each model can be found on their documentation [website](https://docs.ai21.com/docs/jurassic-2-models).

In parallel, there are a set of 3 models, which we think are similar to the above three, but with different names.

| Jurassic variation    | Parameters  	|
|:---------------------:|:-------------:|
| Jurassic-2 Light      |7B      	|
| Jurassic-2 Grande     |17B            |
| Jurassic-2 Jumbo      |178B     	|

There are also other more specialized instruction-tuned models which can be found on their website. In this repository, we have experimented with all variations of Jurassic-2 models for out-of-the-box tasks as well as fine-tuning purposes.

## What does this folder contain? 

This folder contains ready-to-use scripts, using which you can do the following:

* Prompts used:
	* ```prompts.py```: Zero-shot, Few-shot and instruction tuning for classification and summarization
* Infer Jurassic-2 models:
	* ```jurassic_baseline_inference.py```: Zero-shot, Few-shot and fine-tuned versions for summarization task
	* ```baseline_inference.sh```: Shell script to loop over all combinations of Jurassic-2's models for out-of-the-box summarization capabilities
	* ```custom_inference.sh```: Shell script to loop over fine-tuned Jurassic-2's models


Note: To learn how to fine-tune Jurassic-2 line of models, please follow the simple and easy-to-understand [tutorial](https://docs.ai21.com/docs/custom-models) on AI21's website.

## Evaluation Framework

In this section, we bring to you our insights after extensively experimenting with InstructJurassic-30B across different tasks. For a thorough evaluation, we need to evaluate the __four pillars__:

* Performance
* Cost to Train
* Time to Train
* Inference Costs


### <img src="../assets/rocket.gif" width="32" height="32"/> Performance <img src="../assets/rocket.gif" width="32" height="32"/>

We evaluated InstructJurassic-30B under the following conditions:

* Tasks & Datasets:
	* Summarization: Samsum dataset. 
* Experiments:
	* Zero-Shot prompting vs Few-Shot prompting vs Fine-tuning
* Hardware:
	* No requirements as all the foundational and custom models are hosted by AI21 Labs. 
	
#### Classification ####

We track accuracy to evaluate model's performance on classification task.

<u> Table 1: Zero-Shot prompting </u>

|Model          | Accuracy (in %) |
|:-------------:|:---------------:|
|J2-Light       | 1.82            | 
|J2-Mid         | 22.93           |
|J2-Ultra       | 43.62           |


<u> Insight: </u>

* Jurassic-2 models show consistent improvement in zero-shot prompting setting as the model size increases.
* J2-Ultra achieves the highest accuracy, followed by J2-Mid and J2-Light.
* In our opinion, these numbers are very high considering these models are used out-of-the-box.

#### Summarization ####

We track ROUGE-1 and ROUGE-2 metrics to evaluate model's performance on summarization task.

<u> Table 2: Zero-Shot prompting </u>

|Model          | ROUGE-1 (in %) | ROUGE-2 (in %) |
|:-------------:|:--------------:|:--------------:|
|J2-Light       | 38.218         | 14.780         |
|J2-Mid         | 39.119         | 15.591         |
|J2-Ultra       | 41.635         | 17.273         |


<u> Table 3: Few-Shot prompting </u>

|Model          | ROUGE-1 (in %) | ROUGE-2 (in %) |
|:-------------:|:--------------:|:--------------:|
|J2-Light       | 40.736         | 17.092         |
|J2-Mid         | 43.390         | 18.346         |
|J2-Ultra       | 45.317         | 19.276         |


<u> Table 4: Fine-tuning (custom model) </u>

|Model          | ROUGE-1 (in %) | ROUGE-2 (in %) |
|:-------------:|:--------------:|:--------------:|
|J2-Light       | 44.694         | 20.153         |
|J2-Grande      | 48.385         | 23.901         |
|J2-Jumbo       | DNF\*          | DNF\*          |


Note: All models for fine-tuned for 1 epoch.


<u> Insight: </u>

* For out-of-the-box performance, Jurassic-2's biggest model J2-Ultra achieves the best ROUGE-1 and ROUGE-2 when compared to the other models.
* Jurassic-2's model performance steadily increases from Light to Mid to Ultra, across both zero-shot and few-shot settings.
* Few-shot prompting consistently performs better than zero-shot prompting across all model variations.
* As expected, J2-Light and J2-Grande fine tuned (custom) models achieve better performance than their out-of-the-box counterparts, i.e., zero-shot and few-shot.
* Despite debugging efforts, the custom version of J2-Jumbo (the biggest model) does not generate anything (DNF) for the exact same inputs.
 


### <img src="../assets/time.gif" width="32" height="32"/> <img src="../assets/money.gif" width="32" height="32"/> Time & Cost to Train <img src="../assets/money.gif" width="32" height="32"/> <img src="../assets/time.gif" width="32" height="32"/>


* Fine-tuning Jurassic-2 custom models for 1 epoch took between 15 minutes to 1 hour, depending on the size of the model in consideration.


Following is the cost breakdown for fine-tuning Jurassic-2's different models, as per what AI21 charged:

|Model          | MB / Epochs | Cost  |
|:-------------:|:-----------:|:-----:|
|J2-Light       | 8           | $0.88 |
|J2-Grande      | 8           | $4.00 |
|J2-Jumbo       | 8           | $26.47|


<u> Insights: </u>

* J2-Jumbo (the biggest model) does not generate anything for the exact same inputs.
* MB / Epochs is basically the size of training data being used per epoch. In the case of summarization, Samsum's training split is roughly 8 MB in size.


### <img src="../assets/progress.gif" width="32" height="32"/> Inference <img src="../assets/progress.gif" width="32" height="32"/>

Following are the tables of the APIs' mean response time (MRT), i.e., how long it took for the API to return responses on average.

#### Classification ####

<u> Table 5: Zero-Shot prompting </u>

|Model          | Mean Response Time (MRT)(in sec) |
|:-------------:|:--------------------------------:|
|J2-Light       | 0.58 ± 0.19                      | 
|J2-Mid         | 0.53 ± 0.27                      |
|J2-Ultra       | 0.66 ± 0.45                      |


#### Summarization ####

<u> Table 5: Zero-Shot prompting VS Few-shot prompting VS Fine-tuning </u>

|Model          | Zero-Shot MRT | Few-Shot MRT | Fine-tune MRT |
|:-------------:|:-------------:|:------------:|:-------------:|
|J2-Light       | 0.63 ± 0.18   | 0.56 ± 0.17  | 0.48 ± 0.08   |  
|J2-Mid         | 1.23 ± 6.45   | 0.83 ± 0.31  | 1.76 ± 1.01   | 
|J2-Ultra       | 1.00 ± 0.44   | 0.92 ± 0.51  | DNF\*         |

<u> Insights: </u>

* The MRT for Few-shot summarization is lower across all three models when compared with Zero-shot's MRT. This could be because the models generate longer summaries in a zero-shot setting as they do not have a sense of how long the summaries should be. As examples from the training split of SAMSUM are provided to the models in the few-shot setting, they guide the models to generate summaries that are smaller in length, similar to their ground-truth counterparts.
* At this time, conclusive insights cannot be drawn from the fine-tuning MRTs. 
