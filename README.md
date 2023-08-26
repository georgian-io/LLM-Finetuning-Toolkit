<!-- markdownlint-configure-file {
  "MD013": {
    "code_blocks": false,
    "tables": false
  },
  "MD033": false,
  "MD041": false
} -->

<div align="center">

# LLM Finetuning Hub 

<img src="assets/repo-main.png" width="512" height="296"/>

LLM Finetuning Hub contains code and insights to finetune various large language models for your use-case.

We stress-test both open-source and close-source LLMs through our Evaluation Framework to check their applicability for real-life business use-cases. Finetuning LLMs has never been easier.

[Evaluation Framework](#evaluation-framework) •
[Getting Started](#getting-started) •
[LLM Roadmap](#llm-roadmap) •
[Contributing](#contributing)

</div>

## Evaluation Framework

For a holistic evaluation, we will make use of the __Evaluation Framework__ that contains __4 pillars__:

- <img src="assets/rocket.gif" width="32" height="32"/> Performance <img src="assets/rocket.gif" width="32" height="32"/>
- <img src="assets/time.gif" width="32" height="32"/> Time to Train <img src="assets/time.gif" width="32" height="32"/>
- <img src="assets/money.gif" width="32" height="32"/> Cost to Train <img src="assets/money.gif" width="32" height="32"/>
- <img src="assets/progress.gif" width="32" height="32"/> Inferencing <img src="assets/progress.gif" width="32" height="32"/>


For each of the above four pillars, we are sharing our codebase and insights to:
- Assist you to leverage LLMs for your business needs and challenges
- Decide which LLM suits your needs from a performance and cost perspective
- Boost reproducibility efforts which are becoming increasingly difficult with LLMs

We are providing scripts that are ready-to-use for:
- Finetuning LLMs on your proprietary dataset via PeFT methodologies such as LoRA and Prefix Tuning
- Performing hyperparameter optimization to get the maximum performance out of these models

## Getting Started 

You can start fine-tuning your choice of LLM in 4 easy steps:

1. **Setup conda environment**

	```shell
	wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.11.0-Linux-x86_64.sh
	bash Miniconda3-py38_4.11.0-Linux-x86_64.sh
	source ~/.bashrc
	conda create --name llm_finetuning python=3.9
	conda activate llm_finetuning
	```

2. **Install relevant packages**

	```shell
	git clone https://github.com/georgian-io/LLM-Finetuning-Hub.git
	cd LLM-Finetuning-Hub/
	pip install -r requirements.txt
	```

3. **Finetune your LLM of choice**

	For instance, to finetune Falcon-7B, do the following:

	```shell
	cd falcon/ # navigate to Falcon folder
	python falcon_classification.py --lora_r 8 --epochs 5 --dropout 0.1 # finetune Falcon-7B on newsgroup classification dataset
	python falcon_classification_inference.py --experiment <experiment folder> # evaluate finetuned Falcon
	python falcon_summarization.py --lora_r 8 --epochs 1 --dropout 0.1 # finetune Falcon-7B on samsum chat dataset
	python falcon_summarization_inference.py --experiment <experiment folder> # evaluate finetuned Falcon
	```

	For instance, to finetune Flan-T5-Large, do the following:

	```shell
	cd flan-t5/ # navigate to Flan-T5 folder
	python flan_classification.py --peft_method prefix --prefix_tokens 20 --epochs 5 # finetune Flan-T5 on newsgroup dataset
	python flan_classification_inference.py --experiment <experiment folder> # evaluate finetuned Flan-T5
	python flan_summarization.py --peft_method lora --lora_r 8 --epochs 1 # finetune Flan-T5 on samsum chat dataset
	python flan_summarization_inference.py --experiment <experiment folder> # evalute finetuned Flan-T5
	```

4. **Zero-shot and Few-shot your LLM of choice**

	For instance, to use Falcon-7B on newsgroup classification task, do the following:

	```shell
	python falcon_baseline_inference.py --task_type classification --prompt_type zero-shot
	python falcon_baseline_inference.py --task_type classification --prompt_type few-shot
	```

	To use Falcon-7B on samsum summarization task, do the following:

	```shell
	python falcon_baseline_inference.py --task_type summarization --prompt_type zero-shot
	python falcon_baseline_inference.py --task_type summarization --prompt_type few-shot
	```

NOTE: All of our experiments were conducted on the AWS EC2 instance: g5.2xlarge. It has one 24GB Nvidia GPU, and is sufficient to finetune the LLMs in this repository.

## LLM Roadmap

Our plan is to perform these experiments on all the LLMs below. To that end, this is a tentative roadmap of the LLMs that we aim to cover, and their corresponding codebase and README links:

| LLM        | Benchmarked?        |  Open-Source?     | Model README | Codebase |
|:----------:|:-------------------:|:-----------------:|:------------:|:------------:|
|Flan-T5     |:white_check_mark:   |:white_check_mark: |[Link](https://github.com/georgian-io/LLM-Finetuning-Hub/blob/readme/update/flan-t5/README.md) | [Folder](https://github.com/georgian-io/LLM-Finetuning-Hub/tree/readme/update/flan-t5) |
|Falcon      |:white_check_mark:   |:white_check_mark: |[Link](https://github.com/georgian-io/LLM-Finetuning-Hub/blob/main/falcon/README.md) | [Folder](https://github.com/georgian-io/LLM-Finetuning-Hub/tree/readme/update/falcon) |
|RedPajama   |:white_check_mark:   |:white_check_mark: |[Link](https://github.com/georgian-io/LLM-Finetuning-Hub/blob/readme/update/redPajama/README.md) | [Folder](https://github.com/georgian-io/LLM-Finetuning-Hub/tree/readme/update/redPajama) |
|Llama-2     | |:white_check_mark: | | |
|OpenLlama   | |:white_check_mark: | | |
|SalesForce XGen | |:white_check_mark: | | |
|Mosaic MPT | |:white_check_mark: | | | 
|Cerebras | |:white_check_mark: | | |
|Writer Palmyra | |:x: | | |
|OpenAI GPT-3.5 | |:x: | | |
|Google PaLM | |:x: | | |
|Inflection Pi | |:x: | | |

## Contributing

If you would like to contribute to this project, we recommend following the "fork-and-pull" Git workflow.

 1. **Fork** the repo on GitHub
 2. **Clone** the project to your own machine
 3. **Commit** changes to your own branch
 4. **Push** your work back up to your fork
 5. Submit a **Pull request** so that we can review your changes

NOTE: Be sure to merge the latest from "upstream" before making a pull request!

## Correspondence

If you have any questions or issues, or would like to contribute to this repository, please reach out to:

- Rohit Saha ([Email](mailto:rohit@georgian.io) | [LinkedIn](https://www.linkedin.com/in/rohit-saha-ai/))
- Kyryl Truskovskyi ([Email](mailto:kyryl@georgian.io) | [LinkedIn](https://www.linkedin.com/in/kyryl-truskovskyi-275b7967/))
- Maria Ponomarenko ([Email](mailto:mariia.ponomarenko@georgian.io) | [LinkedIn](https://www.linkedin.com/in/maria-ponomarenko-71b465179/))

