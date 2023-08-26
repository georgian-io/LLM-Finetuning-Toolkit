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

<img src="assets/repo-main.png" width="512" height="212"/>

LLM Finetuning Hub contains code and insights to finetune various large language models for your use-case.

We stress-test both open-source and close-source LLMs through our Evaluation Framework to check their applicability for real-life business use-cases. Finetuning LLMs has never been easier.

[Evaluation Framework](#evaluation-framework) •
[Getting started](#getting-started) •
[LLM roadmap](#llm-roadmap) •
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

## Getting started 

You can start fine-tuning your choice of LLM in 4 easy steps:

1. **Setup conda environment**

   ```wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.11.0-Linux-x86_64.sh
      bash Miniconda3-py38_4.11.0-Linux-x86_64.sh
      source ~/.bashrc
      conda create --name llm_finetuning python=3.9
      conda activate llm_finetuning
   ```

All of our experiments were conducted on the AWS EC2 instance: g5.2xlarge. It has one 24GB Nvidia GPU, and is sufficient to run all of our codebase.

Go over to the LLM-specific directory that you are interested in, and open the ```README.md```. We have included details about the LLM, followed by performance results on open-source datasets!

## Roadmap

Our plan is to perform these experiments on all the LLMs below. To that end, this is a tentative roadmap of the LLMs that we aim to cover:

- [x] Flan-T5
- [x] Falcon 
- [x] RedPajama
- [ ] Llama-2 (ingredients are being prepped!)
- [ ] OpenLlama
- [ ] SalesForce XGen
- [ ] OpenAI GPT-4
- [ ] Google PaLM
- [ ] Inflection Pi

## Correspondence

If you have any questions or issues, or would like to contribute to this repository, please reach out to:

- Rohit Saha ([Email](mailto:rohit@georgian.io) | [LinkedIn](https://www.linkedin.com/in/rohit-saha-ai/))
- Kyryl Truskovskyi ([Email](mailto:kyryl@georgian.io) | [LinkedIn](https://www.linkedin.com/in/kyryl-truskovskyi-275b7967/))
- Maria Ponomarenko ([Email](mailto:mariia.ponomarenko@georgian.io) | [LinkedIn](https://www.linkedin.com/in/maria-ponomarenko-71b465179/))

