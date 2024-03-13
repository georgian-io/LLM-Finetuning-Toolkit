---
sidebar_label: Zephyr
sidebar_position: 6
---

import rocket from "./img/rocket.gif"
import time from "./img/time.gif"
import money from "./img/money.gif"
import progress from "./img/progress.gif"

# Zephyr

## What is Zephyr?

[Zephyr](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta) models are specifically tailored to function as a helpful assistant. It is an enhanced iteration of Mistral-7B, refined using Direct Preference Optimization (DPO) on a combination of public and synthetic datasets. Notably, the model demonstrates improved performance on MT Bench, resulting in a more helpful output. The authors [report](https://arxiv.org/abs/2310.16944) SOTA results on MT-Bench even compared with models that have much higher parameter counts (40B-70B).

## Variations of Zephyr and Parameters

Zephyr models come in two variations, and can be leveraged depending on the task at hand.

| Zephyr Variant | Parameters |
| :------------: | :--------: |
|   alpha (α)    |     7B     |
|    beta (β)    |     7B     |

Beta variant is newer and more performant. In this repository, we have experimented with the 7B-β variation.

## Evaluation Framework

### <img src={rocket} width="32" height="32"/>Performance<img src={rocket} width="32" height="32"/>

We evaluated Zephyr under the following conditions:

- Tasks & Datasets:
  - Classification: News Group dataset, which is a 20-way classification task.
  - Summarization: Samsum dataset.
- Experiments:
  - Sample Efficiency vs Accuracy
  - Zero-Shot prompting vs Few-Shot prompting vs PeFT QLoRA (for summarization)
  - Training with [NEFTune](https://arxiv.org/abs/2310.05914) vs without
  - Tuning only attention modules (default for `peft` library) vs all modules
- Training config:
  - Epochs: 5 (for classification)
  - Epochs: 1 (for summarization)
  - Zephyr-7B-β:
    - PeFT technique: QLoRA
    - Learning rate: 2e-4
- Hardware:
  - Cloud provider: AWC EC2
  - Instance: g5.2xlarge

#### Classification

<u> Table 1: Sample Efficiency vs Accuracy </u>

| Training samples (fraction) | Zephyr-7B-β | Zephyr-7B-β w/ NEFTune | Zephyr-7B-β w/ Full Module Tuning | Zephyr-7B-β w/ NEFTune + Full Module Tuning |
| :-------------------------: | :---------: | :--------------------: | :-------------------------------: | :-----------------------------------------: |
|         266 (2.5%)          |    46.05    |         49.61          |               65.36               |                    67.23                    |
|          533 (5%)           |    55.66    |         60.33          |               72.26               |                    72.94                    |
|         1066 (10%)          |    66.48    |         64.65          |               73.29               |                    72.82                    |
|         2666 (25%)          |    66.73    |         68.04          |               74.27               |                    75.85                    |
|         5332 (50%)          |    69.54    |         72.10          |               74.83               |                    74.40                    |
|        10664 (100%)         |    74.90    |         72.93          |               77.76               |                    77.86                    |

- Zephyr performance is roughly in-line with that of its base model, Mistral; however, we note that the performance tends to converge faster
- NEFTune tends to help model training when there is few examples; however as training set size increases, the performance is the same as non-NEFTune
- Tuning on all modules (attention + linear) makes the model converge much faster

#### Summarization

<u> Table 2: Zero-Shot prompting vs Few-Shot prompting vs Fine-Tuning QLoRA </u>

|     Method     | Zephyr-7B-β Zero-Shot | Zephyr-7B-β Few-Shot | Fine-Tuning + QLoRA | Fine-Tuning + QLoRA + NEFTune | Fine-Tuning + QLoRA + Full Module Tuning | Fine-Tuning + QLoRA + NEFTune + Full Module Tuning |
| :------------: | :-------------------: | :------------------: | :-----------------: | :---------------------------: | :--------------------------------------: | :------------------------------------------------: |
| ROUGE-1 (in %) |         33.93         |        35.99         |        52.84        |             52.97             |                  53.50                   |                       53.05                        |
| ROUGE-2 (in %) |         11.21         |        12.97         |        27.75        |             28.44             |                  29.66                   |                       29.23                        |

- Zephyr performance is roughly in-line with Mistral but slightly underperforms
- Few-shot approach only yields slight improvement in ROUGE metrics over zero-shot
- Fine-tuning works the best, but we note that using NEFTune and tuning on all modules only yield marginal performance improvements

<u> Table 3: Zephyr vs Other LLMs </u>

|     Model      | Flan-T5-Base Full Fine-Tune | Flan-T5-Large | Falcon-7B | RP-3B | RP-7B | Llama2-7B | Llama2-13B | Mistral-7B | Zephyr-7B-β |
| :------------: | :-------------------------: | :-----------: | :-------: | :---: | :---: | :-------: | :--------: | :--------: | :---------: |
| ROUGE-1 (in %) |            47.23            |     49.21     |   52.18   | 47.75 | 49.96 |   51.71   |   52.97    |   53.61    |    52.84    |
| ROUGE-2 (in %) |            21.01            |     23.39     |   27.84   | 23.53 | 25.94 |   26.86   |   28.32    |   29.28    |    28.44    |

- Zephyr achieves results comparable to Mistral, which is the best among 7B parameter models
