Hi everyone, we've been working on [benchmarking different open-source LLMs](https://github.com/georgian-io/LLM-Finetuning-Hub). We measure, in particular, on the performance of these models once finetued (via QLoRA) on classic NLP downstream tasks like summarization and classification. We also put particular emphasis on benchmarking inference time/cost for these models once deployed.

We've just ran our study on the new Zephyr-7B-beta model, a DPO-tuned version of Mistral-7B. Thought it would be nice to share with the community!

We tested out-of-the-box performance of Zephyr for summarization under zero-shot and few-shot (for classification, we couldn't do few-shot  because of context length, and we haven't tried zero-shot since most other open source models gave subpar results).

Then we tested the performance after QLoRA fine-tuning and saw substantial performance boost (as it should). Afterwards we experimented levers we can pull to increase model performance ()


## Summarization
**Dataset Used:** Newsgroup
**Rank:** 64

|Method         | Zephyr-7B-β Zero-Shot | Zephyr-7B-β Few-Shot | Fine-Tuning + QLoRA | Fine-Tuning + QLoRA + NEFTune  | Fine-Tuning + QLoRA + Full Module Tuning | Fine-Tuning + QLoRA + NEFTune + Full Module Tuning | 
|:-------------:|:---------------------:|:--------------------:|:-------------------:|:------------------------------:|:----------------------------------------:|:--------------------------------------------------:|
|ROUGE-1 (in %) |33.93                  |35.99                 |52.84                |52.97                           | 53.50                                    | 53.05                                              |
|ROUGE-2 (in %) |11.21                  |12.97                 |27.75                |28.44                           | 29.66                                    | 29.23                                              |

- We see that zero-shot and few-shot performance is already pretty good out-of-the box
- QLoRA was able to refine the syntactic style and pithiness of outputs to match that of the training set
- NEFTune did not improve summarization performance noticeably
- Tuning all modules (as opposed to attention modules) yielded slightly better results
  

## Classification
**Dataset Used:** Samsum

**Rank:** 8


|Training samples (fraction) | Zephyr-7B-β     | Zephyr-7B-β w/ NEFTune  | Zephyr-7B-β w/ Full Module Tuning | Zephyr-7B-β w/ NEFTune + Full Module Tuning |
|:--------------------------:|:---------------:|:-----------------------:|:---------------------------------:|:-------------------------------------------:|
|266   (2.5%)                |46.05            |49.61                    |65.36                              |67.23                                        |
|533   (5%)                  |55.66            |60.33                    |72.26                              |72.94                                        |
|1066  (10%)                 |66.48            |64.65                    |73.29                              |72.82                                        |
|2666  (25%)                 |66.73            |68.04                    |74.27                              |75.85                                        |
|5332  (50%)                 |69.54            |72.10                    |74.83                              |74.40                                        |
|10664 (100%)                |74.90            |72.93                    |77.76                              |77.86                                        |

- NEFTune boosted performance in low-data regimes
- Tuning all modules has achived ~10x sample efficiency and better better performance at 100% traning fraction 