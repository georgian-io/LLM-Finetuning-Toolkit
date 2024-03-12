---
sidebar_position: 2
---

# Artefact Outputs

This config will run finetuning and save the results under directory `./experiment/[unique_hash]`. Each unique configuration will generate a unique hash, so that our tool can automatically pick up where it left off. For example, if you need to exit in the middle of the training, by relaunching the script, the program will automatically load the existing dataset that has been generated under the directory, instead of doing it all over again.

After the script finishes running you will see these distinct artifacts:

**`/config/config.yml`**: copy of the config file used for this experiment

**`/dataset/dataset.pkl`**: generated pkl file in huggingface Dataset format

**`/model/*`**: model weights saved using huggingface format

**`/results/results.csv`**: csv of prompt, ground truth, and predicted values

**`/qa/qa.csv`**: csv of quality assurance unit tests (e.g. vector similarity between gold and predicted output)
