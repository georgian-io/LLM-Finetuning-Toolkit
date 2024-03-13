---
sidebar_position: 1
---

# General Structure

The configuration file has a hierarchical structure with the following main sections:

- `save_dir`: The directory where the experiment results will be saved.
- [`ablation`](/docs/configuration/ablation): Settings for ablation studies.
- [`data`](/docs/configuration/data): Configuration for data ingestion.
- [`model`](/docs/configuration/model): Model definition and settings.
- [`lora`](/docs/configuration/lora): Configuration for LoRA (Low-Rank Adaptation).
- [`training`](/docs/configuration/training): Settings for the training process.
- [`inference`](/docs/configuration/inference): Configuration for the inference stage.

Each section contains subsections and parameters that fine-tune the behavior of the toolkit. Let's dive into each section in more detail.
