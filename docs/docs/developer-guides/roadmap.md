---
sidebar_position: 3
---

# Roadmap

The toolkit is an ongoing project, and we have a roadmap that outlines our plans for future development and improvements. This roadmap is intended to give you an idea of the direction we are heading and the features we aim to implement. However, please note that the roadmap is subject to change based on community feedback, new research findings, and evolving priorities.

:::note
We value the feedback and contributions from the community in shaping the future of the toolkit. If you have any suggestions, ideas, or feature requests, please don't hesitate to reach out to us through the appropriate channels.

We are excited about the future of the toolkit and the potential it holds for empowering researchers and developers in their NLP tasks. Stay tuned for updates and new releases as we work towards achieving these goals.
:::

## Documentation

- [ ] **Enhanced Documentation**: Further improve the documentation by adding more examples, tutorials, and use case scenarios. This will make it easier for users to understand and leverage the full potential of the toolkit.

## Code Quality

- [ ] **Unit Tests**: Introduce unit tests to speed up development cycle and spot bugs before we ship.

## Features

- [ ] **Multi-GPU support**: Enable multi-gpu training and inference for large models via `accelerate` library.
- [ ] **Distributed Training and Inference**: Introduce distributed training and inference capabilities to enable users to scale their finetuning and inference processes across multiple machines or clusters.
- [ ] **Guidance Support for Inference**: Enable more sophisticated structured output generation via `guidance` or `lmql` libraries.
- [ ] **Serving**: Provide code snippets and/or documentation to deploy saved weights to a serving instance/cluster on the cloud.
- [ ] **GUI-based Configuration Editor**: Develop a graphical user interface (GUI) for editing and managing configuration files. This will provide a more user-friendly alternative to editing YAML files directly.
- [ ] **Additional Quality Assurance (QA) Tests**: Expand the suite of QA tests to cover a wider range of quality metrics and evaluation criteria.
- [ ] **Report Generation for Ablation Studies**: Generate final report for ablation studies based on QA Test results and provide detailed performance results for top 5 models.
