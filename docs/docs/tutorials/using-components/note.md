---
sidebar_label: Important Note
sidebar_position: 5
---

# A Note on Using Individual Components

While this tutorial provides guidance on using individual components of the toolkit, it's important to acknowledge that it may be slightly more challenging compared to using the toolkit as a whole via the CLI. This is because most of the modules in the toolkit are designed to consume the `Config` object and the `DirectoryHelper` object, which encompass a broader scope than what is needed for each individual component.

The reason for this design choice is that the toolkit is primarily intended to be used as a CLI tool, and the authors elected to use the `Config` and `DirectoryHelper` objects as a single source of truth for the sake of convenience and consistency. By utilizing these objects, the CLI can easily manage configurations and directory structures across the entire pipeline.

However, we recognize that this approach may make it a bit more cumbersome to use individual components independently. Users may need to create and populate the `Config` and `DirectoryHelper` objects even if they only require a specific functionality offered by a particular module.

We encourage and welcome contributions from the community to help refactor and improve the usability of individual components. If you have ideas or are willing to put in the effort to make the components more modular and easier to use independently, we would greatly appreciate your contributions. By collaborating and refining the codebase, we can enhance the flexibility and adaptability of the toolkit.

If you're interested in contributing to this aspect of the project, please refer to the Contribution Guide for more information on how to get started. Your efforts can help make the toolkit more accessible and user-friendly for a wider range of use cases.

Despite the current limitations, we hope that this tutorial still provides valuable insights and guidance on leveraging the individual components of the toolkit. We appreciate your understanding and support as we continue to improve and evolve the project.
