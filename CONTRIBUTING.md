## Contributing

If you would like to contribute to this project, we recommend following the ["fork-and-pull" Git workflow](https://www.atlassian.com/git/tutorials/comparing-workflows/forking-workflow).

1.  **Fork** the repo on GitHub
2.  **Clone** the project to your own machine
3.  **Commit** changes to your own branch
4.  **Push** your work back up to your fork
5.  Submit a **Pull request** so that we can review your changes

NOTE: Be sure to merge the latest from "upstream" before making a pull request!

### Set Up Dev Environment

<details>
<summary>1. Clone Repo</summary>

```shell
git clone https://github.com/georgian-io/LLM-Finetuning-Toolkit.git
cd LLM-Finetuning-Toolkit/
```

</details>

<details>
<summary>2. Install Dependencies</summary>
<details>
<summary>Install with Docker [Recommended]</summary>

```shell
docker build -t llm-toolkit
```

```shell
# CPU
docker run -it llm-toolkit
# GPU
docker run -it --gpus all llm-toolkit
```

</details>

<details>
<summary>Poetry (recommended)</summary>

See poetry documentation page for poetry [installation instructions](https://python-poetry.org/docs/#installation)

```shell
poetry install
```

</details>
<details>
<summary>pip</summary>
We recommend using a virtual environment like `venv` or `conda` for installation

```shell
pip install -e .
```

</details>
</details>

### Checklist Before Pull Request (Optional)

1. Use `ruff check --fix` to check and fix lint errors
2. Use `ruff format` to apply formatting

NOTE: Ruff linting and formatting checks are done when PR is raised via Git Action. Before raising a PR, it is a good practice to check and fix lint errors, as well as apply formatting.

### Releasing

To manually release a PyPI package, please run:

```shell
make build-release
```

Note: Make sure you have a pypi token for this [PyPI repo](https://pypi.org/project/llm-toolkit/).
