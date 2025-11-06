## Environment Setup

Python 3.10 or later is required.

Setting up your GEPA development environment requires you to fork the GEPA repository and clone it locally.
If you are not familiar with the GitHub fork process, please refer to [Fork a repository](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo). After creating the fork, clone
it to your local development device:

```shell
git clone https://github.com/gepa-ai/gepa
cd gepa
```

Next, we must set up a Python environment with the correct dependencies. There are two recommended ways to set up the
dev environment.

### [Recommended] Set Up Environment Using uv

[uv](https://github.com/astral-sh/uv) is a rust-based Python package and project manager that provides a fast
way to set up the development environment. First, install uv by following the
[installation guide](https://docs.astral.sh/uv/getting-started/installation/).

After uv is installed, in your working directory (`gepa/`), run:

```shell
uv sync --extra dev --python 3.11
```

Then you are all set!

To verify that your environment is set up successfully, run some unit tests:

```shell
uv run pytest tests/
```

Note: You need to use the `uv run` prefix for every Python command, as uv creates a Python virtual
environment and `uv run` points the command to that environment. For example, to execute a Python script you will need
`uv run python script.py`.

### Set Up Environment Using conda + pip

You can also set up the virtual environment via conda + pip, which takes a few extra steps but offers more flexibility. Before starting,
make sure you have conda installed. If not, please follow the instructions
[here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

To set up the environment, run:

```shell
conda create -n gepa-dev python=3.11
conda activate gepa-dev
pip install -e ".[dev]"
```

Then verify the installation by running some unit tests:

```shell
pytest tests/
```

## Code Linting with Ruff
We follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) and use `ruff` for both linting and formatting. To ensure consistent code quality, we use pre-commit hooks that automatically check and fix common issues.


First you need to set up the pre-commit hooks (do this once after cloning the repository):

```shell
uv run pre-commit install
```

Then stage and commit your changes. When you run `git commit`, the pre-commit hook will be
automatically run.

```shell
git add .
git commit -m "your commit message"
```

If the hooks make any changes, you'll need to stage and commit those changes as well.

You can also run the hooks manually:

- Check staged files only:

  ```shell
  uv run pre-commit run
  ```

- Check specific files:

  ```shell
  uv run pre-commit run --files path/to/file1.py path/to/file2.py
  ```

Please ensure all pre-commit checks pass before creating your pull request. If you're unsure about any
formatting issues, feel free to commit your changes and let the pre-commit hooks fix them automatically.

## Type Checking with Pyright
Run Pyright before opening a pull request to catch type regressions early:

```shell
uv run pyright
```

You can target specific modules while iterating:

```shell
uv run pyright src/gepa/strategies/
```
