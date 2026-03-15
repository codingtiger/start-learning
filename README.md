# start-learning

## Python Environment

This project uses:

- `pyenv` to manage the Python interpreter version
- `uv` to create and manage the project virtual environment
- a local `.venv/` for project dependencies

The project pins Python with `.python-version`:

```text
3.14.3
```

## Setup

Install the required Python version with `pyenv` if needed:

```bash
pyenv install 3.14.3
```

Create the virtual environment with `uv`:

```bash
cd /Users/guoshuang.dong/workspace/ai-work/start-learning
uv venv --python 3.14.3 .venv
```

Install dependencies:

```bash
uv pip install --python .venv/bin/python -r requirements.txt
```

Activate the environment when working in a shell:

```bash
source .venv/bin/activate
python --version
```

## PyCharm Compatibility

PyCharm's package management tool expects `pip` to be importable from the selected interpreter.

If you recreate `.venv` with `uv`, run the following once so PyCharm can inspect installed packages correctly:

```bash
.venv/bin/python -m ensurepip --upgrade
```

You can verify it with:

```bash
.venv/bin/python -m pip --version
```
