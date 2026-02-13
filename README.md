# diversify

Generate stylistic paraphrases of texts using local transformer models.

## Setup with uv

This project uses [uv](https://docs.astral.sh/uv/) for dependency and environment management.

### Install uv

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# or via Homebrew
brew install uv
```

After installing, restart your terminal (or run `source ~/.bashrc` / `source ~/.zshrc`).

### Create the environment and install

```bash
# Create a virtual environment with uv (uses .venv/ by default)
uv venv

# Activate it
source .venv/bin/activate

# Sync all dependencies from pyproject.toml
uv pip install -e .
```

### For development (includes test dependencies)

```bash
uv pip install -e ".[dev]"
```

## Usage

```python
from diversify import diversify

# Single text
results = diversify("The experiment was conducted in a controlled lab setting.")

# List of texts
results = diversify([
    "The experiment was conducted in a controlled lab setting.",
    "She graduated from MIT in 2019.",
])

# pandas DataFrame
import pandas as pd
df = pd.DataFrame({"text": ["Hello world.", "How are you?"]})
results = diversify(df, text_column="text")

# Control number of paraphrases
results = diversify("Some text.", n_styles=10)
```

Each result is a dict:

```python
{
    "original": "The experiment was conducted in a controlled lab setting.",
    "paraphrases": [
        "They ran the experiment in a lab, everything nice and controlled.",
        "So basically they did this experiment in a lab — pretty standard setup.",
        # ... (n_styles total)
    ],
}
```

### Using the class directly (recommended when processing many texts)

```python
from diversify import Diversifier

div = Diversifier(device="cuda")

# The model is loaded once and reused
batch_1 = div.diversify(texts_1, n_styles=5)
batch_2 = div.diversify(texts_2, n_styles=5)
```

## Running tests

```bash
# Run all tests
uv run pytest

# Run a specific test file
uv run pytest tests/test_diversify.py

# Run a specific test class or method
uv run pytest tests/test_diversify.py::TestDiversifyOutput
uv run pytest tests/test_diversify.py::TestDiversifyOutput::test_single_text_returns_one_result
```

Tests are also individually runnable via PyCharm's built-in test runner (right-click any test class or method).

## Project structure

```
diversify/
├── diversify/              # Python package
│   ├── __init__.py
│   ├── core.py             # Diversifier class & diversify() function
│   └── tinystyler/         # TinyStyler model wrapper (subpackage)
│       ├── __init__.py
│       └── core.py
├── tests/
│   ├── __init__.py
│   └── test_diversify.py
├── data/                   # Sample data
├── legacy_code/            # Original scripts (reference only)
├── pyproject.toml
└── README.md
```

## Installing as a package (non-dev / end-user)

```bash
# From the repo root
pip install .

# Or directly from a git URL (once published)
# pip install git+https://github.com/<user>/diversify.git
```

This installs `diversify` into your current environment so you can `import diversify` from anywhere.
