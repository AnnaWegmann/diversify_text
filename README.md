# diversify

This package helps you generate stylistically diverse paraphrases of your own texts using huggingface transformer models locally.

## Install

> [!NOTE]
> You must have **uv** installed before running `uv sync`.
> Full installation guide: <https://docs.astral.sh/uv/getting-started/installation/>

After installing `uv` on your system, you can now follow either **development mode** or **standard installation** depending on your use case.

### Development mode

Follow these steps to set up the project for development.

- Clone the repo
- Install all dependencies required for development mode:
   ```bash
   uv sync --group dev
   ```
- Activate the Python environment created by `uv`:
   ```bash
   source .venv/bin/activate
   ```

### Standard installation

To use the library directly:

- Clone the repo
- Install all dependencies required for standard mode:
   ```bash
   uv sync --no-group dev
   ```
- Activate the Python environment created by `uv`:
   ```bash
   source .venv/bin/activate
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

# Use specific methods (plugins)
results = diversify(
    "Some text.",
    n_styles=6,
    methods=["tinystyler", "echo"],  # styles are split across methods
)
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

div = Diversifier(device="cuda", methods=["tinystyler"])

# The model is loaded once and reused
batch_1 = div.diversify(texts_1, n_styles=5)
batch_2 = div.diversify(texts_2, n_styles=5)
```

### Creating a custom method

```python
from diversify import Diversifier
from diversify.method.base import DiversificationMethod


class MyMethod(DiversificationMethod):
    name = "my_method"

    def generate(self, texts, *, n_styles, max_new_tokens, temperature, top_p, **kwargs):
        return [[f"{text} :: variant {i}" for i in range(n_styles)] for text in texts]


div = Diversifier(methods=[MyMethod()])
results = div.diversify("Hello", n_styles=3)
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

## Working with uv

### Adding packages with `uv add`

To add packages to your project, always use `uv add` rather than `uv pip install`. This ensures that your dependencies are properly managed and recorded in your `pyproject.toml`.

```bash
uv add <package-name>
```

### Adding packages to the dev group

If you need to add a package specifically for your development environment:

```bash
uv add --group dev <package-name>
```

### Switching between dev and standard mode

After you are done with testing and want to go back to standard mode, you can remove the dev-only packages:

```bash
uv sync --no-group dev
```

This will disable all additional groups and just load your main project dependencies.

### Best practice: run `uv lock -U`

Whenever you upgrade, downgrade, or change versions of packages, it's good practice to run:

```bash
uv lock -U
```

This updates your lock file to ensure all versions are consistent and everything is in sync.

## Project structure

```
diversify/
├── diversify/              # Python package
│   ├── __init__.py
│   ├── core.py             # Diversifier orchestration & diversify() function
│   ├── method/         # Pluggable diversification methods
│   │   ├── __init__.py
│   │   ├── base.py         # DiversificationMethod abstract class
│   │   ├── registry.py     # Method registry + default registrations
│   │   ├── echo.py         # Fallback method
│   │   └── tinystyler/     # TinyStyler method submodule
│   │       ├── __init__.py
│   │       ├── method.py # TinyStyler-backed method
│   │       └── model.py    # TinyStyler model wrapper
├── tests/
│   ├── __init__.py
│   └── test_diversify.py
├── example_scripts/        # Runnable examples + example data
│   ├── data/
│   │   └── bios_400.csv
│   ├── utils/
│   │   └── load_bios.py
│   └── run_diversify_bios.py
├── legacy_code/            # Original scripts (reference only)
├── pyproject.toml
└── README.md
```
