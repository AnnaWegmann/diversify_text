# diversify

This package helps you generate stylistically diverse paraphrases of your own texts using huggingface transformer models locally.

## Table of contents

- [Usage](#usage)
  - [Single text](#single-text)
  - [Control number of paraphrases](#control-number-of-paraphrases)
  - [Using the class directly](#using-the-class-directly)
  - [List of texts](#list-of-texts)
  - [CSV / TSV file](#csv--tsv-file)
  - [TXT file](#txt-file)
  - [Controlling output location](#controlling-output-location)
  - [Punctuation splitting](#punctuation-splitting)
  - [Multiple methods](#multiple-methods)
  - [Customising the TinyStyler style bank](#customising-the-tinystyler-style-bank)
  - [Creating a custom method](#creating-a-custom-method)
- [Install](#install)
- [Development](#development)
  - [Running tests](#running-tests)
  - [Working with uv](#working-with-uv)
- [Project structure](#project-structure)

## Usage

### Single text

```python
from diversify import diversify

results = diversify("The experiment was conducted in a controlled lab setting.")
```

```
[{
    "original": "The experiment was conducted in a controlled lab setting.",
    "paraphrases": [
        "They ran the experiment in a controlled lab setting.",
        "The experiment took place in a controlled lab.",
        "A controlled lab was where the experiment was conducted.",
        "In a controlled lab, the experiment was carried out.",
        "The study was performed in a controlled lab environment.",
    ]
}]
```

### Control number of paraphrases

```python
results = diversify("Some text.", n_styles=3)
```

```
[{"original": "Some text.", "paraphrases": ["...", "...", "..."]}]
```

### Using the class directly

Recommended when processing texts across several calls — the model is loaded once and reused across calls.

```python
from diversify import Diversifier

div = Diversifier(device="cuda", methods=["tinystyler"])

batch_1 = div.diversify(texts_1, n_styles=5)
batch_2 = div.diversify(texts_2, n_styles=5)
```

### List of texts

```python
results = diversify([
    "The experiment was conducted in a controlled lab setting.",
    "She graduated from MIT in 2019.",
])
```

```
[
    {"original": "The experiment ...", "paraphrases": ["...", "...", ...]},
    {"original": "She graduated ...", "paraphrases": ["...", "...", ...]},
]
```

### CSV / TSV file

Reads the file and writes a JSONL file next to the input (`<input>_diversified.jsonl`).

```python
results = diversify("bios.csv", text_column="bio")
# writes bios_diversified.jsonl
```

Each line in the JSONL output is one JSON object:

```json
{"original": "Jane is a ...", "paraphrases": ["Jane works as a ...", "As a ..., Jane ..."]}
{"original": "John studied ...", "paraphrases": ["John was educated ...", "..."]}
```

### TXT file

Each non-empty line is treated as a separate text to diversify. Output is written to `<input>.jsonl`.

```python
results = diversify("texts.txt")
# writes texts.jsonl
```


### Controlling output location

By default, file inputs write output next to the input file and in-memory inputs (strings, lists) return a Python list. You can override this with `output_dir` and `output_name`:

```python
# Write output to a specific directory
results = diversify("bios.csv", text_column="bio", output_dir="/results")
# writes /results/bios_diversified.jsonl

# Also set a custom filename
results = diversify("bios.csv", text_column="bio", output_dir="/results", output_name="my_output")
# writes /results/my_output.jsonl

# Force a list input to write to disk instead of returning in-memory
results = diversify(["text one", "text two"], output_dir=".")
# writes ./diversified_output.jsonl
```

The `.jsonl` extension is always added automatically.

### Punctuation splitting

Splits each text into sentence segments internally before paraphrasing (improving quality on long texts), then reassembles the results. The output still contains one entry per original input text.

```python
results = diversify(["One sentence. Another one!"], split_on_punctuation=True, n_styles=2)
```

```
[{
    "original": "One sentence. Another one!",
    "paraphrases": [
        "A single sentence. Yet another one!",
        "One phrase. One more!",
    ]
}]
```

### Multiple methods

Styles are distributed evenly across methods.

```python
results = diversify("Some text.", n_styles=6, methods=["tinystyler", "echo"])
```

```
[{"original": "Some text.", "paraphrases": ["...", "...", "...", "...", "...", "Some text."]}]
#                                           |--- 4 from tinystyler ---|  |-- 2 from echo --|
```



### Customising the TinyStyler style bank

TinyStyler generates each paraphrase by conditioning on a *style example* — a short sentence that demonstrates the target writing style. The style bank is the list of such examples that get cycled through when producing multiple paraphrases.

The default bank is a dictionary mapping style labels to lists of example sentences (drawn from the CORE corpus). You can replace or extend it by passing a custom bank via `method_kwargs`.

A style bank can be a `dict[str, list[str]]` or a `list[list[str]]`:

```python
from diversify import diversify
from diversify.method.tinystyler import DEFAULT_STYLE_BANK

custom_bank = {
    "academic": ["The results demonstrate a statistically significant effect."],
    "enthusiastic": ["We found something really interesting — check this out!"],
    "telegraphic": ["Key finding: effect confirmed. Details follow."],
}

results = diversify(
    "The experiment was conducted in a controlled lab setting.",
    method_kwargs={"tinystyler": {"style_bank": custom_bank}},
)
```

`DEFAULT_STYLE_BANK` is exported from `diversify.method.tinystyler` so you can build on it:

```python
from diversify.method.tinystyler import DEFAULT_STYLE_BANK

extended_bank = {
    **DEFAULT_STYLE_BANK,
    "scientific": ["The data clearly indicate a statistically significant result."],
}
```

You can also select specific styles by key name with `styles`, instead of cycling through the entire bank.
The number of paraphrases is determined by the number of selected styles:

```python
results = diversify(
    "The experiment was conducted in a controlled lab setting.",
    method_kwargs={"tinystyler": {"styles": ["research_article", "personal_blog", "recipe"]}},
)
```

### Creating a custom method

```python
from diversify import Diversifier
from diversify.method import DiversificationMethod


class MyMethod(DiversificationMethod):
    name = "my_method"

    def generate(self, texts, *, n_styles, max_new_tokens, temperature, top_p, **kwargs):
        return [[f"{text} :: variant {i}" for i in range(n_styles)] for text in texts]


results = Diversifier(methods=[MyMethod()]).diversify("Hello", n_styles=3)
```

```
[{"original": "Hello", "paraphrases": ["Hello :: variant 0", "Hello :: variant 1", "Hello :: variant 2"]}]
```

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

## Development

### Running tests

```bash
# Run all tests
pytest

# Run a specific test file
pytest tests/test_core.py

# Run a specific test class or method
pytest tests/test_core.py::TestDiversifier
pytest tests/test_core.py::TestDiversifier::test_single_text_returns_one_result
```

Tests are also individually runnable via PyCharm's built-in test runner (right-click any test class or method).

### Working with uv

#### Adding packages with `uv add`

To add packages to your project, always use `uv add` rather than `uv pip install`. This ensures that your dependencies are properly managed and recorded in your `pyproject.toml`.

```bash
uv add <package-name>
```

#### Adding packages to the dev group

If you need to add a package specifically for your development environment:

```bash
uv add --group dev <package-name>
```

#### Switching between dev and standard mode

After you are done with testing and want to go back to standard mode, you can remove the dev-only packages:

```bash
uv sync --no-group dev
```

This will disable all additional groups and just load your main project dependencies.

#### Best practice: run `uv lock -U`

Whenever you upgrade, downgrade, or change versions of packages, it's good practice to run:

```bash
uv lock -U
```

This updates your lock file to ensure all versions are consistent and everything is in sync.

## Project structure

```
diversify/
├── diversify/                  # Python package
│   ├── __init__.py
│   ├── core.py                 # Diversifier class & diversify() function
│   ├── _input.py               # Input resolution & lazy file reading
│   ├── _output.py              # Output path resolution & JSONL writing
│   ├── _text.py                # Punctuation-based text splitting
│   └── method/                 # Pluggable diversification methods
│       ├── __init__.py
│       ├── base.py             # DiversificationMethod abstract class
│       ├── registry.py         # Method registry + default registrations
│       ├── echo.py             # Echo method (returns input unchanged)
│       └── tinystyler/         # TinyStyler method
│           ├── __init__.py
│           ├── method.py       # TinyStyler-backed method
│           ├── model.py        # TinyStyler model wrapper
│           └── styles.py       # Default style bank
├── tests/
│   ├── __init__.py
│   ├── fixtures.py             # Shared fake method implementations
│   ├── test_core.py            # Diversifier & diversify() tests
│   ├── test_input.py           # Input resolution tests
│   ├── test_output.py          # Output path & writer tests
│   └── test_text.py            # Punctuation splitting tests
├── example_scripts/            # Runnable examples + example data
│   ├── data/
│   │   └── bios_400.csv
│   ├── utils/
│   │   └── load_bios.py
│   └── run_diversify_bios.py
├── pyproject.toml
└── README.md
```
