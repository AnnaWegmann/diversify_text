# diversify-text

This package helps you generate stylistically diverse paraphrases of your own texts using huggingface transformer models locally.

```bash
pip install diversify-text
```

**[Full documentation](https://annawegmann.github.io/diversify_text/)**

## Table of contents

- [Usage](#usage)
  - [Single text](#single-text)
  - [Control number of paraphrases](#control-number-of-paraphrases)
  - [Caching](#caching)
  - [Using the class directly](#using-the-class-directly)
  - [List of texts](#list-of-texts)
  - [Customising the TinyStyler style bank](#customising-the-tinystyler-style-bank)
- [Install](#install)
- [Contributing](#contributing)
  - [Development setup](#development-setup)
  - [Running tests](#running-tests)
  - [Working with uv](#working-with-uv)
  - [Building docs locally](#building-docs-locally)

## Usage

For file inputs (CSV, TSV, TXT), output options, punctuation splitting, and creating custom methods, see the [full usage guide](https://annawegmann.github.io/diversify_text/usage.html).

### Single text

```python
from diversify_text import diversify

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
results = diversify("Some text.", n=3)
```

```
[{"original": "Some text.", "paraphrases": ["...", "...", "..."]}]
```

### Caching

The `diversify()` function automatically caches loaded models between calls.
The generation model and the semantic filter are cached independently, so
toggling `semantic_filter` does not reload the generation model and vice
versa. Call `clear_cache()` to drop cached models and allow memory to be reclaimed when possible:

```python
from diversify_text import clear_cache

clear_cache()
```

### Using the class directly

You can also instantiate a `Diversifier` yourself for full control over the
model lifecycle:

```python
from diversify_text import Diversifier

div = Diversifier(device="cuda", methods=["tinystyler"])

batch_1 = div.diversify(texts_1, n=5)
batch_2 = div.diversify(texts_2, n=5)
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

### Customising the TinyStyler style bank

TinyStyler generates each paraphrase by conditioning on a *style example* — a short sentence that demonstrates the target writing style. The style bank is the list of such examples that get cycled through when producing multiple paraphrases.

The default bank is a dictionary mapping style labels to lists of example sentences (drawn from the CORE corpus). You can replace or extend it by passing a custom bank via `method_kwargs`.

A style bank can be a `dict[str, list[str]]` or a `list[list[str]]`:

```python
from diversify_text import diversify
from diversify_text.styles import DEFAULT_STYLE_BANK

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

`DEFAULT_STYLE_BANK` is exported from `diversify_text.styles` so you can build on it:

```python
from diversify_text.styles import DEFAULT_STYLE_BANK

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
from diversify_text import Diversifier
from diversify_text.method import DiversificationMethod


class MyMethod(DiversificationMethod):
    name = "my_method"

    def generate(self, texts, *, n, max_new_tokens, temperature, top_p, **kwargs):
        return [[f"{text} :: variant {i}" for i in range(n)] for text in texts]


results = Diversifier(methods=[MyMethod()]).diversify("Hello", n=3)
```

```
[{"original": "Hello", "paraphrases": ["Hello :: variant 0", "Hello :: variant 1", "Hello :: variant 2"]}]
```

## Install

```bash
pip install diversify-text
```

Requires Python 3.10+.

## Contributing

### Development setup

> [!NOTE]
> You must have **uv** installed.
> Full installation guide: <https://docs.astral.sh/uv/getting-started/installation/>

```bash
git clone https://github.com/AnnaWegmann/diversify_text.git
cd diversify_text
uv sync --group dev
source .venv/bin/activate
```

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

### Building docs locally

```bash
uv sync --group docs
sphinx-build -b html docs docs/_build/html
open docs/_build/html/index.html
```
