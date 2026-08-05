# diversify-text

This package helps you generate stylistically diverse paraphrases of your own texts using huggingface transformer models locally.

```bash
pip install diversify-text
```

**[Full documentation](https://annawegmann.github.io/diversify_text/)**

## Table of contents

- [How it works](#how-it-works)
- [Basic usage](#basic-usage)
- [Picking styles from the bank](#picking-styles-from-the-bank)
- [Bring your own style examples](#bring-your-own-style-examples)
- [Repeats](#repeats)
- [Choosing the style transfer method](#choosing-the-style-transfer-method)
- [Semantic filter](#semantic-filter)
- [Caching](#caching)
- [Using the class directly](#using-the-class-directly)
- [Citation](#citation)
- [Install](#install)
- [Contributing](#contributing)
  - [Development setup](#development-setup)
  - [Running tests](#running-tests)
  - [Working with uv](#working-with-uv)
  - [Building docs locally](#building-docs-locally)

<!-- quickstart-start -->

## How it works

`diversify_text` is built around a single idea: **a style is defined by a set of example texts**. Every call takes your input text plus one or more style example sets, and rewrites the input in the style that the examples demonstrate. Which style transfer method does the rewriting (TinyStyler by default) is a background detail — the input/output contract is always the same:

- **Input:** your text(s) and, per target style, a set of example texts.
- **Output:** per input text, one paraphrase per target style, each labeled with the style that produced it.

If you don't provide your own styles, the built-in style bank supplies default ones, so a plain call already produces stylistically diverse paraphrases.

## Basic usage

```python
from diversify_text import diversify

results = diversify("The experiment was conducted in a controlled lab setting.")
```

```python
[{
    "original": "The experiment was conducted in a controlled lab setting.",
    "paraphrases": [
        {"style": "informal", "text": "the experiment was in a controlled lab setting so it didnt suck..."},
        {"style": "obama", "text": "Well it was a controlled lab setting that the experiment was conducted in."},
        {"style": "question", "text": "Did you know that the experiment was conducted in a controlled lab setting? It was a re-test."},
        {"style": "formal", "text": "I heard the experiment was conducted in a controlled lab setting."},
        {"style": "song_lyrics", "text": "I mean, this experiment was conducted in a controlled lab setting, so that was a good thing."},
    ]
}]
```

Each paraphrase corresponds to one style from the built-in style bank — by default the first five. Ask for more distinct styles with `n`:

```python
results = diversify("The experiment was conducted in a controlled lab setting.", n=10)
```

`n` always means *number of distinct styles*, drawn from the bank in order. Requesting more styles than the bank contains raises an error — you never silently get the same style twice.

For file inputs (CSV, TSV, TXT), output options, and punctuation splitting, see the [full usage guide](https://annawegmann.github.io/diversify_text/usage.html).

## Picking styles from the bank

Select specific built-in styles with `styles`, by name and/or by (0-based) bank index:

```python
results = diversify(
    "The experiment was conducted in a controlled lab setting.",
    styles=["recipe", "personal_blog"],
)

# indices work too — handy for trying things without knowing the names
results = diversify(
    "The experiment was conducted in a controlled lab setting.",
    styles=[0, 7, "recipe"],
)
```

Unknown names and out-of-range indices raise an error listing what is available. Note that indices follow bank order, which may change between releases as the bank is curated — names are the stable way to pin a style.

## Bring your own style examples

Pass `style_examples` to define target styles with your own texts. A flat list is one style; a list of lists is several styles; a dict maps style names to example sets:

```python
# one style, defined by its example texts
results = diversify(
    "The experiment was conducted in a controlled lab setting.",
    style_examples=[
        "We found something really interesting — check this out!",
        "You won't believe how well this worked!",
    ],
)

# several styles, named
results = diversify(
    "The experiment was conducted in a controlled lab setting.",
    style_examples={
        "academic": [
            "The results demonstrate a statistically significant effect.",
            "Participants were randomly assigned to one of two conditions.",
        ],
        "enthusiastic": [
            "We found something really interesting — check this out!",
            "You won't believe how well this worked!",
        ],
    },
)
```

```python
[{
    "original": "The experiment was conducted in a controlled lab setting.",
    "paraphrases": [
        {"style": "academic", "text": "The experiment was carried out under controlled laboratory conditions."},
        {"style": "enthusiastic", "text": "Guess what — we ran the whole experiment in a controlled lab, how cool is that!"},
    ]
}]
```

`styles` and `style_examples` can be combined in one call (bank styles come first in the output). `n` cannot be combined with either — the number of styles is already determined, so passing `n` raises an error.

## Repeats

`repeats` controls how many paraphrases are generated *per style* (default 1). With more than one repeat, the output interleaves the styles:

```python
results = diversify(
    "The experiment was conducted in a controlled lab setting.",
    styles=["recipe", "personal_blog"],
    repeats=2,
)
```

```python
# styles interleave: recipe, personal_blog, recipe, personal_blog
[{
    "original": "The experiment was conducted in a controlled lab setting.",
    "paraphrases": [
        {"style": "recipe", "text": "..."},
        {"style": "personal_blog", "text": "..."},
        {"style": "recipe", "text": "..."},
        {"style": "personal_blog", "text": "..."},
    ]
}]
```

## Choosing the style transfer method

The style examples stay the same regardless of which method rewrites your text. The default method is [TinyStyler](https://huggingface.co/tinystyler/tinystyler), which conditions on the example texts via authorship embeddings. Alternatively, the `prompting` method inserts the example texts into a few-shot style transfer prompt for a causal language model (default: [SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B)):

```python
results = diversify(
    "The experiment was conducted in a controlled lab setting.",
    methods=["prompting"],
    style_examples={
        "academic": [
            "The results demonstrate a statistically significant effect.",
            "Participants were randomly assigned to one of two conditions.",
        ],
    },
)
```

Only prompts that take style example texts are supported — every method receives the same input (your text plus style example sets) and produces the same output.

## Semantic filter

Enable the semantic filter to score each paraphrase with the [Mutual Implication Score](https://huggingface.co/s-nlp/Mutual_Implication_Score) model and automatically select the best candidate above a minimum score. Candidates are compared per style, so the filter improves semantic fidelity without reducing stylistic diversity:

```python
results = diversify(
    "The experiment was conducted in a controlled lab setting.",
    semantic_filter=True,
)
```

## Caching

The `diversify()` function automatically caches loaded models between calls. The generation model and the semantic filter are cached independently, so toggling `semantic_filter` does not reload the generation model and vice versa. Call `clear_cache()` to release cached model references when you are done. On CUDA devices, memory may remain reserved by the underlying framework's caching allocator and be reused in future calls rather than immediately returned to the OS/driver:

```python
from diversify_text import clear_cache

clear_cache()
```

## Using the class directly

You can also instantiate a `Diversifier` yourself for full control over the model lifecycle:

```python
from diversify_text import Diversifier

div = Diversifier(device="cuda", methods=["tinystyler"])

batch_1 = div.diversify(texts_1, styles=["recipe", "personal_blog"])
batch_2 = div.diversify(texts_2, style_examples=my_examples)
```

## Citation

If you use `diversify` in your research, we are happy about a citation (placeholder currently).

```bibtex
@inproceedings{wegmann2026diversify,
    title = {diversify_text: An Amazing Library for Text Diversification},
    author = {Wegmann, Anna and Others},
    url={https://github.com/AnnaWegmann/diversify_text},
    year = {2026},
}
```

<!-- quickstart-end -->

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
