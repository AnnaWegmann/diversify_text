# diversify-text

This package helps you generate stylistically diverse paraphrases of your own texts using huggingface transformer models locally.

```bash
pip install diversify-text
```

**[Full documentation](https://annawegmann.github.io/diversify_text/)**

## Table of contents

- [diversify-text](#diversify-text)
  - [Table of contents](#table-of-contents)
  - [Usage](#usage)
    - [Single text](#single-text)
    - [Control number of styles](#control-number-of-styles)
    - [Pick styles from the bank](#pick-styles-from-the-bank)
    - [Bring your own style examples](#bring-your-own-style-examples)
    - [Repeats](#repeats)
    - [Prompting method](#prompting-method)
    - [Zero-shot method](#zero-shot-method)
    - [Caching](#caching)
    - [Using the class directly](#using-the-class-directly)
    - [List of texts](#list-of-texts)
    - [Evaluating paraphrases](#evaluating-paraphrases)
    - [Creating a custom method](#creating-a-custom-method)
  - [Install](#install)
  - [Contributing](#contributing)
    - [Development setup](#development-setup)
    - [Running tests](#running-tests)
    - [Working with uv](#working-with-uv)
      - [Adding packages with `uv add`](#adding-packages-with-uv-add)
      - [Adding packages to the dev group](#adding-packages-to-the-dev-group)
      - [Switching between dev and standard mode](#switching-between-dev-and-standard-mode)
      - [Best practice: run `uv lock -U`](#best-practice-run-uv-lock--u)
    - [Building docs locally](#building-docs-locally)
  - [Citation](#citation)

## Usage

<!-- quickstart-start -->

For file inputs (CSV, TSV, TXT), output options, and punctuation splitting, see the [full usage guide](https://annawegmann.github.io/diversify_text/usage.html).

### Single text

```python
from diversify_text import diversify

results = diversify("The experiment was conducted in a controlled lab setting.")
```

```python
[{
    "original": "The experiment was conducted in a controlled lab setting.",
    "paraphrases": [
        {"style": "informal", "text": "yeah but the experiment was done in a controlled lab setting."},
        {"style": "formal", "text": "The experiment was conducted in a controlled lab setting, I believe."},
        {"style": "question", "text": "The experiment was conducted in a controlled lab setting? I don't see how this could be a problem."},
        {"style": "question_answer_forum", "text": "Isn't it just because the experiment was conducted in a controlled lab setting?"},
        {"style": "discussion_forum", "text": "I think it was done in a controlled lab setting. I'm not sure how the experiment went though, but"},
    ]
}]
```

### Control number of styles

```python
results = diversify("Some text.", n=3)
```

```python
[{"original": "Some text.", "paraphrases": [
    {"style": "informal", "text": "..."},
    {"style": "formal", "text": "..."},
    {"style": "question", "text": "..."},
]}]
```

`n` is the number of distinct styles (default 5), drawn from the active method's style bank in order — one paraphrase per style. Requesting more styles than the bank contains raises an error.

Each method can have its own bank. The default method (TinyStyler) uses a small bank of styles it demonstrably handles (`informal`, `formal`, `question`, ...); the prompting method uses the larger default bank (dialects, registers, historical English, ...). The [styles page](https://annawegmann.github.io/diversify_text/styles.html) lists all of them.

### Pick styles from the bank

Select specific built-in styles with `styles`, by name and/or by (0-based) bank index:

```python
results = diversify(
    "The experiment was conducted in a controlled lab setting.",
    styles=["question", "personal_blog"],
)

# indices work too — handy for trying things without knowing the names
results = diversify(
    "The experiment was conducted in a controlled lab setting.",
    styles=[0, 7, "obama"],
)
```

Unknown names and out-of-range indices raise an error listing what is available. Note that indices follow bank order, which may change between releases as the bank is curated — names are the stable way to pin a style.

A few styles that are too far from contemporary English to be useful defaults (`old_english`, `middle_english`) live outside the regular bank: they are selectable by name only, have no index, and are never picked by `n`.

There is also a set of surface-level styles (e.g., `all_caps`, `lowercase`, `passive_voice`), defined by example texts. They are selectable by name only as well:

```python
diversify(
    "The experiment was conducted in a controlled lab setting.",
    styles=["all_caps", "active_voice"],
    method="prompting"
)
```
```python
[{'original': 'The experiment was conducted in a controlled lab setting.',
  'paraphrases': [
    {'style': 'all_caps', 'text': 'THE EXPERIMENT TOOK PLACE IN A CONTROLLED LAB ENVIRONMENT.'}, 
    {'style': 'active_voice', 'text': 'The researchers conducted the experiment in a controlled lab setting.'}
  ]
}]

```

### Bring your own style examples

Pass `style_texts` to define target styles with your own texts. A flat list is one style; a list of lists is several styles; a dict maps style names to example sets:

```python
# one style, defined by its example texts
results = diversify(
    "The experiment was conducted in a controlled lab setting.",
    style_texts=[
        "We found something really interesting — check this out!",
        "You won't believe how well this worked!",
    ],
)

# several styles, named
results = diversify(
    "The experiment was conducted in a controlled lab setting.",
    style_texts={
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

`styles` and `style_texts` can be combined in one call (bank styles come first in the output). `n` cannot be combined with either — the number of styles is already determined, so passing `n` raises an error.

### Repeats

`repeats` controls how many paraphrases are generated *per style* (default 1). With more than one repeat, the output interleaves the styles:

```python
results = diversify(
    "The experiment was conducted in a controlled lab setting.",
    styles=["informal", "question"],
    repeats=2,
)
```

```python
# styles interleave: informal, question, informal, question
[{
    "original": "The experiment was conducted in a controlled lab setting.",
    "paraphrases": [
        {"style": "informal", "text": "..."},
        {"style": "question", "text": "..."},
        {"style": "informal", "text": "..."},
        {"style": "question", "text": "..."},
    ]
}]
```

### Prompting method

The default style transfer method is [TinyStyler](https://huggingface.co/tinystyler/tinystyler). Alternatively, use the prompting method, which generates paraphrases via a causal language model (default: [SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B)) with the style examples inserted into a few-shot style transfer prompt:

```python
results = diversify(
    "The experiment was conducted in a controlled lab setting.",
    method="prompting",
    style_texts={
        "academic": [
            "The results demonstrate a statistically significant effect.",
            "Participants were randomly assigned to one of two conditions.",
        ],
    },
)
```

Only prompts that take style example texts are supported — every method receives the same input and produces the same output.

Pick a different model with `model=` (any HuggingFace causal LM that can handle instructions):

```python
results = diversify(
    "The experiment was conducted in a controlled lab setting.",
    method="prompting",
    model="Qwen/Qwen3-4B-Instruct-2507",
)
```

This works for the prompting and zero-shot methods; TinyStyler has a fixed model, so passing `model` with it raises an error.

### Zero-shot method

The `zero_shot` method defines styles by rewrite *instructions* instead of example texts. It has its own style bank of instruction styles:

```python
results = diversify(
    "The experiment was conducted in a controlled lab setting.",
    method="zero_shot",
    styles=["formal", "caps"],
)
```

With this method, `style_texts` are instructions — exactly one per style. An instruction can place the input text itself with `[DOCUMENT SEGMENT]`; otherwise the text is appended at the end:

```python
results = diversify(
    "The experiment was conducted in a controlled lab setting.",
    method="zero_shot",
    style_texts={"pirate": ["Rewrite the text as an old-timey pirate would say it."]},
)
```

### Caching

The `diversify()` function automatically caches loaded models between calls. The generation model and the semantic filter are cached independently, so toggling `semantic_filter` does not reload the generation model and vice versa. Call `clear_cache()` to drop cached models and allow memory to be reclaimed when possible:

```python
from diversify_text import clear_cache

clear_cache()
```

### Using the class directly

You can also instantiate a `Diversifier` yourself for full control over the model lifecycle:

```python
from diversify_text import Diversifier

div = Diversifier(device="cuda", method="tinystyler")

batch_1 = div.diversify(texts_1, styles=["informal", "question"])
batch_2 = div.diversify(texts_2, style_texts=my_examples)
```

### List of texts

```python
results = diversify([
    "The experiment was conducted in a controlled lab setting.",
    "She graduated from MIT in 2019.",
])
```

```python
[
    {"original": "The experiment ...", "paraphrases": [{"style": "informal", "text": "..."}, ...]},
    {"original": "She graduated ...", "paraphrases": [{"style": "informal", "text": "..."}, ...]},
]
```
### Evaluating paraphrases

Score paraphrases against the originals they came from. The originals are
already in the output, so nothing needs to be passed in again:

```python
from diversify_text import diversify, evaluate

results = diversify("The experiment was conducted in a controlled lab setting.")

results.evaluate()      # method on the output
evaluate(results)       # identical, as a function
```

Five metrics run by default: `style_similarity` (how similarly two texts
are *written*), `bertscore` and `mis` (meaning overlap), `rouge` and
`chrf` (word and character overlap).

Pick a subset with `metrics`, configure them with `metric_kwargs`:

```python
results.evaluate(
    metrics=["bertscore", "rouge"],
    metric_kwargs={
        "bertscore": {"model": "microsoft/deberta-xlarge-mnli"},
        "rouge": {"variants": ["rouge1", "rougeL"]},
    },
)
```

`style_similarity` and `bertscore` accept a `model`; the others have
fixed ones. `variants` selects which sub-scores are reported.

`granularity` sets the level of detail — `"dataset"` (default), `"text"`,
`"pair"`, or `"all"`. Results can be written out with `.to_jsonl(path)`.

Output written to disk returns a `Path`, which has no `.evaluate()` —
pass the path to `evaluate()` instead.

### Creating a custom method

```python
from diversify_text import Diversifier
from diversify_text.method import DiversificationMethod


class MyMethod(DiversificationMethod):
    name = "my_method"

    def generate(self, texts, style_dict, *, max_new_tokens, temperature, top_p, **kwargs):
        # style_dict maps each target style name to its example texts,
        # e.g. {"scottish_english": ["The boat I had, was a seventy-two foot boat...", ...]}.
        # It is resolved by the core from the caller's `styles` / `style_texts`.
        return [[f"{text} :: {name}" for name in style_dict] for text in texts]


results = Diversifier(method=MyMethod()).diversify("Hello", styles=["scottish_english", "opinion"])
```

```python
[{"original": "Hello", "paraphrases": [
    {"style": "scottish_english", "text": "Hello :: scottish_english"},
    {"style": "opinion", "text": "Hello :: opinion"},
]}]
```

A method returns `list[list[str]]` — for each input text, one generated string per style in `style_dict` order. The core attaches the style labels to the output and runs the method once per repeat, so custom methods stay simple and stateless.

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

<!-- citation-start -->

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

<!-- citation-end -->