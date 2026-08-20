# Quickstart

## Installation

```bash
pip install diversify-text
```

## Usage

<!-- The usage content lives in README.md (between the
     quickstart-start/quickstart-end markers) so it is maintained
     in exactly one place. -->

```{include} ../README.md
:start-after: <!-- quickstart-start -->
:end-before: <!-- quickstart-end -->
```

## Semantic filter

Enable the semantic filter to score each paraphrase with the
[Mutual Implication Score](https://huggingface.co/s-nlp/Mutual_Implication_Score)
model and automatically select the best candidate above a minimum score.
Candidates are compared per style, so the filter improves semantic
fidelity without reducing stylistic diversity. Note the generation cost:
up to styles × `repeats` × candidate rounds model calls:

```python
results = diversify(
    "The experiment was conducted in a controlled lab setting.",
    semantic_filter=True,
)
```

```python
[{
    "original": "The experiment was conducted in a controlled lab setting.",
    "paraphrases": [
        {"style": "digital_communication", "text": "..."},
        {"style": "informational", "text": "..."},
        {"style": "spoken_communication", "text": "..."},
        {"style": "lyrical", "text": "..."},
        {"style": "scottish_english", "text": "..."},
    ]
}]
```

```{include} ../README.md
:start-after: <!-- citation-start -->
:end-before: <!-- citation-end -->
```
