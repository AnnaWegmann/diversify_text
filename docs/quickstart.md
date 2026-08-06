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
        {"style": "informal", "text": "the experiment was in a controlled lab setting so it didnt suck..."},
        {"style": "obama", "text": "Well it was a controlled lab setting that the experiment was conducted in."},
        {"style": "question", "text": "Can you explain the experiment? It was conducted in a controlled lab setting."},
        {"style": "formal", "text": "I heard the experiment was conducted in a controlled lab setting."},
        {"style": "song_lyrics", "text": "I mean, this experiment was conducted in a controlled lab setting, so that was a good thing."},
    ]
}]
```

```{include} ../README.md
:start-after: <!-- citation-start -->
:end-before: <!-- citation-end -->
```
