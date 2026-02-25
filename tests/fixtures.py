"""Shared fake DiversificationMethod implementations for tests."""

from diversify.method import DiversificationMethod


class PrefixMethod(DiversificationMethod):
    """Returns prefixed paraphrases: ``<prefix>:<text>:<i>``."""

    name = "prefix"

    def __init__(self, prefix: str) -> None:
        self.prefix = prefix

    def generate(self, texts, *, n_styles, max_new_tokens, temperature, top_p, **kwargs):
        return [
            [f"{self.prefix}:{text}:{i}" for i in range(n_styles)]
            for text in texts
        ]


class FailingMethod(DiversificationMethod):
    """Always raises RuntimeError."""

    name = "failing"

    def generate(self, texts, *, n_styles, max_new_tokens, temperature, top_p, **kwargs):
        raise RuntimeError("boom")


class CountingMethod(DiversificationMethod):
    """Counts generate() calls and returns ``<text>:<i>`` paraphrases."""

    name = "counting"

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, texts, *, n_styles, max_new_tokens, temperature, top_p, **kwargs):
        self.calls += 1
        return [[f"{text}:{i}" for i in range(n_styles)] for text in texts]
