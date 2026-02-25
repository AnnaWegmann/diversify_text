"""Default style bank for TinyStyler.

Each entry is a *style group*: a list of one or more example sentences that
together define a target writing style.  TinyStyler cycles through the groups
when producing multiple paraphrases, so having more groups than ``n_styles``
is fine — only the first ``n_styles`` groups (modulo the bank length) are used.

A style group can be a single string or a list of strings.  Multiple strings
let you give the model richer style examples.

Import and pass to the method via ``method_kwargs`` to override the default:

    from diversify.method.tinystyler import DEFAULT_STYLE_BANK

    custom_bank = DEFAULT_STYLE_BANK + [
        ["The data clearly indicate a statistically significant result."],
    ]
    results = diversify(
        texts,
        method_kwargs={"tinystyler": {"style_bank": custom_bank}},
    )
"""

DEFAULT_STYLE_BANK: list[list[str]] = [
    ["Dear Sir or Madam, I appreciate your thoughtful correspondence."],
    ["Hey, thanks a lot for your message. Really appreciate it!"],
    ["Objective: acknowledge receipt and express appreciation succinctly."],
    ["Yo, got your note. Super grateful for the heads-up."],
]
