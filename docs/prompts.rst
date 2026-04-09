Prompt Templates
=================

The :doc:`prompting method <methods>` generates paraphrases by sending input
texts to a causal language model together with a prompt template.  This page
documents every built-in template and explains how to select or customise them.

.. contents:: On this page
   :local:
   :depth: 2


Zero-shot prompts
-----------------

Zero-shot templates describe the desired output style directly in the prompt.
They use a single placeholder, ``[DOCUMENT SEGMENT]``, which is replaced with
the input text at generation time.

All zero-shot templates are stored in
:data:`~diversify_text.method.prompting.prompts.ZS_PROMPT_BANK`.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Key
     - Description
   * - ``wikipedia_paraphrase``
     - Paraphrase in Wikipedia-style.
       From `Becker et al. (2024) <https://arxiv.org/abs/2401.16380>`_.
   * - ``finephrase_article``
     - Rewrite as a magazine-style feature article with an engaging lead and
       narrative storytelling.
   * - ``finephrase_commentary``
     - Summarise the text and add an expert commentary on its implications and
       limitations.
   * - ``finephrase_discussion``
     - Reformulate as a teacher-student dialogue that explains the key points.
   * - ``finephrase_explanation``
     - Rewrite with explicit scientific or logical explanations of the
       underlying concepts and mechanisms.
   * - ``finephrase_faq``
     - Convert into a comprehensive FAQ with self-contained answers.
   * - ``finephrase_math``
     - Create a mathematical word problem from the text's numerical data, with
       a step-by-step solution.
   * - ``finephrase_narrative``
     - Rewrite as a narrative emphasising temporal sequence and causal
       relationships.
   * - ``finephrase_table``
     - Organise the key information into a markdown table, then generate a
       question-answer pair based on it.
   * - ``finephrase_tutorial``
     - Rewrite as a step-by-step tutorial or instructional guide.
   * - ``humanize_llm-as-coauthor_original``
     - Humanise machine-generated text by introducing typos, slang, hashtags,
       and varied casing (no emojis).
       From `Zhang et al. (2024) <https://arxiv.org/abs/2401.05952>`_.
   * - ``formal_reif``
     - Rewrite the text in a more formal tone.
       From `Reif et al. (2022) <https://arxiv.org/abs/2109.03910>`_.
   * - ``simple_reif``
     - Rewrite the text in simpler language.
       From `Reif et al. (2022) <https://arxiv.org/abs/2109.03910>`_.
   * - ``simple_kew``
     - Simplify a complex sentence for non-native English speakers.
       From `Kew et al. (2023) <https://aclanthology.org/2023.emnlp-main.821.pdf>`_.
   * - ``complex_kew``
     - Rewrite a simple sentence in a more complex and sophisticated style.
   * - ``passive_reif``
     - Rewrite the text switching between active and passive voice.
       Few-shot prompt with examples.
       From `Reif et al. (2022) <https://arxiv.org/abs/2109.03910>`_.
   * - ``caps_reif``
     - Rewrite the text in ALL CAPS.
   * - ``lowcaps_reif``
     - Rewrite the text in all lower case.
   * - ``text_emojis_reif``
     - Rewrite the text including text emojis like :-) or ;-D.
   * - ``less_common_verbs_reif``
     - Rewrite the text using less common verbs.

The ``finephrase_*`` templates are taken from the
`FinePhrasing tool <https://huggingface.co/spaces/HuggingFaceFW/finephrase>`_
by HuggingFace.


Default selection
^^^^^^^^^^^^^^^^^

When no explicit prompt selection is made, the method cycles through the
templates listed in
:data:`~diversify_text.method.prompting.prompts.DEFAULT_PROMPTS`.

.. code-block:: python

   from diversify_text import diversify

   # Zero-shot: uses the default templates, no style examples needed.
   results = diversify("The cat sat on the mat.", methods=["prompting"])


Selecting specific prompts
^^^^^^^^^^^^^^^^^^^^^^^^^^

Pass a list of keys via ``method_kwargs`` to pick specific templates:

.. code-block:: python

   results = diversify(
       "The cat sat on the mat.",
       methods=["prompting"],
       method_kwargs={"prompting": {"prompt_keys": ["finephrase_faq", "finephrase_tutorial"]}},
   )


Few-shot prompts
----------------

Few-shot templates combine style examples from the
:data:`~diversify_text.styles.DEFAULT_STYLE_BANK` with a prompt that instructs
the model to match the demonstrated writing style.  They use three
placeholders:

- ``[DOCUMENT SEGMENT]`` — the input text
- ``[STYLE EXAMPLES]`` — a formatted list of example sentences from the style bank
- ``[STYLE NAME]`` — the human-readable name of the style

All example-based templates are stored in
:data:`~diversify_text.method.prompting.prompts.EXAMPLE_BASED_PROMPT_BANK`.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Key
     - Description
   * - ``style_transfer``
     - Paraphrase the text to match the writing style demonstrated in the
       examples.  Preserves information while shifting style.
   * - ``humanize_transfer``
     - Humanise machine-generated text using the stylistic patterns from the
       examples.  Inspired by
       `Zhang et al. (2024) <https://arxiv.org/abs/2401.05952>`_.


Automatic few-shot selection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When ``style_example_keys`` is provided without explicit ``prompt_keys``, the
method automatically uses the ``style_transfer`` template:

.. code-block:: python

   results = diversify(
       "The experiment was conducted in a controlled lab setting.",
       methods=["prompting"],
       method_kwargs={
           "prompting": {
               "style_example_keys": ["informal_tinystyler"],
           }
       },
   )

To use a different few-shot template, pass it via ``prompt_keys``:

.. code-block:: python

   results = diversify(
       "The experiment was conducted in a controlled lab setting.",
       methods=["prompting"],
       method_kwargs={
           "prompting": {
               "style_example_keys": ["informal_tinystyler"],
               "prompt_keys": ["humanize_transfer"],
           }
       },
   )


Custom prompt banks
-------------------

You can replace the entire prompt bank with your own templates.  Each template
must contain the ``[DOCUMENT SEGMENT]`` placeholder:

.. code-block:: python

   custom_bank = {
       "simple": "Rewrite the following text in simpler words: [DOCUMENT SEGMENT]",
       "formal": "Rewrite the following text in a formal academic tone: [DOCUMENT SEGMENT]",
   }

   results = diversify(
       "The cat sat on the mat.",
       methods=["prompting"],
       method_kwargs={"prompting": {"prompt_bank": custom_bank}},
   )


Placeholder reference
---------------------

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Placeholder
     - Used in
     - Description
   * - ``[DOCUMENT SEGMENT]``
     - All templates
     - Replaced with the input text to paraphrase.
   * - ``[STYLE EXAMPLES]``
     - Few-shot only
     - Replaced with formatted example sentences from a style set.
   * - ``[STYLE NAME]``
     - Few-shot only
     - Replaced with the human-readable style name (e.g. "informal").
