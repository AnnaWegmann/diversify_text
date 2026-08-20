Methods
=======

``diversify-text`` uses a pluggable method architecture. Each method is a
:class:`~diversify_text.method.base.DiversificationMethod` subclass that generates
paraphrases using a different model or algorithm.

Overview
--------

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 15 35

   * - Method
     - Model Size
     - Speed
     - Performance
     - Description
   * - ``tinystyler``
     - ~800M params
     - TBD
     - TBD
     - Few-shot style transfer using authorship embeddings
   * - ``prompting``
     - ~3B params (default)
     - TBD
     - TBD
     - Prompt-based paraphrasing using a causal LM
   * - ``zero_shot``
     - ~3B params (default)
     - TBD
     - TBD
     - Styles defined by rewrite instructions, via a causal LM

TinyStyler
----------

`TinyStyler <https://huggingface.co/tinystyler/tinystyler>`_ is a T5-based
model that performs few-shot text style transfer by conditioning on
authorship-embedding representations.

Given a source text and a set of style example sentences, TinyStyler generates
a paraphrase that preserves the content while shifting toward the demonstrated
writing style. ``diversify-text`` cycles through different style groups from a
configurable *style bank* to produce multiple stylistically diverse outputs.

.. note::

   TinyStyler is based on `CISR <https://huggingface.co/AnnaWegmann/Style-Embedding>`_
   style embeddings, which have been shown to work well for **social-media-like
   settings** and **formality transfer**. The model may not perform as expected
   when reproducing other styles.

**Default style bank.** The built-in styles are loaded from
``stylebank.json``, which organizes them in a language-variation
taxonomy: individual styles (idiolects) and group-level variation
across time (diachronic), region (diatopic), social group
(diastratic), register (diaphasic), and medium (diamesic).
The :doc:`styles` page lists every available style by key, including
the name-only styles in
:data:`diversify_text.styles.UNUSUAL_STYLE_BANK` and
:data:`diversify_text.styles.SURFACE_STYLE_BANK`.

**Citation:**

.. code-block:: bibtex

   @inproceedings{horvitz-etal-2024-tinystyler,
       title = "{T}iny{S}tyler: Efficient Few-Shot Text Style Transfer with Authorship Embeddings",
       author = "Horvitz, Zachary  and
         Patel, Ajay  and
         Singh, Kanishk  and
         Callison-Burch, Chris  and
         McKeown, Kathleen  and
         Yu, Zhou",
       editor = "Al-Onaizan, Yaser  and
         Bansal, Mohit  and
         Chen, Yun-Nung",
       booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2024",
       month = nov,
       year = "2024",
       address = "Miami, Florida, USA",
       publisher = "Association for Computational Linguistics",
       url = "https://aclanthology.org/2024.findings-emnlp.781",
       pages = "13376--13390",
   }

Prompting
---------

The ``prompting`` method generates paraphrases by sending input texts to a
local HuggingFace causal language model with a prompt template. The default
model is `SmolLM3-3B <https://huggingface.co/HuggingFaceTB/SmolLM3-3B>`_
using insights from `The Synthetic Data Playbook <https://huggingface.co/spaces/HuggingFaceFW/finephrase>`_.

.. code-block:: python

   results = diversify("The cat sat on the mat.", method="prompting")

**Choosing a model.** Any HuggingFace causal LM can be used. Pass the model
identifier to the constructor:

.. code-block:: python

   from diversify_text import Diversifier
   from diversify_text.method.prompting import PromptingMethod

   method = PromptingMethod(model="mistralai/Mistral-7B-Instruct-v0.3")
   results = Diversifier(method=method).diversify("The cat sat on the mat.")

Instruct-tuned models are recommended. Chat templates are applied automatically
when the tokenizer provides one.

.. note::

   Thinking/reasoning models (e.g. SmolLM3-3B) are detected automatically and
   have their thinking mode turned off (``enable_thinking=False``) during
   generation. Thinking tokens add overhead without improving paraphrase
   quality in this setting.

**Inference backend.** The method currently uses the ``transformers`` library
for inference.

.. note::

   `vLLM <https://vllm.ai/>`_ support, batched inference, and streaming from
   large files are planned for a future release.

**Prompt templates.** All templates are example-based style transfer prompts:
the target style is demonstrated through example texts inserted into the
prompt; prompts without style examples are intentionally not supported. The
default template is ``style_transfer``; ``humanize_transfer`` (inspired by
`Zhang et al. (2024) <https://arxiv.org/abs/2401.05952>`_) additionally
instructs the model to imitate human imperfections found in the style
examples. Select a template — or pass your own — via the ``prompt`` option:

.. code-block:: python

   results = diversify(
       "The experiment was conducted in a controlled lab setting.",
       method="prompting",
       method_kwargs={"prompt": "humanize_transfer"},
   )

A custom template must contain both the ``[DOCUMENT SEGMENT]`` and
``[STYLE EXAMPLES]`` placeholders (``[STYLE NAME]`` is optional):

.. code-block:: python

   my_prompt = (
       "Study these examples:\n[STYLE EXAMPLES]\n"
       "Rewrite the following text in the same style. "
       "Text: [DOCUMENT SEGMENT]"
   )

   results = diversify(
       "The cat sat on the mat.",
       method="prompting",
       method_kwargs={"prompt": my_prompt},
   )

**Style examples.** Styles come from the shared style bank; select them with
the top-level ``styles`` parameter (or pass your own via ``style_texts``):

.. code-block:: python

   results = diversify(
       "The experiment was conducted in a controlled lab setting.",
       method="prompting",
       styles=["informational"],
   )

Zero-shot
---------

The ``zero_shot`` method defines each style by a rewrite *instruction*
instead of example texts, and sends one instruction per style to a
causal language model (same default model and options — ``model``,
``precision`` — as the prompting method).

Its own style bank maps style names to instructions
(:data:`diversify_text.method.zero_shot.ZERO_SHOT_STYLE_BANK`:
``formal``, ``simple``, ``complex``, ``caps``, ``lowercase``, and
more), so ``styles`` and ``n`` select from these:

.. code-block:: python

   results = diversify(
       "The experiment was conducted in a controlled lab setting.",
       method="zero_shot",
       styles=["formal", "caps"],
   )

With this method, ``style_texts`` are instructions — exactly one per
style. An instruction can place the input text itself with
``[DOCUMENT SEGMENT]``; otherwise the text is appended at the end:

.. code-block:: python

   results = diversify(
       "The experiment was conducted in a controlled lab setting.",
       method="zero_shot",
       style_texts={"pirate": ["Rewrite the text as an old-timey pirate would say it."]},
   )

Development
^^^^^^^^^^^

To see the exact prompts sent to the model, enable debug logging:

.. code-block:: python

   import logging
   logging.basicConfig(level=logging.DEBUG)

Adding a new method
-------------------

See :ref:`creating-a-custom-method` in the Usage Guide for instructions on
implementing your own :class:`~diversify_text.method.base.DiversificationMethod`.
