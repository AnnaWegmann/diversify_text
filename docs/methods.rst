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
     - ~1.7B params (default)
     - TBD
     - TBD
     - Prompt-based paraphrasing using a causal LM

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

**Default style bank.** The built-in bank contains named styles drawn from
the `CORE corpus <https://doi.org/10.1007/s10579-013-9256-1>`_, the
`TinyStyler repository <https://github.com/zacharyhorvitz/TinyStyler>`_ and
the `STEL demo for the formality dimension <https://github.com/nlpsoc/STEL/blob/main/Data/STEL/dimensions/quad_stel-dimension_formal-100_sample.tsv>`_.
See :data:`diversify_text.method.tinystyler.styles.DEFAULT_STYLE_BANK` for the
full list of available styles.

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

   results = diversify("The cat sat on the mat.", methods=["prompting"])

**Choosing a model.** Any HuggingFace causal LM can be used. Pass the model
identifier to the constructor:

.. code-block:: python

   from diversify_text import Diversifier
   from diversify_text.method.prompting import PromptingMethod

   method = PromptingMethod(model="mistralai/Mistral-7B-Instruct-v0.3")
   results = Diversifier(methods=[method]).diversify("The cat sat on the mat.")

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

**Default prompt bank.** The built-in bank contains multiple prompt templates
covering different rewriting styles (paraphrasing, simplification, dialogue,
tables, and more). When no explicit selection is made, the templates listed in
:data:`~diversify_text.method.prompting.prompts.DEFAULT_PROMPTS` are used.
See :doc:`prompts` for the full list of available templates.

**Customising the prompt bank.** Like TinyStyler's style bank, you can provide
a custom prompt bank or select specific prompts via ``method_kwargs``. Each
prompt template must contain the placeholder ``[DOCUMENT SEGMENT]``:

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

You can also select specific prompts by key name:

.. code-block:: python

   results = diversify(
       "The cat sat on the mat.",
       methods=["prompting"],
       method_kwargs={"prompting": {"prompt_keys": ["wikipedia_paraphrase"]}},
   )

Zero-shot humanize rewriting
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The prompt bank includes humanize prompts based on
`Zhang et al. (2024) <https://arxiv.org/abs/2401.05952>`_ that rewrite
machine-generated text to appear more human-written. These prompts instruct the
model to introduce informal elements such as typos, slang, hashtags, and
varied casing:

.. code-block:: python

   results = diversify(
       "The experiment was conducted in a controlled lab setting.",
       methods=["prompting"],
       method_kwargs={"prompting": {"prompt_keys": ["humanize_llm-as-coauthor"]}},
   )

A stricter variant, ``humanize_llm-as-coauthor_original``, uses the original
five modifications from the paper and explicitly forbids emojis.

Few-shot style transfer with prompting
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The prompting method can also perform few-shot style transfer by combining
style examples from the shared style bank with a few-shot prompt template.
When ``style_example_keys`` is provided without explicit ``prompt_keys``, the method
automatically uses the ``style_transfer`` template from
:data:`~diversify_text.method.prompting.prompts.EXAMPLE_BASED_PROMPT_BANK`:

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

You can select a different few-shot template via ``prompt_keys``. For
example, ``humanize_transfer`` combines humanization instructions with the
style examples:

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
