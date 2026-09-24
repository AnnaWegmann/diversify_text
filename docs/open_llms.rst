Open LLMs
=========

The ``prompting`` and ``zero_shot`` methods generate paraphrases with an
open general-purpose language model that runs locally.

Prompting
---------

The ``prompting`` method generates paraphrases by sending input texts to a
local HuggingFace causal language model with a prompt template. The default
model is `SmolLM3-3B <https://huggingface.co/HuggingFaceTB/SmolLM3-3B>`_
using insights from `The Synthetic Data Playbook <https://huggingface.co/spaces/HuggingFaceFW/finephrase>`_.

.. code-block:: python

   results = diversify("The cat sat on the mat.", method="prompting")

**Choosing a model.** Any HuggingFace causal LM can be used. Pass the model
identifier directly to :func:`~diversify_text.diversify`:

.. code-block:: python

   results = diversify(
       "The cat sat on the mat.",
       method="prompting",
       model="Qwen/Qwen3-4B-Instruct-2507",
   )

or to the method constructor:

.. code-block:: python

   from diversify_text import Diversifier
   from diversify_text.method.prompting import PromptingMethod

   method = PromptingMethod(model="mistralai/Mistral-7B-Instruct-v0.3")
   results = Diversifier(method=method).diversify("The cat sat on the mat.")

``model`` works for the ``prompting`` and ``zero_shot`` methods; the
``tinystyler`` method has a fixed model, so passing ``model`` with it
raises an error.

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

**Style examples.** The prompting method uses the default style bank,
loaded from ``stylebank.json``: styles organized in a
language-variation taxonomy — individual styles (idiolects) and
group-level variation across time (diachronic), region (diatopic),
social group (diastratic), register (diaphasic), and medium
(diamesic).  See :data:`diversify_text.styles.DEFAULT_STYLE_BANK` and
the :doc:`styles` page.  Select styles with the top-level ``styles``
parameter (or pass your own via ``style_texts``):

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
(``ZeroShotMethod.style_bank``:
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
-----------

To see the exact prompts sent to the model, enable debug logging:

.. code-block:: python

   import logging
   logging.basicConfig(level=logging.DEBUG)
