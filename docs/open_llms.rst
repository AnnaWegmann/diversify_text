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
raises an error. See :ref:`which-model` below for models we have tried.

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

.. _which-model:

Which model?
------------

These notes are based on manually trying the models on a small set of
sentences and styles — first impressions, not a systematic evaluation.
Which model works best can change with your texts, styles, and
hardware.

Generation speed depends mostly on model size: producing a token
requires reading all model weights from memory once, so on the same
machine, expect speed to scale roughly with one over the parameter
count.

.. list-table::
   :header-rows: 1
   :widths: 30 12 58

   * - Model
     - Parameters
     - Notes
   * - `SmolLM3-3B <https://huggingface.co/HuggingFaceTB/SmolLM3-3B>`_
     - 3B
     - The **default** because fastest. Quality ok-ish. Tends to add info that is not in the input. Sometimes starts replies with "Here\'s a paraphrased version of the document in the same style as the examples provided:"
   * - `Ministral-3-3B-Instruct-2512 <https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512>`_
     - 3B
     - Outputs often contain commentary
       and formatting ("Here's a paraphrase: ..."). Not recommended.
   * - `Qwen3.5-4B <https://huggingface.co/Qwen/Qwen3.5-4B>`_
     - 4B
     - Similar to Qwen3-4B-Instruct-2507. Occasionally repeats an
       emoji many times.
   * - `Qwen3-4B-Instruct-2507 <https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507>`_
     - 4B
     - Best 4B. Mostly faithful to the input and concise.
   * - `gemma-4-E2B-it <https://huggingface.co/google/gemma-4-E2B-it>`_
     - 5B
     - Best 5B. Best of tested **overall**. The most natural-sounding outputs. Some emoji repetitions happening though.


Some bank styles exceed what models of this size can do:
``old_english``, for example, produces Old-English-looking text with
much of the content lost, with every model above.

Example outputs
^^^^^^^^^^^^^^^

Outputs are shown as generated. For dialect styles we can not judge
how authentic the outputs are — the examples only show how differently
the models behave.

``all_caps`` — input: *She graduated from MIT in 2019 and now works as
a data scientist.*

.. list-table::
   :widths: 25 75

   * - SmolLM3-3B
     - THE WOMAN WHO CAME FROM MIT IN 2019 IS NOW A DATA SCIENTIST.
   * - Qwen3-4B-Instruct-2507
     - SHE GRADUATED FROM MIT IN 2019 AND NOW WORKS AS A DATA SCIENTIST.
   * - gemma-4-E2B-it
     - SHE GRADUATED FROM MIT IN TWO THOUSAND NINETEEN AND NOW WORKS AS
       A DATA SCIENTIST.
   * - Qwen3.5-4B
     - SHE GRADUATED FROM MIT IN 2019 AND NOW WORKS AS A DATA SCIENTIST.
   * - Ministral-3-3B
     - I DIDN'T EXPECT SHE WOULD GET THIS FAR IN HER CAREER SO
       QUICKLY—GRADUATING FROM MIT IN 2019 IS A MAJOR ACHIEVEMENT.
       NOW, AS A DATA SCIENTIST, SHE IS

``arianagrande`` (from the bank's individual-styles group) — input:
*I loved the movie, but the ending felt rushed.*

.. list-table::
   :widths: 25 75

   * - SmolLM3-3B
     - OMG, I just watched that flick and it was so good, but the
       ending was like a roller coaster - too fast, too much. I was
       like "WTF, where's the drama?" But the whole movie was worth
       it! [...]
   * - Qwen3-4B-Instruct-2507
     - omg i absolutely loved the movie 💛 but the ending? so fast 🙈
       like i didn't even get to breathe before it was over. 🌌
   * - gemma-4-E2B-it
     - omg the movie was fire 🔥 but the ending kinda rushed me 😩
   * - Qwen3.5-4B
     - i loved the movie but the ending felt so rushed 🌑🌑🌑🌑🌑
   * - Ministral-3-3B
     - ☁️❤️❤️❤️❤️❤️ Ohhh, I *knew* this film was gonna steal my
       heart—until... well, that last act? *Whew.*

``scottish_english`` — input: *Could you send me the report before
Friday?*

.. list-table::
   :widths: 25 75

   * - SmolLM3-3B
     - Ye'll want tae send me that report afore Friday, that's for sure.
   * - Qwen3-4B-Instruct-2507
     - Could ye send me the report before Friday?
   * - gemma-4-E2B-it
     - Could ye send oot the report afore Friday?
   * - Qwen3.5-4B
     - Could you post the report to me before Friday?
   * - Ministral-3-3B
     - The report's due by Friday afore. — Aye, I'll send it to you
       befo' Friday proper.

Development
-----------

To see the exact prompts sent to the model, enable debug logging:

.. code-block:: python

   import logging
   logging.basicConfig(level=logging.DEBUG)
