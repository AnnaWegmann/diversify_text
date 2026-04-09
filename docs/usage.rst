Usage Guide
===========

Control number of paraphrases
-----------------------------

.. code-block:: python

   results = diversify("Some text.", n=3)

.. code-block:: python

   [{"original": "Some text.", "paraphrases": ["...", "...", "..."]}]

Reproducibility (seed)
----------------------

``diversify`` sets a default random seed (``51173``) to make runs more
reproducible.  The seed is applied to Python's ``random``, PyTorch
(CPU and CUDA), and NumPy.  It is logged at the start of each run, but
exact determinism is **not** guaranteed across different hardware, library
versions, or backends.

To get a different set of paraphrases, pass a different seed:

.. code-block:: python

   results = diversify("Some text.", seed=123)

To disable seeding entirely (non-deterministic output):

.. code-block:: python

   results = diversify("Some text.", seed=None)

List of texts
-------------

.. code-block:: python

   results = diversify([
       "The experiment was conducted in a controlled lab setting.",
       "She graduated from MIT in 2019.",
   ])

.. code-block:: python

   [
       {"original": "The experiment ...", "paraphrases": ["...", "...", ...]},
       {"original": "She graduated ...", "paraphrases": ["...", "...", ...]},
   ]

CSV / TSV file
--------------

Reads the file and writes a JSONL file next to the input
(``<input>_diversified.jsonl``).

.. code-block:: python

   results = diversify("bios.csv", text_column="bio")
   # writes bios_diversified.jsonl

Each line in the JSONL output is one JSON object:

.. code-block:: json

   {"original": "Jane is a ...", "paraphrases": ["Jane works as a ...", "As a ..., Jane ..."]}
   {"original": "John studied ...", "paraphrases": ["John was educated ...", "..."]}

TXT file
--------

Each non-empty line is treated as a separate text to diversify. Output is
written to ``<input>.jsonl``.

.. code-block:: python

   results = diversify("texts.txt")
   # writes texts.jsonl

Controlling output location
----------------------------

By default, file inputs write output next to the input file and in-memory
inputs (strings, lists) return a Python list. You can override this with
``output_dir`` and ``output_name``:

.. code-block:: python

   # Write output to a specific directory
   results = diversify("bios.csv", text_column="bio", output_dir="/results")
   # writes /results/bios_diversified.jsonl

   # Also set a custom filename
   results = diversify("bios.csv", text_column="bio", output_dir="/results", output_name="my_output")
   # writes /results/my_output.jsonl

   # Force a list input to write to disk instead of returning in-memory
   results = diversify(["text one", "text two"], output_dir=".")
   # writes ./diversified_output.jsonl

The ``.jsonl`` extension is always added automatically.

Longer texts
-------------

For tips on handling longer texts (punctuation splitting, increasing
``max_new_tokens``), see :doc:`longer_texts`.

Multiple methods
----------------

You can combine methods to get diverse paraphrases from different approaches.
The requested ``n`` are distributed across the methods:

.. code-block:: python

   results = diversify("The cat sat on the mat.", methods=["tinystyler", "prompting"], n=4)

Customising the TinyStyler style bank
--------------------------------------

TinyStyler generates each paraphrase by conditioning on a *style example* — a
short sentence that demonstrates the target writing style. The style bank is
the list of such examples that get cycled through when producing multiple
paraphrases.

The default bank is a dictionary mapping style labels to lists of example
sentences (drawn from the CORE corpus). You can replace or extend it by
passing a custom bank via ``method_kwargs``.

A style bank can be a ``dict[str, list[str]]`` or a ``list[list[str]]``:

.. code-block:: python

   from diversify_text import diversify
   from diversify_text.styles import DEFAULT_STYLE_BANK

   custom_bank = {
       "academic": ["The results demonstrate a statistically significant effect."],
       "enthusiastic": ["We found something really interesting — check this out!"],
       "telegraphic": ["Key finding: effect confirmed. Details follow."],
   }

   results = diversify(
       "The experiment was conducted in a controlled lab setting.",
       method_kwargs={"tinystyler": {"style_bank": custom_bank}},
   )

``DEFAULT_STYLE_BANK`` is exported from ``diversify_text.styles`` so you
can build on it:

.. code-block:: python

   from diversify_text.styles import DEFAULT_STYLE_BANK

   extended_bank = {
       **DEFAULT_STYLE_BANK,
       "scientific": ["The data clearly indicate a statistically significant result."],
   }

You can also select specific styles by key name with ``styles``, instead of
cycling through the entire bank. The number of paraphrases is determined by
the number of selected styles:

.. code-block:: python

   results = diversify(
       "The experiment was conducted in a controlled lab setting.",
       method_kwargs={"tinystyler": {"styles": ["research_article", "personal_blog", "recipe"]}},
   )

.. _creating-a-custom-method:

Creating a custom method
------------------------

.. code-block:: python

   from diversify_text import Diversifier
   from diversify_text.method import DiversificationMethod


   class MyMethod(DiversificationMethod):
       name = "my_method"

       def generate(self, texts, *, n, max_new_tokens, temperature, top_p, **kwargs):
           return [[f"{text} :: variant {i}" for i in range(n)] for text in texts]


   results = Diversifier(methods=[MyMethod()]).diversify("Hello", n=3)

.. code-block:: python

   [{"original": "Hello", "paraphrases": ["Hello :: variant 0", "Hello :: variant 1", "Hello :: variant 2"]}]
