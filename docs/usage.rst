Usage Guide
===========

Control number of styles
------------------------

``n`` selects how many of the default styles are used — one paraphrase
per style. It cannot be combined with ``styles`` or ``style_examples``:

.. code-block:: python

   results = diversify("Some text.", n=3)

.. code-block:: python

   [{"original": "Some text.", "paraphrases": [
       {"style": "informal_tinystyler", "text": "..."},
       {"style": "obama_tinystyler", "text": "..."},
       {"style": "question_tinystyler", "text": "..."},
   ]}]

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
       {"original": "The experiment ...", "paraphrases": [{"style": "...", "text": "..."}, ...]},
       {"original": "She graduated ...", "paraphrases": [{"style": "...", "text": "..."}, ...]},
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

   {"original": "Jane is a ...", "paraphrases": [{"style": "informal_tinystyler", "text": "Jane works as a ..."}]}
   {"original": "John studied ...", "paraphrases": [{"style": "informal_tinystyler", "text": "John was educated ..."}]}

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

Selecting styles
----------------

Every paraphrase is produced by transferring the input text into a target
style, and each target style is defined by a set of example texts. Select
built-in styles from the style bank with ``styles`` (by name and/or
0-based index), or define your own with ``style_examples``; both can be
combined in one call. One paraphrase is generated per style:

.. code-block:: python

   results = diversify(
       "The experiment was conducted in a controlled lab setting.",
       styles=["research_article", "personal_blog", "recipe"],
       style_examples={
           "telegraphic": ["Key finding: effect confirmed. Details follow."],
       },
   )

``DEFAULT_STYLE_BANK`` (``diversify_text.styles``) holds the built-in
styles and their example texts.

.. _creating-a-custom-method:

Creating a custom method
------------------------

.. code-block:: python

   from diversify_text import Diversifier
   from diversify_text.method import DiversificationMethod


   class MyMethod(DiversificationMethod):
       name = "my_method"

       def generate(self, texts, style_dict, *, max_new_tokens=None,
                    temperature=None, top_p=None, **kwargs):
           # style_dict maps each target style name to its example texts.
           return [[f"{text} :: {name}" for name in style_dict] for text in texts]


   results = Diversifier(method=MyMethod()).diversify(
       "Hello", styles=["recipe", "poem"],
   )

.. code-block:: python

   [{"original": "Hello", "paraphrases": [
       {"style": "recipe", "text": "Hello :: recipe"},
       {"style": "poem", "text": "Hello :: poem"},
   ]}]

**Required:** a method must accept the two positional arguments —
``texts`` (the input texts) and ``style_dict`` (style name → example
texts) — and return one generated string per style for each text, in
``style_dict`` order (shape ``len(texts)`` x ``len(style_dict)``).
The style labels are attached by the core afterwards.

**Optional:** ``max_new_tokens``, ``temperature`` and ``top_p`` are
passed by the core; a method that has no use for them can ignore them.
Anything the caller provides via ``method_kwargs`` arrives as extra
keyword arguments, so method-specific options go there.
