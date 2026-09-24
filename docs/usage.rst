Usage Guide
===========

Control number of styles
------------------------

``n`` selects how many of the default styles are used — one paraphrase
per style. It cannot be combined with ``styles`` or ``style_texts``:

.. code-block:: python

   results = diversify("Some text.", n=3)

.. code-block:: python

   [{"original": "Some text.", "paraphrases": [
       {"style": "informal", "text": "..."},
       {"style": "formal", "text": "..."},
       {"style": "question", "text": "..."},
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

   {"original": "Jane is a ...", "paraphrases": [{"style": "...", "text": "..."}]}
   {"original": "John studied ...", "paraphrases": [{"style": "...", "text": "..."}]}

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
0-based index), or define your own with ``style_texts``; both can be
combined in one call. One paraphrase is generated per style:

.. code-block:: python

   results = diversify(
       "The experiment was conducted in a controlled lab setting.",
       styles=["informal", "personal_blog", "obama"],
       style_texts={
           "telegraphic": ["Key finding: effect confirmed. Details follow."],
       },
   )

Each method can have its own style bank, so which names are available
depends on the active method. The default method (TinyStyler) uses a
small bank of styles it demonstrably handles; the prompting method
uses ``DEFAULT_STYLE_BANK`` (``diversify_text.styles``), the large
bank of dialects, registers, and more. The full lists are on the
:doc:`styles` page.

Creating a custom method
------------------------

See :ref:`creating-a-custom-method` on the Methods pages for
instructions on implementing your own
:class:`~diversify_text.method.base.DiversificationMethod`.
