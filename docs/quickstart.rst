Quickstart
==========

Installation
------------

.. code-block:: bash

   pip install diversify-text

How it works
------------

``diversify_text`` is built around a single idea: **a style is defined by a
set of example texts**.  Every call takes your input text plus one or more
style example sets, and rewrites the input in the style that the examples
demonstrate.  Which style transfer method does the rewriting (TinyStyler by
default) is a background detail — the input/output contract is always the
same:

* **Input:** your text(s) and, per target style, a set of example texts.
* **Output:** per input text, one paraphrase per target style, each labeled
  with the style that produced it.

If you don't provide your own styles, the built-in style bank supplies
default ones, so a plain call already produces stylistically diverse
paraphrases.

Basic usage
-----------

.. code-block:: python

   from diversify_text import diversify

   results = diversify("The experiment was conducted in a controlled lab setting.")

.. code-block:: python

   [{
       "original": "The experiment was conducted in a controlled lab setting.",
       "paraphrases": [
           {"style": "informal", "text": "the experiment was in a controlled lab setting so it didnt suck..."},
           {"style": "obama", "text": "Well it was a controlled lab setting that the experiment was conducted in."},
           {"style": "question", "text": "Did you know that the experiment was conducted in a controlled lab setting? It was a re-test."},
           {"style": "formal", "text": "I heard the experiment was conducted in a controlled lab setting."},
           {"style": "song_lyrics", "text": "I mean, this experiment was conducted in a controlled lab setting, so that was a good thing."},
       ]
   }]

Each paraphrase corresponds to one style from the built-in style bank —
by default the first five.  Ask for more distinct styles with ``n``:

.. code-block:: python

   results = diversify("The experiment was conducted in a controlled lab setting.", n=10)

``n`` always means *number of distinct styles*, drawn from the bank in
order.  Requesting more styles than the bank contains raises an error —
you never silently get the same style twice.

Picking styles from the bank
----------------------------

Select specific built-in styles with ``styles``, by name and/or by
(0-based) bank index:

.. code-block:: python

   results = diversify(
       "The experiment was conducted in a controlled lab setting.",
       styles=["recipe", "personal_blog"],
   )

   # indices work too — handy for trying things without knowing the names
   results = diversify(
       "The experiment was conducted in a controlled lab setting.",
       styles=[0, 7, "recipe"],
   )

Unknown names and out-of-range indices raise an error listing what is
available.  Note that indices follow bank order, which may change between
releases as the bank is curated — names are the stable way to pin a style.

Bring your own style examples
-----------------------------

Pass ``style_examples`` to define target styles with your own texts.  A
flat list is one style; a list of lists is several styles; a dict maps
style names to example sets:

.. code-block:: python

   # one style, defined by its example texts
   results = diversify(
       "The experiment was conducted in a controlled lab setting.",
       style_examples=[
           "We found something really interesting — check this out!",
           "You won't believe how well this worked!",
       ],
   )

   # several styles, named
   results = diversify(
       "The experiment was conducted in a controlled lab setting.",
       style_examples={
           "academic": [
               "The results demonstrate a statistically significant effect.",
               "Participants were randomly assigned to one of two conditions.",
           ],
           "enthusiastic": [
               "We found something really interesting — check this out!",
               "You won't believe how well this worked!",
           ],
       },
   )

.. code-block:: python

   [{
       "original": "The experiment was conducted in a controlled lab setting.",
       "paraphrases": [
           {"style": "academic", "text": "The experiment was carried out under controlled laboratory conditions."},
           {"style": "enthusiastic", "text": "Guess what — we ran the whole experiment in a controlled lab, how cool is that!"},
       ]
   }]

``styles`` and ``style_examples`` can be combined in one call (bank styles
come first in the output).  ``n`` cannot be combined with either — the
number of styles is already determined, so passing ``n`` raises an error.

Repeats
-------

``repeats`` controls how many paraphrases are generated *per style*
(default 1).  With more than one repeat, the output interleaves the
styles:

.. code-block:: python

   results = diversify(
       "The experiment was conducted in a controlled lab setting.",
       styles=["recipe", "personal_blog"],
       repeats=2,
   )

.. code-block:: python

   # styles interleave: recipe, personal_blog, recipe, personal_blog
   [{
       "original": "The experiment was conducted in a controlled lab setting.",
       "paraphrases": [
           {"style": "recipe", "text": "..."},
           {"style": "personal_blog", "text": "..."},
           {"style": "recipe", "text": "..."},
           {"style": "personal_blog", "text": "..."},
       ]
   }]

Choosing the style transfer method
----------------------------------

The style examples stay the same regardless of which method rewrites your
text.  The default method is
`TinyStyler <https://huggingface.co/tinystyler/tinystyler>`_, which
conditions on the example texts via authorship embeddings.  Alternatively,
the ``prompting`` method inserts the example texts into a few-shot style
transfer prompt for a causal language model (default:
`SmolLM3-3B <https://huggingface.co/HuggingFaceTB/SmolLM3-3B>`_):

.. code-block:: python

   results = diversify(
       "The experiment was conducted in a controlled lab setting.",
       methods=["prompting"],
       style_examples={
           "academic": [
               "The results demonstrate a statistically significant effect.",
               "Participants were randomly assigned to one of two conditions.",
           ],
       },
   )

Only prompts that take style example texts are supported — every method
receives the same input (your text plus style example sets) and produces
the same output.

Semantic filter
-----------------

Enable the semantic filter to score each paraphrase with the
`Mutual Implication Score <https://huggingface.co/s-nlp/Mutual_Implication_Score>`_
model and automatically select the best candidate above a minimum score.
Candidates are compared per style, so the filter improves semantic
fidelity without reducing stylistic diversity:

.. code-block:: python

   results = diversify(
       "The experiment was conducted in a controlled lab setting.",
       semantic_filter=True,
   )

.. code-block:: python

   [{
       "original": "The experiment was conducted in a controlled lab setting.",
       "paraphrases": [
           {"style": "informal", "text": "the experiment was in a controlled lab setting so it didnt suck..."},
           {"style": "obama", "text": "Well it was a controlled lab setting that the experiment was conducted in."},
           {"style": "question", "text": "Can you explain the experiment? It was conducted in a controlled lab setting."},
           {"style": "formal", "text": "I heard the experiment was conducted in a controlled lab setting."},
           {"style": "song_lyrics", "text": "I mean, this experiment was conducted in a controlled lab setting, so that was a good thing."},
       ]
   }]

Caching
-------

The ``diversify()`` function automatically caches loaded models between calls.
The generation model and the semantic filter are cached independently, so
toggling ``semantic_filter`` does not reload the generation model and vice
versa. Call ``clear_cache()`` to release cached model references when you are done.
On CUDA devices, memory may remain reserved by the underlying framework's caching
allocator and be reused in future calls rather than immediately returned to the OS/driver:

.. code-block:: python

   from diversify_text import clear_cache

   clear_cache()

Using the class directly
------------------------

You can also instantiate a ``Diversifier`` yourself for full control over the
model lifecycle:

.. code-block:: python

   from diversify_text import Diversifier

   div = Diversifier(device="cuda", methods=["tinystyler"])

   batch_1 = div.diversify(texts_1, styles=["recipe", "personal_blog"])
   batch_2 = div.diversify(texts_2, style_examples=my_examples)

Citation
--------

If you use ``diversify`` in your research, we are happy about a citation (placeholder currently).

.. code-block:: bibtex

    @inproceedings{wegmann2026diversify,
        title = {diversify_text: An Amazing Library for Text Diversification},
        author = {Wegmann, Anna and Others},
        url={https://github.com/AnnaWegmann/diversify_text},
        year = {2026},
    }
