Longer Texts
============

``diversify`` is designed for **short texts** — single sentences or short
paragraphs.

If you need to diversify longer texts, there are two approaches: increasing
the token limit and splitting on punctuation.

Increasing ``max_new_tokens``
-----------------------------

By default, the number of new tokens is capped automatically based on input
length (up to 256 tokens). You can override this with ``max_new_tokens``:

.. code-block:: python

   from diversify import diversify

   results = diversify(
       "Stephen Hawking was born on 8 January 1942 to Frank and Isobel Hawking. "
       "Despite their families' financial constraints, both parents attended "
       "the University of Oxford.",
       max_new_tokens=512,
   )

.. code-block:: python

   [{
       "original": "Stephen Hawking was born on 8 January 1942 to Frank and Isobel Hawking. Despite their families' financial constraints, both parents attended the University of Oxford.",
       "paraphrases": [
           "both parents went to the university of Oxford, Stephen Hawking was born 8 January 1942...",
           "Well I know that both parents went to Oxford.",
           "How is that? Stephen Hawking was born 8 January 1942 to Frank and Isobel Hawking, who both attended the University of Oxford.",
           "Isobel and Frank Hawking were both at Oxford.",
           "Well I mean, Stephen Hawking was born on 8 January 1942 to Frank and Isobel Hawking who attended Oxford.",
       ]
   }]

Note how information is lost in several paraphrases (e.g. the financial
constraints are dropped entirely).

.. warning::

   Increasing ``max_new_tokens`` beyond the default may produce unexpected
   results. The used models were not tested for long-form
   generation and may hallucinate, repeat itself, or drift off-topic.

Splitting on punctuation
------------------------

This package also provides the option to ``split_on_punctuation=True``.
This splits each input into sentence-level segments, paraphrases each segment
independently (where the model works best), and reassembles the results:

.. code-block:: python

   results = diversify(
       "Stephen Hawking was born on 8 January 1942 to Frank and Isobel Hawking. "
       "Despite their families' financial constraints, both parents attended "
       "the University of Oxford.",
       split_on_punctuation=True,
   )

.. code-block:: python

   [{
       "original": "Stephen Hawking was born on 8 January 1942 to Frank and Isobel Hawking. Despite their families' financial constraints, both parents attended the University of Oxford.",
       "paraphrases": [
           "Stephen Hawking was born 8 January 1942 to Frank and isobel Hawking... both parents went to the university of oxford despite their families financial constraints...",
           "Well, Stephen Hawking was born on 8 January 1942 to Frank and Isobel Hawking. Well, both parents went to the University of Oxford despite their families' financial constraints.",
           "How is Stephen Hawking? He was born on 8 January 1942 to Frank and Isobel Hawking. What? Both parents went to the University of Oxford despite their families financial constraints.",
           "I believe Stephen Hawking was born on 8 January 1942. I have heard both parents went to Oxford despite their families financial constraints.",
           "Well, Stephen Hawking was born on 8 January 1942 to Frank and Isobel Hawking. So both parents went to Oxford despite their families financial constraints? I just want to say, the university is a great place to live.",
       ]
   }]

The paraphrases retain more information compared to the ``max_new_tokens``
approach above.

Combining both
--------------

You can combine both approaches — split on punctuation *and* raise the token
limit for individual segments that may still be long:

.. code-block:: python

   results = diversify(
       "Stephen Hawking was born on 8 January 1942 to Frank and Isobel Hawking. "
       "Despite their families' financial constraints, both parents attended "
       "the University of Oxford.",
       split_on_punctuation=True,
       max_new_tokens=512,
   )
