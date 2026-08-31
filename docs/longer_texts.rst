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

   from diversify_text import diversify

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
           {"style": "informal", "text": "Frank and Isobel Hawking went to Oxford in 1942"},
           {"style": "formal", "text": "I believe Stephen Hawking was born on 8 January 1942."},
           {"style": "question", "text": "Is this true? Both parents went to Oxford and were born on 8 January 1942."},
           {"style": "question_answer_forum", "text": "Isn't it interesting that both parents attended Oxford? Stephen Hawking was born on 8 January 1942."},
           {"style": "discussion_forum", "text": "I think you are right, Stephen Hawking was born on 8 January 1942 to Frank and Isobel Hawking. I think it was a lot of money for them to go to the University of Oxford."},
       ]
   }]

Note how information gets lost in several paraphrases.

.. warning::

   Increasing ``max_new_tokens`` beyond the default may produce unexpected
   results. The used models were not tested for long-form
   generation and may hallucinate, repeat itself, or drift off-topic.

Splitting on punctuation
------------------------

This package also provides the option to split on punctuation.
This splits each input into sentence-level segments, paraphrases each segment
independently (where the model works best), and reassembles the results:

.. code-block:: python

   results = diversify(
       "Stephen Hawking was born on 8 January 1942 to Frank and Isobel Hawking. "
       "Despite their families' financial constraints, both parents attended "
       "the University of Oxford.",
       preprocess_kwargs={"split_on_punctuation": True},
   )

.. code-block:: python

   [{
       "original": "Stephen Hawking was born on 8 January 1942 to Frank and Isobel Hawking. Despite their families' financial constraints, both parents attended the University of Oxford.",
       "paraphrases": [
           {"style": "informal", "text": "i mean Stephen Hawking was born on 8 Jan 1942 to Frank and Isobel Hawking... both parents went to Oxford despite their families financial constraints."},
           {"style": "formal", "text": "Stephen Hawking was born 8 January 1942 to Frank and Isobel Hawking. I heard that both parents went to Oxford despite their families financial constraints."},
           {"style": "question", "text": "Did you know Stephen Hawking was born on 8 January 1942? His parents were Frank and Isobel Hawking. Is that true? Both parents went to Oxford despite their families financial constraints."},
           {"style": "question_answer_forum", "text": "Isn't Stephen Hawking born on 8 January 1942 to Frank and Isobel Hawking? Isn't it interesting that both parents went to Oxford despite their families financial constraints?"},
           {"style": "discussion_forum", "text": "I'm not sure if it is true, but Stephen Hawking was born on 8 January 1942 to Frank and Isobel Hawking. I think it is interesting that both parents went to the University of Oxford despite their families financial constraints."},
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
       preprocess_kwargs={"split_on_punctuation": True},
       max_new_tokens=512,
   )
