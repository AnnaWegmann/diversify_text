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
           {"style": "informal", "text": "i mean both parents went to Oxford (despite their families financial constraints)."},
           {"style": "formal", "text": "I think both parents went to the University of Oxford."},
           {"style": "question", "text": "Is this true? Both parents went to Oxford despite their families financial constraints."},
           {"style": "question_answer_forum", "text": "Did you know that both parents went to the University of Oxford despite their families' financial constraints?"},
           {"style": "discussion_forum", "text": "I think you are right. I'm just saying that both parents attended Oxford despite their families financial constraints, Stephen Hawking was born on 8 January 1942."},
       ]
   }]

Note how information gets lost in several paraphrases: only
``discussion_forum`` keeps the birth date, and ``formal`` drops both
the date and the financial constraints.

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
           {"style": "informal", "text": "well Stephen Hawking was born on 8 January 1942 to Frank and isobel Hawking... both parents went to Oxford despite their families financial constraints..."},
           {"style": "formal", "text": "I believe Stephen Hawking was born on 8 January 1942. Both parents went to Oxford despite their families' financial constraints."},
           {"style": "question", "text": "Stephen Hawking was born on 8 January 1942. Are you kidding me? Is this true? Both parents went to Oxford despite their families' financial constraints."},
           {"style": "question_answer_forum", "text": "Isn't Stephen Hawking born on 8 January 1942 to Frank and Isobel Hawking? I thought they both went to Oxford because their families were financially challenged?"},
           {"style": "discussion_forum", "text": "Stephen Hawking was born on 8 January 1942 to Frank and Isobel Hawking. I think it was just a coincidence, but I guess it was not that big of a deal I mean they both went to Oxford despite their families financial constraints. I guess it isn't that hard to say, if you have the money."},
       ]
   }]

The core information (birth date and the financial constraints) survives
in every paraphrase here, unlike in the ``max_new_tokens`` approach
above.  The chatty styles pad the seams between segments with filler
(see ``discussion_forum``), and a paraphrase can flip a nuance —
``question_answer_forum`` turns "despite" into "because".

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
