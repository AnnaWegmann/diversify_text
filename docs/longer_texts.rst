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
           {"style": "digital_communication", "text": "I mean both parents went to Oxford. Stephen Hawking was born on 8 January 1942 despite their families financial constraints. What are you talking about?"},
           {"style": "informational", "text": "The parents of Stephen Hawking and Isobel Hawking went to the University of Oxford despite their financial constraints. They both studied for a degree in physics and a degree in physics. Stephen Hawking was the first person to be a professor at Oxford, and is the first person to have been a professor at Oxford."},
           {"style": "spoken_communication", "text": "Yes, Stephen Hawking was born on 8 January 1942 to Frank and Isobel Hawking. And despite their families' financial constraints, both parents went to the University of Oxford. So you can say that he's a genius."},
           {"style": "lyrical", "text": "Despite their families' financial constraints, both parents went to Oxford and were a part of the 'Oxford University'. Stephen Hawking was born on 8 January 1942. 'Oxford University' was the university where he was born and raised. In the same way that the 'Oxford University' is the university of Oxford."},
           {"style": "scottish_english", "text": "I think both parents went to the University of Oxford and Stephen Hawking was born 8 January 1942."},
       ]
   }]

Note how information gets lost in some paraphrases (``scottish_english``
drops the financial constraints) and invented in others
(``informational`` adds physics degrees and professorships that are not
in the input).

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
           {"style": "digital_communication", "text": "Stephen Hawking was born 8 January 1942 to Frank and Isobel Hawking. That is just so amazing to think about Both parents went to Oxford despite their families financial constraints. I mean if you can afford to go to Oxford then you can go to Oxford. EDIT: oh my god"},
           {"style": "informational", "text": "Stephen Hawking was born on 8 January 1942 to Frank and Isobel Hawking. Stephen Hawking's father, Frank Hawking, is an astronomer and has been a Both parents attended Oxford despite their families' financial constraints. Both parents were very successful in their careers and in their personal lives. Their parents also were able to afford to take the course at Oxford"},
           {"style": "spoken_communication", "text": "Isn't that a bit strange? Stephen Hawking was born on 8 January 1942, to Frank and Isobel Hawking. And yes, they are the parents of the Yeah, both parents went to Oxford despite their families' financial constraints. And I mean, they're both British. So, I'm not saying they didn't go to Oxford,"},
           {"style": "lyrical", "text": "Stephen Hawking was born on 8 January 1942 to Frank and Isobel Hawking. He was born in 1942. He was born in 1942 to Frank and Isobel Hawking. I'm not saying they aren't great people, but their parents both went to Oxford despite their families financial constraints. My dad went to Oxford for his masters and my mom went"},
           {"style": "scottish_english", "text": "Stephen Hawking was born on 8 January 1942 to Frank and Isobel Hawking. Both parents went to Oxford despite their families' financial constraints."},
       ]
   }]

The core information (birth date and the financial constraints) survives
in every paraphrase here, unlike in the ``max_new_tokens`` approach
above.  The seams between segments are visible, though: a segment's
paraphrase can be cut off mid-sentence where the next one is stitched on
(see ``informational`` and ``spoken_communication``).

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
