Quickstart
==========

Installation
------------

.. code-block:: bash

   pip install diversify-text

Basic usage
-----------

.. code-block:: python

   from diversify_text import diversify

   results = diversify("The experiment was conducted in a controlled lab setting.")

.. code-block:: python

   [{
       "original": "The experiment was conducted in a controlled lab setting.",
       "paraphrases": [
           "the experiment was in a controlled lab setting so it didnt suck...",
           "Well it was a controlled lab setting that the experiment was conducted in.",
           "Did you know that the experiment was conducted in a controlled lab setting? It was a re-test.",
           "I heard the experiment was conducted in a controlled lab setting.",
           "I mean, this experiment was conducted in a controlled lab setting, so that was a good thing.",
       ]
   }]

Semantic filter
-----------------

Enable the semantic filter to score each paraphrase with the
`Mutual Implication Score <https://huggingface.co/s-nlp/Mutual_Implication_Score>`_
model and automatically select the best candidate above a minimum score:

.. code-block:: python

   results = diversify(
       "The experiment was conducted in a controlled lab setting.",
       semantic_filter=True,
   )

.. code-block:: python

   [{
       "original": "The experiment was conducted in a controlled lab setting.",
       "paraphrases": [
           "the experiment was in a controlled lab setting so it didnt suck...",
           "Well it was a controlled lab setting that the experiment was conducted in.",
           "Can you explain the experiment? It was conducted in a controlled lab setting.",
           "I heard the experiment was conducted in a controlled lab setting.",
           "I mean, this experiment was conducted in a controlled lab setting, so that was a good thing.",
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

   batch_1 = div.diversify(texts_1, n=5)
   batch_2 = div.diversify(texts_2, n=5)

Citation
--------

.. note::

   TinyStyler is currently the only built-in generation method.
   See :doc:`methods` for details.

If you use ``diversify`` in your research, please cite TinyStyler:

.. code-block:: bibtex

   @inproceedings{horvitz-etal-2024-tinystyler,
       title = "{T}iny{S}tyler: Efficient Few-Shot Text Style Transfer with Authorship Embeddings",
       author = "Horvitz, Zachary  and
         Patel, Ajay  and
         Singh, Kanishk  and
         Callison-Burch, Chris  and
         McKeown, Kathleen  and
         Yu, Zhou",
       editor = "Al-Onaizan, Yaser  and
         Bansal, Mohit  and
         Chen, Yun-Nung",
       booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2024",
       month = nov,
       year = "2024",
       address = "Miami, Florida, USA",
       publisher = "Association for Computational Linguistics",
       url = "https://aclanthology.org/2024.findings-emnlp.781",
       pages = "13376--13390",
   }
