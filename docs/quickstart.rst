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

Similarity filter
-----------------

Enable the similarity filter to score each paraphrase with the
`Mutual Implication Score <https://huggingface.co/s-nlp/Mutual_Implication_Score>`_
model and automatically select the best candidate above a minimum score:

.. code-block:: python

   results = diversify(
       "The experiment was conducted in a controlled lab setting.",
       similarity_filter=True,
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

Using the class directly
------------------------

Recommended when processing texts across several calls — the model is loaded
once and reused:

.. code-block:: python

   from diversify_text import Diversifier

   div = Diversifier(device="cuda", methods=["tinystyler"])

   batch_1 = div.diversify(texts_1, n_styles=5)
   batch_2 = div.diversify(texts_2, n_styles=5)

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
