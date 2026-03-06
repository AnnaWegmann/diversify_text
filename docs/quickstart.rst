Quickstart
==========

Installation
------------

.. note::

   You must have **uv** installed before running ``uv sync``.
   Full installation guide: https://docs.astral.sh/uv/getting-started/installation/

After installing ``uv`` on your system, clone the repo and install:

.. code-block:: bash

   git clone https://github.com/AnnaWegmann/diversify.git
   cd diversify
   uv sync
   source .venv/bin/activate

Basic usage
-----------

.. code-block:: python

   from diversify import diversify

   results = diversify("The experiment was conducted in a controlled lab setting.")

.. code-block:: python

   [{
       "original": "The experiment was conducted in a controlled lab setting.",
       "paraphrases": [
           "They ran the experiment in a controlled lab setting.",
           "The experiment took place in a controlled lab.",
           "A controlled lab was where the experiment was conducted.",
           "In a controlled lab, the experiment was carried out.",
           "The study was performed in a controlled lab environment.",
       ]
   }]

Using the class directly
------------------------

Recommended when processing texts across several calls — the model is loaded
once and reused:

.. code-block:: python

   from diversify import Diversifier

   div = Diversifier(device="cuda", methods=["tinystyler"])

   batch_1 = div.diversify(texts_1, n_styles=5)
   batch_2 = div.diversify(texts_2, n_styles=5)
