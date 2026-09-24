Methods
=======

``diversify-text`` uses a pluggable method architecture. Each method is a
:class:`~diversify_text.method.base.DiversificationMethod` subclass that generates
paraphrases using a different model or algorithm.

Overview
--------

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 15 35

   * - Method
     - Model Size
     - Speed
     - Performance
     - Description
   * - ``tinystyler``
     - ~800M params
     - TBD
     - TBD
     - Few-shot style transfer using authorship embeddings
   * - ``prompting``
     - ~3B params (default)
     - TBD
     - TBD
     - Prompt-based paraphrasing using a causal LM
   * - ``zero_shot``
     - ~3B params (default)
     - TBD
     - TBD
     - Styles defined by rewrite instructions, via a causal LM

Each method has its own page:

.. toctree::
   :maxdepth: 1

   tinystyler
   open_llms
   adding_a_method
