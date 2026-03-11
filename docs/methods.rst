Methods
=======

``diversify`` uses a pluggable method architecture. Each method is a
:class:`~diversify.method.base.DiversificationMethod` subclass that generates
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

TinyStyler
----------

`TinyStyler <https://huggingface.co/tinystyler/tinystyler>`_ is a T5-based
model that performs few-shot text style transfer by conditioning on
authorship-embedding representations.

Given a source text and a set of style example sentences, TinyStyler generates
a paraphrase that preserves the content while shifting toward the demonstrated
writing style. ``diversify`` cycles through different style groups from a
configurable *style bank* to produce multiple stylistically diverse outputs.

.. note::

   TinyStyler is based on `CISR <https://huggingface.co/AnnaWegmann/Style-Embedding>`_
   style embeddings, which have been shown to work well for **social-media-like
   settings** and **formality transfer**. The model may not perform as expected
   when reproducing other styles.

**Default style bank.** The built-in bank contains named styles drawn from
the `CORE corpus <https://doi.org/10.1007/s10579-013-9256-1>`_, the
`TinyStyler repository <https://github.com/zacharyhorvitz/TinyStyler>`_ and
the `STEL demo for the formality dimension <https://github.com/nlpsoc/STEL/blob/main/Data/STEL/dimensions/quad_stel-dimension_formal-100_sample.tsv>`_.
See :data:`diversify.method.tinystyler.styles.DEFAULT_STYLE_BANK` for the
full list of available styles.

**Citation:**

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

Adding a new method
-------------------

See :ref:`creating-a-custom-method` in the Usage Guide for instructions on
implementing your own :class:`~diversify.method.base.DiversificationMethod`.
