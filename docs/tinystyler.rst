TinyStyler
==========

`TinyStyler <https://huggingface.co/tinystyler/tinystyler>`_ is a T5-based
model that performs few-shot text style transfer by conditioning on
authorship-embedding representations.

Given a source text and a set of style example sentences, TinyStyler generates
a paraphrase that preserves the content while shifting toward the demonstrated
writing style. ``diversify-text`` cycles through different style groups from a
configurable *style bank* to produce multiple stylistically diverse outputs.

.. note::

   TinyStyler is based on `CISR <https://huggingface.co/AnnaWegmann/Style-Embedding>`_
   style embeddings, which have been shown to work well for **social-media-like
   settings** and **formality transfer**. The model may not perform as expected
   when reproducing other styles.

**Style bank.** TinyStyler does not use the package's default style
bank: it has its own small bank of styles it demonstrably handles
(``informal``, ``formal``, ``question``, ...), selected by running the
model on every available style and comparing outputs
(``evaluations/tinystyler_style_ratings.txt`` in the repository).
A few additional styles work but are more likely to produce swearing;
they are selectable by name only.  The :doc:`styles` page lists all of
them, along with the default bank used by the prompting method.

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
