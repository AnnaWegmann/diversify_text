Styles
======

Every built-in style, by key.  Names are the stable way to select a
style; indices follow bank order, which may change between releases as
the bank is curated.  Each method has its own banks: the default
method (TinyStyler) uses the small curated bank below, the prompting
method uses the default style bank.

TinyStyler bank (default method)
--------------------------------

TinyStyler transfers formality and social-media voice well, but not
dialects, historical English, or most genres, so it uses its own bank
of styles it demonstrably handles (selected by comparing model outputs
for every style; see ``evaluations/tinystyler_style_ratings.txt`` in
the repository).  The example texts are in `tinystylerbank.json
<https://github.com/AnnaWegmann/diversify_text/blob/main/diversify_text/method/tinystyler/tinystylerbank.json>`_,
except the celebrity styles, which come from ``stylebank.json``.
Selectable via ``n``, by name, or by 0-based index.  The listing
below (with two example texts per style) is generated from the code at
every docs build:

.. include:: _generated/tinystyler_bank.inc

By name only — these work but are more likely to produce swearing:

.. include:: _generated/tinystyler_unusual.inc

Of the surface-level styles, TinyStyler supports the following
(by name only):

.. include:: _generated/tinystyler_surface.inc

Every style is defined by a set of example texts.  To print the
current style lists and the examples of one style from Python:

.. code-block:: python

   from diversify_text.method.tinystyler import TinyStylerMethod
   from diversify_text.styles import DEFAULT_STYLE_BANK

   print(list(TinyStylerMethod.style_bank))
   print(list(DEFAULT_STYLE_BANK))
   print(DEFAULT_STYLE_BANK["scottish_english"][:3])

Default style bank
------------------

Used by the prompting method.  The example texts for these styles
(and for the unusual styles below) are in `stylebank.json
<https://github.com/AnnaWegmann/diversify_text/blob/main/diversify_text/styles/stylebank.json>`_.
Selectable via ``n``, by name, or by 0-based index.  The listing
below (with two example texts per style) is generated from the code
at every docs build:

.. include:: _generated/default_bank.inc

Unusual styles
--------------

Too far from contemporary English to be useful defaults.  Selectable by
name only: no index, never picked by ``n``.

.. include:: _generated/default_unusual.inc

Surface-level styles
--------------------

Surface manipulations of the text, defined by example texts like every
other style; the examples are in `surfacebank.json
<https://github.com/AnnaWegmann/diversify_text/blob/main/diversify_text/styles/surfacebank.json>`_.
Also selectable by name only:

.. include:: _generated/surface_bank.inc

Zero-shot bank
--------------

Used by the ``zero_shot`` method.  Each style is defined by one rewrite
*instruction* for the language model instead of example texts.
Selectable via ``n``, by name, or by 0-based index:

.. include:: _generated/zero_shot_bank.inc
