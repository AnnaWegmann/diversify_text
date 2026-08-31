Styles
======

Every built-in style, by key.  Names are the stable way to select a
style; indices follow bank order, which may change between releases as
the bank is curated.  Each method can have its own banks: the default
method (TinyStyler) uses the small curated bank below, the prompting
method uses the default style bank.

TinyStyler bank (default method)
--------------------------------

TinyStyler transfers formality and social-media voice well, but not
dialects, historical English, or most genres, so it uses its own bank
of styles it demonstrably handles (selected by comparing model outputs
for every style; see ``evaluations/tinystyler_style_ratings.txt`` in
the repository).

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
Selectable via ``n``, by name, or by 0-based index (the number below):

.. code-block:: text

    0  digital_communication
    1  informational
    2  spoken_communication
    3  lyrical
    4  scottish_english
    5  interactive
    6  britneyspears
    7  late_modern_english
    8  earlier_african_american_vernacular_english
    9  age_55-74
   10  jamaican_creole
   11  barackobama
   12  narrative
   13  instructional
   14  arianagrande
   15  welsh_english
   16  opinion
   17  bahamian_creole
   18  persuasive
   19  cristiano
   20  age_18-24
   21  age_25-34
   22  early_modern_english
   23  vernacular_liberian_english
   24  age_35-44
   25  age_45-54
   26  ddlovato
   27  pakistani_english
   28  jimmyfallon
   29  tanzanian_english
   30  education_bachelor
   31  jtimberlake
   32  australian_english
   33  education_edu-associatedegree
   34  justinbieber
   35  ghanaian_english
   36  education_highschool
   37  katyperry
   38  newfoundland_english
   39  education_master
   40  ladygaga
   41  liberian_settler_english
   42  education_nodegree
   43  rihanna
   44  education_tradeortechnicalorvocationaltraining
   45  selenagomez
   46  belizean_creole
   47  ethnic_caucasian
   48  shakira
   49  kenyan_english
   50  gender_female
   51  taylorswift13
   52  channel_islands_english
   53  gender_male
   54  irish_english
   55  politics_centrist
   56  jamaican_english
   57  politics_leftwing
   58  hong_kong_english
   59  falkland_islands_english
   60  ethnic_african
   61  aboriginal_english
   62  ethnic_eastasian
   63  malaysian_english
   64  ethnic_hispanicorlatino
   65  barbadian_creole
   66  ethnic_other
   67  palmerston_english
   68  education_doctorate
   69  trinidadian_creole
   70  education_some_highschool_no_diploma
   71  indian_english
   72  ethnic_southasian
   73  croker_island_english
   74  ugandan_english
   75  english_dialects_in_the_southwest_of_england
   76  indian_south_african_english
   77  manx_english
   78  white_zimbabwean_english
   79  eastern_maroon_creole
   80  vincentian_creole
   81  politics_rightwing

Unusual styles
--------------

Too far from contemporary English to be useful defaults.  Selectable by
name only: no index, never picked by ``n``.

.. code-block:: text

   middle_english
   old_english

Surface-level styles
--------------------

Surface manipulations of the text, defined by example texts like every
other style; the examples are in `surfacebank.json
<https://github.com/AnnaWegmann/diversify_text/blob/main/diversify_text/styles/surfacebank.json>`_.
Also selectable by name only:

.. code-block:: text

   all_caps
   lowercase
   no_punctuation
   exclamations
   passive_voice
   active_voice
   texting_abbreviations
