"""The built-in style banks, loaded from ``stylebank.json``.

``stylebank.json`` (in this directory) is the single source of truth
for the style data; :mod:`diversify_text.styles.load` turns its nested
taxonomy into a flat ``name → examples`` dict.  This module only
decides which styles are exposed where, and in which order:

* ``_BANK_ORDER`` lists every style of :data:`DEFAULT_STYLE_BANK` by
  name.  Its order is public API — ``n`` selects the first *n* styles
  and 0-based indices follow it — and is curated so that every
  first-*n* prefix is as diverse as possible: styles rotate round-robin
  through the six taxonomy branches (one diaphasic style, one diamesic,
  one idiolect, one diatopic, one diastratic, one diachronic, then the
  second of each, ...), with each branch's best-stocked styles first.
  The list is hand-editable; the round-robin is how it was generated,
  not a constraint.
* ``_UNCOMMON_ORDER`` holds the styles of :data:`UNCOMMON_STYLE_BANK`:
  kept out of the default bank because they are too far from
  contemporary English to be a useful default rewrite target, but still
  selectable by name (never via ``n`` or an index).  The split is a
  curation choice, not a technical one — moving a name between the two
  lists is all it takes to revisit it.

Together the two lists must contain every leaf of ``stylebank.json``
exactly once; importing this module fails otherwise, so editing the
JSON forces a conscious ordering decision here.
"""

from __future__ import annotations

from diversify_text.styles.load import load_style_bank

_BANK_ORDER: list[str] = [
    "informational",  # diaphasic
    "digital_communication",  # diamesic
    "barackobama",  # idiolect
    "earlier_african_american_vernacular_english",  # diatopic
    "age_18-24",  # diastratic
    "late_modern_english",  # diachronic
    "interactive",  # diaphasic
    "spoken_communication",  # diamesic
    "arianagrande",  # idiolect
    "welsh_english",  # diatopic
    "age_25-34",  # diastratic
    "early_modern_english",  # diachronic
    "narrative",  # diaphasic
    "britneyspears",  # idiolect
    "vernacular_liberian_english",  # diatopic
    "age_35-44",  # diastratic
    "opinion",  # diaphasic
    "cristiano",  # idiolect
    "bahamian_creole",  # diatopic
    "age_45-54",  # diastratic
    "persuasive",  # diaphasic
    "ddlovato",  # idiolect
    "pakistani_english",  # diatopic
    "age_55-74",  # diastratic
    "instructional",  # diaphasic
    "jimmyfallon",  # idiolect
    "tanzanian_english",  # diatopic
    "education_bachelor",  # diastratic
    "lyrical",  # diaphasic
    "jtimberlake",  # idiolect
    "australian_english",  # diatopic
    "education_edu-associatedegree",  # diastratic
    "justinbieber",  # idiolect
    "ghanaian_english",  # diatopic
    "education_highschool",  # diastratic
    "katyperry",  # idiolect
    "newfoundland_english",  # diatopic
    "education_master",  # diastratic
    "ladygaga",  # idiolect
    "liberian_settler_english",  # diatopic
    "education_nodegree",  # diastratic
    "rihanna",  # idiolect
    "scottish_english",  # diatopic
    "education_tradeortechnicalorvocationaltraining",  # diastratic
    "selenagomez",  # idiolect
    "belizean_creole",  # diatopic
    "ethnic_caucasian",  # diastratic
    "shakira",  # idiolect
    "kenyan_english",  # diatopic
    "gender_female",  # diastratic
    "taylorswift13",  # idiolect
    "channel_islands_english",  # diatopic
    "gender_male",  # diastratic
    "irish_english",  # diatopic
    "politics_centrist",  # diastratic
    "jamaican_english",  # diatopic
    "politics_leftwing",  # diastratic
    "hong_kong_english",  # diatopic
    "politics_rightwing",  # diastratic
    "falkland_islands_english",  # diatopic
    "ethnic_african",  # diastratic
    "aboriginal_english",  # diatopic
    "ethnic_eastasian",  # diastratic
    "malaysian_english",  # diatopic
    "ethnic_hispanicorlatino",  # diastratic
    "barbadian_creole",  # diatopic
    "ethnic_other",  # diastratic
    "palmerston_english",  # diatopic
    "education_doctorate",  # diastratic
    "trinidadian_creole",  # diatopic
    "education_some_highschool_no_diploma",  # diastratic
    "indian_english",  # diatopic
    "ethnic_southasian",  # diastratic
    "croker_island_english",  # diatopic
    "ugandan_english",  # diatopic
    "english_dialects_in_the_southwest_of_england",  # diatopic
    "indian_south_african_english",  # diatopic
    "manx_english",  # diatopic
    "white_zimbabwean_english",  # diatopic
    "eastern_maroon_creole",  # diatopic
    "jamaican_creole",  # diatopic
    "vincentian_creole",  # diatopic
]

_UNCOMMON_ORDER: list[str] = [
    "middle_english",  # diachronic
    "old_english",  # diachronic
]

_FLAT = load_style_bank()

_listed = _BANK_ORDER + _UNCOMMON_ORDER
if sorted(_listed) != sorted(_FLAT):
    raise ValueError(
        "The order lists in diversify_text/styles/bank.py are out of sync "
        "with stylebank.json: "
        f"missing {sorted(set(_FLAT) - set(_listed))}, "
        f"unknown or duplicated {sorted(n for n in set(_listed) if n not in _FLAT or _listed.count(n) > 1)}."
    )

#: The default style bank: every style selectable via ``n`` or index.
DEFAULT_STYLE_BANK: dict[str, list[str]] = {
    name: _FLAT[name] for name in _BANK_ORDER
}

#: Styles selectable by name only — see the module docstring.
UNCOMMON_STYLE_BANK: dict[str, list[str]] = {
    name: _FLAT[name] for name in _UNCOMMON_ORDER
}
