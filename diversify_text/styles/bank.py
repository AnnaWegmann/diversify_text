"""Building the style bank that is used by the package.

Loads ``stylebank.json`` via  :mod:`diversify_text.styles._load`.
This module decides which styles are exposed in which order.

Together the two lists must contain every leaf of ``stylebank.json``
exactly once; importing this module fails otherwise, so editing the
JSON forces a conscious ordering decision here.
"""

from __future__ import annotations

from diversify_text.styles._load import load_style_bank

# ``n`` selects the first *n* styles -> curate _BANK_ORDER to make
#   first *n* style set is as diverse as possible
#   Manually editable and created
_BANK_ORDER: list[str] = [
    "digital_communication",  # diamesic
    "informational",  # diaphasic
    "spoken_communication",  # diamesic
    "lyrical",  # diaphasic
    "scottish_english",  # diatopic
    "interactive",  # diaphasic
    "britneyspears",  # idiolect
    "late_modern_english",  # diachronic
    "earlier_african_american_vernacular_english",  # diatopic
    "age_55-74",  # diastratic
    "jamaican_creole",  # diatopic
    "barackobama",  # idiolect
    "narrative",  # diaphasic
    "instructional",  # diaphasic
    "arianagrande",  # idiolect
    "welsh_english",  # diatopic
    "opinion",  # diaphasic
    "bahamian_creole",  # diatopic
    "persuasive",  # diaphasic
    "cristiano",  # idiolect
    #
    "age_18-24",  # diastratic
    "age_25-34",  # diastratic
    "early_modern_english",  # diachronic
    "vernacular_liberian_english",  # diatopic
    "age_35-44",  # diastratic
    "age_45-54",  # diastratic
    "ddlovato",  # idiolect
    "pakistani_english",  # diatopic
    "jimmyfallon",  # idiolect
    "tanzanian_english",  # diatopic
    "education_bachelor",  # diastratic
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
    "vincentian_creole",  # diatopic
    # maybe leave out?
    "politics_rightwing",  # diastratic
]

# ``_UNUSUAL_ORDER`` holds styles of :data:`UNUSUAL_STYLE_BANK`:
#   kept out of the default bank because they are too far from
#   contemporary English to be a useful default rewrite target, but still
#   selectable by name (never via ``n`` or an index).
_UNUSUAL_ORDER: list[str] = [
    "middle_english",  # diachronic
    "old_english",  # diachronic
]

_FLAT = load_style_bank()

_listed = _BANK_ORDER + _UNUSUAL_ORDER
if sorted(_listed) != sorted(_FLAT):  # hand-written ordering and selection is not complete/ contains an error
    raise ValueError(
        "stylebank.json and the order lists in diversify_text/styles/bank.py are out of sync "
        f"missing keys: {sorted(set(_FLAT) - set(_listed))}, "
        f"unknown or duplicated {sorted(n for n in set(_listed) if n not in _FLAT or _listed.count(n) > 1)}."
    )

#: The default style bank: every style selectable via ``n`` or index.
DEFAULT_STYLE_BANK: dict[str, list[str]] = {
    name: _FLAT[name] for name in _BANK_ORDER
}

#: Styles selectable by name only. Might delete later.
UNUSUAL_STYLE_BANK: dict[str, list[str]] = {
    name: _FLAT[name] for name in _UNUSUAL_ORDER
}

#: Surface-level rewrites (all caps, passive voice, ...), loaded from
#: ``surfacebank.json``. The file's order is the
#: bank's order.  Selectable by name only, like ``UNUSUAL_STYLE_BANK``.
SURFACE_STYLE_BANK: dict[str, list[str]] = load_style_bank("surfacebank.json")
