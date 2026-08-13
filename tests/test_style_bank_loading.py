"""Tests for loading and flattening ``stylebank.json``."""

import pytest

from diversify_text.styles.bank import DEFAULT_STYLE_BANK, UNUSUAL_STYLE_BANK
from diversify_text.styles.load import flatten_style_bank, load_style_bank, _RENAMES


class TestLoadStyleBank:
    def test_type_check(self):
        bank = load_style_bank()
        assert len(bank) == 84
        for name, examples in bank.items():
            assert isinstance(name, str) and name
            assert isinstance(examples, list) and examples
            assert all(isinstance(x, str) for x in examples)

    def test_leaf_names_are_renamed(self):
        bank = load_style_bank()
        for item in _RENAMES.items():
            assert item[0] not in bank
            assert item[1] in bank

    def test_flattens(self):
        nested = {
            "top": {
                "branch_a": {"style_1": ["a"], "style_2": ["b"]},
                "branch_b": {"style_3": ["c"]},
            }
        }
        assert flatten_style_bank(nested) == {
            "style_1": ["a"],
            "style_2": ["b"],
            "style_3": ["c"],
        }

    def test_duplicate_leaf_names_raise(self):
        nested = {"a": {"same": ["x"]}, "b": {"same": ["y"]}}
        with pytest.raises(ValueError, match="Duplicate style name 'same'"):
            flatten_style_bank(nested)

    def test_leaf_that_is_not_a_list_raises(self):
        nested = {"a": {"style": "not a list"}}
        with pytest.raises(ValueError, match="expected a taxonomy dict or a list"):
            flatten_style_bank(nested)

    def test_empty_or_non_string_examples_raise(self):
        with pytest.raises(ValueError, match="must be a non-empty list of strings"):
            flatten_style_bank({"a": {"style": []}})
        with pytest.raises(ValueError, match="must be a non-empty list of strings"):
            flatten_style_bank({"a": {"style": ["ok", 3]}})


class TestBankSplit:
    def test_banks_cover_the_json_exactly_and_are_disjoint(self):
        flat = load_style_bank()
        assert set(DEFAULT_STYLE_BANK) | set(UNUSUAL_STYLE_BANK) == set(flat)
        assert not set(DEFAULT_STYLE_BANK) & set(UNUSUAL_STYLE_BANK)

    def test_uncommon_bank_holds_the_unreadable_historical_styles(self):
        assert set(UNUSUAL_STYLE_BANK) == {"old_english", "middle_english"}

    def test_default_n_prefix_is_pinned(self):
        # The first five styles are what every default caller (n=5) gets;
        # changing them is a conscious decision, hence this pin.
        assert list(DEFAULT_STYLE_BANK)[:5] == [
            "informational",
            "digital_communication",
            "barackobama",
            "earlier_african_american_vernacular_english",
            "age_18-24",
        ]
