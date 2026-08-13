"""Tests for loading and flattening ``stylebank.json``."""

import pytest

from diversify_text.styles.load import flatten_style_bank, load_style_bank


class TestLoadStyleBank:
    def test_loads_all_styles_as_example_lists(self):
        bank = load_style_bank()
        assert len(bank) == 84
        for name, examples in bank.items():
            assert isinstance(name, str) and name
            assert isinstance(examples, list) and examples
            assert all(isinstance(x, str) for x in examples)

    def test_awkward_leaf_names_are_cleaned(self):
        bank = load_style_bank()
        assert "barbadian_creole" in bank
        assert "barbadian_creole_(bajan)" not in bank
        assert "education_some_highschool_no_diploma" in bank
        assert "education_somehighschool,nodiploma" not in bank


class TestFlattenStyleBank:
    def test_flattens_to_leaf_names_in_traversal_order(self):
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
        with pytest.raises(ValueError, match="non-empty list"):
            flatten_style_bank({"a": {"style": []}})
        with pytest.raises(ValueError, match="non-empty list"):
            flatten_style_bank({"a": {"style": ["ok", 3]}})
