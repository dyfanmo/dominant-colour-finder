"""Unit tests for cli.py."""

import argparse

import pytest

from dominant_colour.cli import positive_int


def test_positive_int_parses_a_number() -> None:
    """A whole number above zero comes back as an int."""
    assert positive_int("3") == 3


@pytest.mark.parametrize("value", ["0", "-1"])
def test_positive_int_rejects_anything_below_one(value) -> None:
    """Zero and negatives are refused, since neither makes sense here."""
    with pytest.raises(argparse.ArgumentTypeError):
        positive_int(value)


@pytest.mark.parametrize("value", ["abc", "1.5", ""])
def test_positive_int_rejects_anything_that_is_not_a_whole_number(value) -> None:
    """Words and decimals are refused rather than silently truncated."""
    with pytest.raises(argparse.ArgumentTypeError):
        positive_int(value)
