"""Unit tests for colour_matching.py.

These test that the lookup works, not that the names it returns are the
ones a person would pick - how accurate the model is gets measured
separately.
"""

import numpy as np

from dominant_colour import colour_matching
from dominant_colour.colour_matching import LOOKUP_TABLE, lookup_indices, rgb_to_colour_names
from dominant_colour.constants import BIN_WIDTH, BINS_PER_CHANNEL, LOOKUP_TABLE_COLUMNS


def test_lookup_table_has_a_row_for_every_bin() -> None:
    """The table covers all 32x32x32 bins, one row each."""
    assert LOOKUP_TABLE.shape == (BINS_PER_CHANNEL**3, len(LOOKUP_TABLE_COLUMNS))


def test_lookup_table_rows_are_probabilities() -> None:
    """Each row is a probability across the 11 names, so it sums to 1."""
    assert np.allclose(LOOKUP_TABLE.sum(axis=1), 1.0)


def test_lookup_indices_ends_at_the_last_bin() -> None:
    """White sits in the last bin, so the arithmetic covers the whole range."""
    assert lookup_indices([(255, 255, 255)]) == [BINS_PER_CHANNEL**3 - 1]


def test_lookup_indices_groups_colours_in_the_same_bin() -> None:
    """Colours closer together than the bin width share an index."""
    same_bin = lookup_indices([(0, 0, 0), (BIN_WIDTH - 1, 0, 0)])

    assert same_bin[0] == same_bin[1]


def test_lookup_indices_gives_each_channel_its_own_step() -> None:
    """Green steps by a row of red bins, blue by a whole layer."""
    steps = lookup_indices([(BIN_WIDTH, 0, 0), (0, BIN_WIDTH, 0), (0, 0, BIN_WIDTH)])

    assert list(steps) == [1, BINS_PER_CHANNEL, BINS_PER_CHANNEL**2]


def test_lookup_indices_handles_uint8_pixels() -> None:
    """Images arrive as uint8, which overflows if it isn't widened first."""
    pixels = np.array([[255, 255, 255]], dtype=np.uint8)

    assert lookup_indices(pixels) == [BINS_PER_CHANNEL**3 - 1]


def test_rgb_to_colour_names_picks_the_most_likely_column(monkeypatch) -> None:
    """The name returned is whichever column has the highest probability."""
    fake_table = np.zeros((2, len(LOOKUP_TABLE_COLUMNS)))
    fake_table[0, 4] = 1.0
    fake_table[1, 8] = 1.0
    monkeypatch.setattr(colour_matching, "LOOKUP_TABLE", fake_table)
    monkeypatch.setattr(colour_matching, "lookup_indices", lambda _: np.array([0, 1]))

    expected = [LOOKUP_TABLE_COLUMNS[4], LOOKUP_TABLE_COLUMNS[8]]
    assert rgb_to_colour_names([(0, 0, 0), (255, 255, 255)]) == expected
