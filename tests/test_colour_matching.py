"""Unit tests for colour_matching.py.

These test that the lookup works, not that the names it returns are the
ones a person would pick - how accurate the model is gets measured
separately.
"""

import numpy as np

from dominant_colour import colour_matching
from dominant_colour.colour_matching import (
    BIN_WIDTH,
    BINS_PER_CHANNEL,
    LOOKUP_TABLE,
    LOOKUP_TABLE_PATH,
    colour_bins,
    rgb_to_colour_names,
)
from dominant_colour.constants import LOOKUP_TABLE_COLOUR_COLUMNS


def test_lookup_table_is_a_bin_per_channel_grid() -> None:
    """The table covers all 32x32x32 bins, with the 11 names on the last axis."""
    expected_shape = (BINS_PER_CHANNEL, BINS_PER_CHANNEL, BINS_PER_CHANNEL, len(LOOKUP_TABLE_COLOUR_COLUMNS))

    assert LOOKUP_TABLE.shape == expected_shape


def test_lookup_table_rows_are_probabilities() -> None:
    """Each bin is a probability across the 11 names, so it sums to 1."""
    assert np.allclose(LOOKUP_TABLE.sum(axis=-1), 1.0)


def test_lookup_table_axes_are_red_green_blue() -> None:
    """The grid is indexed [red][green][blue], not the order a default reshape gives.

    Red varies fastest in the file, so the rows one step along each channel have
    to land on the matching axis - getting this wrong names colours silently
    wrongly rather than failing.
    """
    rows = np.loadtxt(LOOKUP_TABLE_PATH)
    first_bin, red_step, green_step, blue_step = 0, 1, BINS_PER_CHANNEL, BINS_PER_CHANNEL**2

    for row, bin_index in [
        (first_bin, (0, 0, 0)),
        (red_step, (1, 0, 0)),
        (green_step, (0, 1, 0)),
        (blue_step, (0, 0, 1)),
    ]:
        expected = rows[row, len(bin_index) :]
        np.testing.assert_array_equal(LOOKUP_TABLE[bin_index], expected)


def test_colour_bins_ends_at_the_last_bin() -> None:
    """White sits in the last bin of every channel, so the range is covered."""
    last_bin = BINS_PER_CHANNEL - 1

    np.testing.assert_array_equal(colour_bins([(255, 255, 255)]), [[last_bin] * 3])


def test_colour_bins_groups_colours_in_the_same_bin() -> None:
    """Colours closer together than the bin width share a bin, so they share a name."""
    same_bin = colour_bins([(0, 0, 0), (BIN_WIDTH - 1, 0, 0)])

    np.testing.assert_array_equal(same_bin[0], same_bin[1])


def test_colour_probabilities_works_on_a_uint8_image() -> None:
    """Naming runs on a whole (H, W, 3) image, in the uint8 that images arrive as."""
    image = np.full((2, 3, 3), 255, dtype=np.uint8)

    probabilities = colour_matching.colour_probabilities(image)

    assert probabilities.shape == (2, 3, len(LOOKUP_TABLE_COLOUR_COLUMNS))


def test_rgb_to_colour_names_picks_the_most_likely_column(monkeypatch) -> None:
    """The name returned is whichever column has the highest probability."""
    fake_table = np.zeros((2, len(LOOKUP_TABLE_COLOUR_COLUMNS)))
    fake_table[0, 4] = 1.0
    fake_table[1, 8] = 1.0
    monkeypatch.setattr(colour_matching, "colour_probabilities", lambda _: fake_table)

    expected = [LOOKUP_TABLE_COLOUR_COLUMNS[4], LOOKUP_TABLE_COLOUR_COLUMNS[8]]
    assert rgb_to_colour_names([(0, 0, 0), (255, 255, 255)]) == expected
