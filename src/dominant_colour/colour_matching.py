from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from dominant_colour.constants import LOOKUP_TABLE_COLOUR_COLUMNS

LOOKUP_TABLE_PATH = Path(__file__).parent / "data" / "w2c.txt"
BINS_PER_CHANNEL = 32
BIN_WIDTH = 256 // BINS_PER_CHANNEL
FIRST_PROBABILITY_COLUMN = 3
COLUMN_BY_COLOUR = {name: column for column, name in LOOKUP_TABLE_COLOUR_COLUMNS.items()}


def colour_names_to_columns(colour_names) -> list[int]:
    """Which lookup table columns the given colour names sit in."""
    return [COLUMN_BY_COLOUR[name] for name in colour_names]


def load_lookup_table() -> NDArray:
    """Load the trained probabilities as a 32x32x32 grid indexed by [red][green][blue].

    The file is one row per bin with red varying fastest, so it reshapes in
    Fortran order - NumPy's default would put the axes in blue-green-red order.
    """
    table = np.loadtxt(LOOKUP_TABLE_PATH)
    probabilities = table[:, FIRST_PROBABILITY_COLUMN:]
    grid = probabilities.reshape(BINS_PER_CHANNEL, BINS_PER_CHANNEL, BINS_PER_CHANNEL, -1, order="F")

    return np.ascontiguousarray(grid)


LOOKUP_TABLE = load_lookup_table()


def colour_bins(pixels) -> NDArray:
    """Which bin of the 32x32x32 grid each colour falls in, as (..., 3)."""
    # Widening to int first matters: uint8 wraps around when the maths runs on it.
    return np.asarray(pixels, dtype=int) // BIN_WIDTH


def colour_probabilities(pixels) -> NDArray:
    """How likely each colour is to be called each of the 11 basic colours."""
    # Indexing the last axis rather than transposing means this works on a
    # flat list of colours and on a whole image alike.
    bins = colour_bins(pixels)

    return LOOKUP_TABLE[bins[..., 0], bins[..., 1], bins[..., 2]]


def rgb_to_colour_columns(pixels) -> NDArray:
    """Which of the 11 colour columns each RGB value belongs to."""
    return np.argmax(colour_probabilities(pixels), axis=-1)


def rgb_to_colour_names(pixels) -> list[str]:
    """Name each RGB colour with its most likely basic colour."""
    return [LOOKUP_TABLE_COLOUR_COLUMNS[column] for column in rgb_to_colour_columns(pixels)]
