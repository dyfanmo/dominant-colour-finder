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
    """Load the trained RGB-to-colour-name probabilities."""
    table = np.loadtxt(LOOKUP_TABLE_PATH)

    return table[:, FIRST_PROBABILITY_COLUMN:]


LOOKUP_TABLE = load_lookup_table()


def lookup_indices(pixels) -> NDArray:
    """Find each colour's bin in the 32x32x32 grid the table is built on."""
    # Indexing the last axis rather than transposing means this works on a
    # flat list of colours and on a whole image alike.
    pixels = np.asarray(pixels, dtype=int)
    red, green, blue = pixels[..., 0], pixels[..., 1], pixels[..., 2]

    return red // BIN_WIDTH + BINS_PER_CHANNEL * (green // BIN_WIDTH) + BINS_PER_CHANNEL**2 * (blue // BIN_WIDTH)


def colour_probabilities(pixels) -> NDArray:
    """How likely each colour is to be called each of the 11 basic colours."""
    return LOOKUP_TABLE[lookup_indices(pixels)]


def rgb_to_colour_columns(pixels) -> NDArray:
    """Which of the 11 colour columns each RGB value belongs to."""
    return np.argmax(colour_probabilities(pixels), axis=-1)


def rgb_to_colour_names(pixels) -> list[str]:
    """Name each RGB colour with its most likely basic colour."""
    return [LOOKUP_TABLE_COLOUR_COLUMNS[column] for column in rgb_to_colour_columns(pixels)]
