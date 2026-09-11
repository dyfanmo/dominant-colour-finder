import numpy as np
from numpy.typing import NDArray

from dominant_colour.constants import (
    LOOKUP_TABLE_PATH,
    LOOKUP_TABLE_RGB_COLUMNS,
)

BINS_PER_CHANNEL = 32
BIN_WIDTH = 8


def load_lookup_table() -> NDArray:
    """Load the trained RGB-to-colour-name probabilities."""
    table = np.loadtxt(LOOKUP_TABLE_PATH)
    RGB_COLUMNS_END = 3
    return table[:, RGB_COLUMNS_END:]


LOOKUP_TABLE = load_lookup_table()


def lookup_indices(pixels) -> NDArray:
    """Find each colour's bin in the 32x32x32 grid the table is built on."""
    red, green, blue = np.asarray(pixels, dtype=int).T

    return red // BIN_WIDTH + BINS_PER_CHANNEL * (green // BIN_WIDTH) + BINS_PER_CHANNEL**2 * (blue // BIN_WIDTH)


def colour_probabilities(pixels) -> NDArray:
    """How likely each colour is to be called each of the 11 basic colours."""
    return LOOKUP_TABLE[lookup_indices(pixels)]


def rgb_to_colour_names(pixels) -> list[str]:
    """Name each RGB colour with its most likely basic colour."""
    most_likely = np.argmax(colour_probabilities(pixels), axis=1)

    return [LOOKUP_TABLE_RGB_COLUMNS[column] for column in most_likely]
