from pathlib import Path

import numpy as np

from dominant_colour.colour_matching import colour_names_to_columns, rgb_to_colour_columns

BLACK = (0, 0, 0)
OUTPUT_SUFFIX = "_edited"


def default_output_path(image_path) -> Path:
    """Where to save the edit if the user didn't say."""
    path = Path(image_path)

    return path.with_name(f"{path.stem}{OUTPUT_SUFFIX}.png")


def black_out(pixels, colours_to_remove) -> tuple[np.ndarray, int]:
    """Paint every pixel of the given colours black, and count how many changed."""
    pixel_columns = rgb_to_colour_columns(pixels)
    should_black_out = np.isin(pixel_columns, colour_names_to_columns(colours_to_remove))

    edited = pixels.copy()
    edited[should_black_out] = BLACK

    return edited, int(should_black_out.sum())
