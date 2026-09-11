import numpy as np
from PIL import Image

from dominant_colour.colour_matching import rgb_to_colour_names
from dominant_colour.constants import QUANTISE_BUCKET_SIZE

RGBColour = tuple[int, int, int]
MAX_COLOURS = 256**3


class DominantColourFinder:
    def __init__(self, pixels):
        self.pixels = pixels
        self.colours = None
        self.counts = None
        self.quantised_pixels = None

    def quantise_colours(self, bucket_size=QUANTISE_BUCKET_SIZE) -> None:
        """Round each RGB channel down to its bucket, e.g. 234 -> 230 for bucket_size=10."""
        self.quantised_pixels = (self.pixels // bucket_size) * bucket_size

    def count_unique_colours(self) -> None:
        """Count how many pixels share each colour, tallying rather than sorting."""
        counted = Image.fromarray(self.quantised_pixels).getcolors(maxcolors=MAX_COLOURS)

        if counted is None:
            raise ValueError(f"Image has more than {MAX_COLOURS} distinct colours")

        self.counts = np.array([count for count, _colour in counted])
        self.colours = np.array([colour for _count, colour in counted])

    def remove_ignored_colour_names(self, colour_to_ignore) -> None:
        """Drop counted colours belonging to any of the given colour families."""
        colour_names = np.array(rgb_to_colour_names(self.colours))
        colours_to_keep = ~np.isin(colour_names, colour_to_ignore)

        self.colours, self.counts = self.colours[colours_to_keep], self.counts[colours_to_keep]

    def rank_top_colours(self, top_n) -> list[RGBColour]:
        """Return the top_n colours ordered from most to least frequent."""
        descending_order = np.argsort(self.counts)[::-1]
        top_colours = self.colours[descending_order[:top_n]].tolist()

        return [(red, green, blue) for red, green, blue in top_colours]

    def rank_top_colours_with_names(self, top_n) -> list[tuple[RGBColour, str]]:
        """Return the top_n colours as (RGB, colour_name) pairs, most frequent first."""
        top_colours = self.rank_top_colours(top_n)
        names = rgb_to_colour_names(np.array(top_colours))

        return list(zip(top_colours, names))

    def find(self, top_n=1, bucket_size=QUANTISE_BUCKET_SIZE, colours_to_ignore=None) -> list[tuple[RGBColour, str]]:
        """Run the whole pipeline and return the top_n colours as (RGB, name) pairs."""

        self.quantise_colours(bucket_size)
        self.count_unique_colours()
        self.remove_ignored_colour_names(colours_to_ignore or [])
        return self.rank_top_colours_with_names(top_n)
