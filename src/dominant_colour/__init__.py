"""Dominant colour finder package."""

from dominant_colour.colour_analysis import DominantColourFinder
from dominant_colour.colour_matching import rgb_to_colour_names
from dominant_colour.image_loading import load_image_as_pixels

__all__ = [
    "DominantColourFinder",
    "load_image_as_pixels",
    "rgb_to_colour_names",
]
