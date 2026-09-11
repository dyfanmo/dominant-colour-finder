import argparse
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from PIL import Image

from dominant_colour.colour_matching import LOOKUP_TABLE, lookup_indices
from dominant_colour.constants import COLOUR_NAMES, LOOKUP_TABLE_RGB_COLUMNS
from dominant_colour.image_loading import load_image_as_pixels

BLACK = (0, 0, 0)
OUTPUT_SUFFIX = "_edited"


@contextmanager
def timed(label):
    """Time a block of work and print how long it took."""
    start = time.perf_counter()
    yield
    print(f"  {label:22} {(time.perf_counter() - start) * 1000:8.0f}ms")


def name_every_pixel(pixels) -> np.ndarray:
    """Name each pixel in an (H, W, 3) image, keeping the image shape."""
    height, width, _channels = pixels.shape
    flat_pixels = pixels.reshape(-1, 3)

    probabilities = LOOKUP_TABLE[lookup_indices(flat_pixels)]
    columns = np.argmax(probabilities, axis=1)

    return columns.reshape(height, width)


def columns_for(colour_names) -> list[int]:
    """Which lookup table columns the given colour names sit in."""
    column_by_name = {name: column for column, name in LOOKUP_TABLE_RGB_COLUMNS.items()}

    return [column_by_name[name] for name in colour_names]


def black_out(pixels, colours_to_remove) -> np.ndarray:
    """Paint every pixel of the given colours black, leaving the rest alone."""
    pixel_columns = name_every_pixel(pixels)
    should_black_out = np.isin(pixel_columns, columns_for(colours_to_remove))

    edited = pixels.copy()
    edited[should_black_out] = BLACK

    return edited


def default_output_path(image_path) -> Path:
    """Where to save the edit if the user didn't say."""
    path = Path(image_path)

    return path.with_name(f"{path.stem}{OUTPUT_SUFFIX}.png")


def build_parser() -> argparse.ArgumentParser:
    """Set up the command line arguments the script accepts."""
    parser = argparse.ArgumentParser(description="Black out every pixel of a given colour.")
    parser.add_argument("image_path", help="Path to the input image file.")
    parser.add_argument(
        "colours",
        nargs="+",
        choices=COLOUR_NAMES,
        metavar="COLOUR",
        help=f"Colours to black out. One or more of: {', '.join(COLOUR_NAMES)}",
    )
    parser.add_argument(
        "--output",
        help="Where to save the edited image (default: alongside the original).",
    )

    return parser


def main() -> int:
    """Run the script, returning the exit code."""
    args = build_parser().parse_args()
    output_path = Path(args.output) if args.output else default_output_path(args.image_path)

    try:
        with timed("load"):
            pixels = load_image_as_pixels(args.image_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    megapixels = pixels.shape[0] * pixels.shape[1] / 1e6
    print(f"  {'size':22} {pixels.shape[1]}x{pixels.shape[0]} ({megapixels:.1f}MP)")

    with timed("black out"):
        edited = black_out(pixels, args.colours)

    with timed("count changed"):
        changed = int((edited != pixels).any(axis=2).sum())

    with timed("save"):
        Image.fromarray(edited).save(output_path)

    print(f"\nBlacked out {', '.join(args.colours)} - {changed:,} pixels changed")
    print(f"Saved to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
