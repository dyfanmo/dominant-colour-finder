import argparse
import sys

from dominant_colour.cli import (
    add_colours_argument,
    add_verbose_argument,
    measured,
    positive_int,
    report_error,
)
from dominant_colour.colour_analysis import DominantColourFinder
from dominant_colour.constants import QUANTISE_BUCKET_SIZE
from dominant_colour.image_loading import load_image_as_pixels


def build_parser() -> argparse.ArgumentParser:
    """Set up the command line arguments the script accepts."""
    parser = argparse.ArgumentParser(description="Find the dominant colour(s) of an image.")
    parser.add_argument("image_path", help="Path to the input image file.")
    parser.add_argument(
        "--top-n",
        type=positive_int,
        default=1,
        help="How many dominant colours to return (default: 1).",
    )
    parser.add_argument(
        "--bucket-size",
        type=positive_int,
        default=QUANTISE_BUCKET_SIZE,
        help=f"Group similar RGB values into buckets this wide (default: {QUANTISE_BUCKET_SIZE}).",
    )
    add_colours_argument(parser, "--ignore", "Colours to exclude.")
    add_verbose_argument(parser)

    return parser


def print_results(results) -> None:
    """Print the colours found, or say why there aren't any."""
    if not results:
        print("No colours left after ignoring the ones you asked to exclude.")
        return

    if len(results) == 1:
        (rgb, name) = results[0]
        print(f"Dominant colour: {name} {rgb}")
        return

    print(f"Top {len(results)} colours:")
    for rank, (rgb, name) in enumerate(results, start=1):
        print(f"  {rank}. {name} {rgb}")


def main(argv=None) -> int:
    """Run the script, returning the exit code."""
    args = build_parser().parse_args(argv)

    try:
        with measured("load", args.verbose):
            pixels = load_image_as_pixels(args.image_path)
    except (FileNotFoundError, ValueError) as exc:
        return report_error(exc)

    with measured("find", args.verbose):
        results = DominantColourFinder(pixels).find(
            top_n=args.top_n,
            bucket_size=args.bucket_size,
            colours_to_ignore=args.ignore,
        )

    print_results(results)

    return 0


if __name__ == "__main__":
    sys.exit(main())
