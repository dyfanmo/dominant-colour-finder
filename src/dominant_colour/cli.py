import argparse

from dominant_colour.constants import COLOUR_NAMES, QUANTISE_BUCKET_SIZE


def positive_int(value) -> int:
    """Parse an argparse value that has to be 1 or more."""
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a whole number, got '{value}'") from None

    if number < 1:
        raise argparse.ArgumentTypeError(f"expected 1 or more, got {number}")

    return number


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
    parser.add_argument(
        "--ignore",
        nargs="*",
        default=[],
        choices=COLOUR_NAMES,
        metavar="COLOUR",
        help=f"Colours to exclude. One or more of: {', '.join(COLOUR_NAMES)}",
    )

    return parser
