"""The parts of the command line handling that both scripts need."""

import argparse
import sys

from dominant_colour.constants import COLOUR_NAMES


def positive_int(value) -> int:
    """Parse an argparse value that has to be 1 or more."""
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a whole number, got '{value}'") from None

    if number < 1:
        raise argparse.ArgumentTypeError(f"expected 1 or more, got {number}")

    return number


def add_colours_argument(parser, name, help_text) -> None:
    """Add an argument that only accepts the names of the eleven colours."""
    parser.add_argument(
        name,
        nargs="*" if name.startswith("-") else "+",
        default=[] if name.startswith("-") else None,
        choices=COLOUR_NAMES,
        metavar="COLOUR",
        help=f"{help_text} One or more of: {', '.join(COLOUR_NAMES)}",
    )


def report_error(exc) -> int:
    """Print a problem to stderr and hand back the exit code to use."""
    print(f"Error: {exc}", file=sys.stderr)

    return 1
