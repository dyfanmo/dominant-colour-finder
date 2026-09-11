import argparse
import sys
import time
import tracemalloc
from contextlib import contextmanager

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
    optional = name.startswith("-")

    parser.add_argument(
        name,
        nargs="*" if optional else "+",
        default=[] if optional else None,
        choices=COLOUR_NAMES,
        metavar="COLOUR",
        help=f"{help_text} One or more of: {', '.join(COLOUR_NAMES)}",
    )


def add_verbose_argument(parser) -> None:
    """Add the flag that turns on timing and memory reporting for each step."""
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print how long each step took and how much memory it used.",
    )


@contextmanager
def measured(label, verbose=False):
    """Time a block of work and report its peak memory, if asked to."""
    if not verbose:
        yield
        return

    tracemalloc.start()
    start = time.perf_counter()

    yield

    elapsed = time.perf_counter() - start
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"  {label:22} {elapsed * 1000:8.0f}ms {peak / 1e6:8.1f}MB peak")


def report_error(exc) -> int:
    """Print a problem to stderr and hand back the exit code to use."""
    print(f"Error: {exc}", file=sys.stderr)

    return 1
