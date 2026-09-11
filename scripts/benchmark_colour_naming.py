import argparse
import sys

from dominant_colour.cli import add_verbose_argument, measured, positive_int
from dominant_colour.evaluation import (
    accuracy_by_colour,
    agreed_samples,
    agreement_scores,
    clearest_failures,
    defensible_answers,
    most_common_mistakes,
    run_benchmark,
)

MISTAKES_TO_SHOW = 8
FAILURES_TO_SHOW = 8


def format_accuracy(matches) -> str:
    """Format a boolean array of pass/fail as 'n/total (pp%)'."""
    if matches.size == 0:
        return "no samples"

    return f"{matches.sum():3}/{matches.size:3} ({100 * matches.mean():5.1f}%)"


def print_headline(scores, predicted, expected) -> None:
    """How often we match the name people chose."""
    matches = predicted == expected
    agreed = agreed_samples(scores)

    print(f"samples: {matches.size}\n")
    print(f"  matches the name people chose      {format_accuracy(matches)}")
    print(f"  a name a fair few people chose     {format_accuracy(defensible_answers(scores, predicted))}")
    print()
    print(f"  on samples people agreed on        {format_accuracy(matches[agreed])}")
    print(f"  on samples people disagreed on     {format_accuracy(matches[~agreed])}")


def print_per_colour(predicted, expected) -> None:
    """Where the naming holds up and where it falls down, colour by colour."""
    print("\nby colour, out of the samples people gave that name:")
    for colour, matches in accuracy_by_colour(predicted, expected).items():
        print(f"  {colour:8} {format_accuracy(matches)}")


def print_mistakes(predicted, expected) -> None:
    """The mistakes it makes most, which say more than the headline number."""
    print("\nmost common mistakes:")
    for (expected_name, predicted_name), count in most_common_mistakes(predicted, expected, MISTAKES_TO_SHOW):
        print(f"  {expected_name:8} called {predicted_name:8} {count:3}")


def print_failures(colours, scores, predicted, expected) -> None:
    """Wrong answers on samples people were sure about - the real failures."""
    print("\nclearest failures (people were sure, we disagreed):")

    agreement = agreement_scores(scores)
    for sample in clearest_failures(scores, predicted, expected, FAILURES_TO_SHOW):
        rgb = tuple(colours[sample].tolist())
        print(f"  {rgb!s:18} {expected[sample]:8} ({agreement[sample]:.1f}/10) called {predicted[sample]}")


def build_parser() -> argparse.ArgumentParser:
    """Set up the command line arguments the script accepts."""
    parser = argparse.ArgumentParser(description="Measure colour naming against human judgements.")
    parser.add_argument(
        "--bucket-size",
        type=positive_int,
        help="Quantise the samples before naming them, to see what that costs.",
    )
    add_verbose_argument(parser)

    return parser


def main(argv=None) -> int:
    """Run the script, returning the exit code."""
    args = build_parser().parse_args(argv)

    if args.bucket_size is not None:
        print(f"quantising at bucket size {args.bucket_size} before naming\n")

    with measured("benchmark", args.verbose):
        colours, scores, predicted, expected = run_benchmark(args.bucket_size)

    print_headline(scores, predicted, expected)
    print_per_colour(predicted, expected)
    print_mistakes(predicted, expected)
    print_failures(colours, scores, predicted, expected)

    return 0


if __name__ == "__main__":
    sys.exit(main())
