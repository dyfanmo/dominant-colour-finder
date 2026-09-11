import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from dominant_colour.colour_matching import rgb_to_colour_names

BENCHMARK_DATA_PATH = Path(__file__).parent.parent / "src" / "dominant_colour" / "data" / "benchmark_data.txt"

BENCHMARK_COLOUR_NAMES = [
    "red",
    "orange",
    "brown",
    "yellow",
    "green",
    "blue",
    "purple",
    "pink",
    "white",
    "grey",
    "black",
]

CONFIDENT_SCORE = 7.0
DEFENSIBLE_SCORE = 2.0


def load_benchmark() -> tuple[np.ndarray, np.ndarray]:
    """Load the sample colours and the scores people gave each name."""
    data = np.loadtxt(BENCHMARK_DATA_PATH)

    return data[:, :3].astype(int), data[:, 3:]


def consensus_names(scores) -> list[str]:
    """The name that scored highest for each sample."""
    return [BENCHMARK_COLOUR_NAMES[column] for column in np.argmax(scores, axis=1)]


def score_for(scores, sample, name) -> float:
    """What a given name scored for a given sample, or 0 if we don't use that name."""
    if name not in BENCHMARK_COLOUR_NAMES:
        return 0.0

    return scores[sample, BENCHMARK_COLOUR_NAMES.index(name)]


def print_headline(predicted, expected, scores) -> None:
    """How often we match the name people chose."""
    matches = [p == e for p, e in zip(predicted, expected)]
    top_scores = scores.max(axis=1)

    confident = top_scores >= CONFIDENT_SCORE
    confident_matches = [m for m, c in zip(matches, confident) if c]
    ambiguous_matches = [m for m, c in zip(matches, confident) if not c]

    defensible = [score_for(scores, sample, name) >= DEFENSIBLE_SCORE for sample, name in enumerate(predicted)]

    print(f"samples: {len(predicted)}\n")
    print(f"  matches the name people chose      {share(matches)}")
    print(f"  a name a fair few people chose     {share(defensible)}")
    print()
    print(f"  on samples people agreed on        {share(confident_matches)}")
    print(f"  on samples people disagreed on     {share(ambiguous_matches)}")


def print_per_colour(predicted, expected) -> None:
    """Where the naming holds up and where it falls down, colour by colour."""
    print("\nby colour, out of the samples people gave that name:")
    for colour in BENCHMARK_COLOUR_NAMES:
        matches = [p == e for p, e in zip(predicted, expected) if e == colour]
        if matches:
            print(f"  {colour:8} {share(matches)}")


def print_confusions(predicted, expected) -> None:
    """The mistakes it makes most, which say more than the headline number."""
    mistakes = Counter((e, p) for p, e in zip(predicted, expected) if p != e)

    print("\nmost common mistakes:")
    for (expected_name, predicted_name), count in mistakes.most_common(8):
        print(f"  {expected_name:8} called {predicted_name:8} {count:3}")


def print_worst_samples(colours, predicted, expected, scores) -> None:
    """Wrong answers on samples people were sure about - the real failures."""
    print("\nclearest failures (people were sure, we disagreed):")

    top_scores = scores.max(axis=1)
    wrong = [
        (top_scores[sample], colours[sample], expected[sample], predicted[sample])
        for sample in range(len(predicted))
        if predicted[sample] != expected[sample] and top_scores[sample] >= CONFIDENT_SCORE
    ]

    for score, rgb, expected_name, predicted_name in sorted(wrong, key=lambda row: -row[0])[:8]:
        print(f"  {tuple(rgb)!s:18} {expected_name:8} ({score:.1f}/10) called {predicted_name}")


def share(matches) -> str:
    """Format a list of pass/fail as 'n/total (pp%)'."""
    if not matches:
        return "no samples"

    return f"{sum(matches):3}/{len(matches):3} ({100 * sum(matches) / len(matches):5.1f}%)"


def quantise(colours, bucket_size) -> np.ndarray:
    """Round each channel down to its bucket, as the analysis pipeline does."""
    return (colours // bucket_size) * bucket_size


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure colour naming against human judgements.")
    parser.add_argument(
        "--bucket-size",
        type=int,
        help="Quantise the samples before naming them, to see what that costs.",
    )
    args = parser.parse_args()

    colours, scores = load_benchmark()
    expected = consensus_names(scores)

    if args.bucket_size:
        print(f"quantising at bucket size {args.bucket_size} before naming\n")
        colours = quantise(colours, args.bucket_size)

    predicted = rgb_to_colour_names(colours)

    print_headline(predicted, expected, scores)
    print_per_colour(predicted, expected)
    print_confusions(predicted, expected)
    print_worst_samples(colours, predicted, expected, scores)


if __name__ == "__main__":
    main()
