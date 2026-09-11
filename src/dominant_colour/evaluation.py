from collections import Counter
from pathlib import Path
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from dominant_colour.colour_analysis import DominantColourFinder
from dominant_colour.colour_matching import rgb_to_colour_names

BENCHMARK_DATA_PATH = Path(__file__).parent / "data" / "benchmark_data.txt"

# The order the score columns appear in, which is not the order our own
BENCHMARK_COLOURS = np.array(
    ["red", "orange", "brown", "yellow", "green", "blue", "purple", "pink", "white", "grey", "black"]
)

COLUMN_BY_COLOUR = {colour: column for column, colour in enumerate(BENCHMARK_COLOURS)}

# Out of the 10 points each subject had to give away. A sample whose winning
# colour scored this or more is one people broadly agreed on.
CONFIDENT_SCORE = 7.0

# A colour that scored at least this much was chosen by a fair few people, so
# answering with it isn't unreasonable even if it didn't win.
DEFENSIBLE_SCORE = 2.0


def load_benchmark() -> tuple[NDArray, NDArray]:
    """Load the sample colours and the scores people gave each colour."""
    data = np.loadtxt(BENCHMARK_DATA_PATH)

    return data[:, :3].astype(int), data[:, 3:]


class BenchmarkResults(NamedTuple):
    """Every sample's colour and scores, with our answer and people's answer."""

    colours: NDArray
    scores: NDArray
    predicted: NDArray
    expected: NDArray


def run_benchmark(bucket_size=None) -> BenchmarkResults:
    """Name every benchmark sample and pair our answers with what people chose.

    Quantises the samples first if asked to, to see what that costs.
    """
    colours, scores = load_benchmark()

    if bucket_size is not None:
        finder = DominantColourFinder(colours)
        finder.quantise_colours(bucket_size)
        colours = finder.quantised_pixels

    expected = highest_scoring_colours(scores)
    predicted = np.array(rgb_to_colour_names(colours))

    return BenchmarkResults(colours, scores, predicted, expected)


def highest_scoring_colours(scores) -> NDArray:
    """The colour that scored highest for each sample."""
    return BENCHMARK_COLOURS[np.argmax(scores, axis=1)]


def scores_for_colours(scores, colours) -> NDArray:
    """What each sample's given colour scored with the subjects."""
    columns = np.array([COLUMN_BY_COLOUR[colour] for colour in colours])

    return scores[np.arange(len(scores)), columns]


def agreement_scores(scores) -> NDArray:
    """How much the subjects agreed on each sample, out of 10."""
    return scores.max(axis=1)


def agreed_samples(scores) -> NDArray:
    """Which samples people broadly agreed on."""
    return agreement_scores(scores) >= CONFIDENT_SCORE


def defensible_answers(scores, predicted) -> NDArray:
    """Which answers a fair few people also gave, even where they didn't win."""
    return scores_for_colours(scores, predicted) >= DEFENSIBLE_SCORE


def accuracy_by_colour(predicted, expected) -> dict[str, NDArray]:
    """For each colour people chose, which of those samples we matched."""
    matches = predicted == expected

    return {colour: matches[expected == colour] for colour in BENCHMARK_COLOURS if (expected == colour).any()}


def most_common_mistakes(predicted, expected, mistakes_to_show) -> list[tuple[tuple[str, str], int]]:
    """The wrong answers it gives most, as ((expected, predicted), count) pairs."""
    wrong = predicted != expected

    return Counter(zip(expected[wrong], predicted[wrong])).most_common(mistakes_to_show)


def clearest_failures(scores, predicted, expected, failures_to_show) -> NDArray:
    """Samples people were sure about that we got wrong, least ambiguous first."""
    wrong_but_certain = (predicted != expected) & agreed_samples(scores)
    samples = np.flatnonzero(wrong_but_certain)
    most_certain_first = np.argsort(agreement_scores(scores)[samples])[::-1]

    return samples[most_certain_first][:failures_to_show]
