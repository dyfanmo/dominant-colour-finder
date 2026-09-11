"""Unit tests for evaluation.py.

These use tiny hand-written score rows rather than the real benchmark file,
so each function's behaviour is visible in the test itself. One test checks
the real file loads with the shape everything else assumes.
"""

import numpy as np

from dominant_colour import evaluation
from dominant_colour.evaluation import (
    BENCHMARK_COLOURS,
    BenchmarkResults,
    accuracy_by_colour,
    agreed_samples,
    agreement_scores,
    clearest_failures,
    defensible_answers,
    highest_scoring_colours,
    load_benchmark,
    most_common_mistakes,
    run_benchmark,
    scores_for_colours,
)


def scores_for(*rows) -> np.ndarray:
    """Build a scores array from dicts of colour name to points, zero elsewhere."""
    scores = np.zeros((len(rows), len(BENCHMARK_COLOURS)))

    for row, points_by_colour in enumerate(rows):
        for colour, points in points_by_colour.items():
            scores[row, list(BENCHMARK_COLOURS).index(colour)] = points

    return scores


def test_load_benchmark_shapes_match() -> None:
    """The real file gives an RGB triple and eleven scores for every sample."""
    colours, scores = load_benchmark()

    assert colours.shape == (len(scores), 3)
    assert scores.shape == (len(colours), len(BENCHMARK_COLOURS))


def test_highest_scoring_colours() -> None:
    """Each sample gets the name that took the most points."""
    scores = scores_for({"red": 7.0, "orange": 3.0}, {"blue": 6.0, "green": 4.0})

    np.testing.assert_array_equal(highest_scoring_colours(scores), ["red", "blue"])


def test_scores_for_colours() -> None:
    """Each sample's given colour is looked up in that sample's own row."""
    scores = scores_for({"red": 7.0, "orange": 3.0}, {"blue": 6.0, "green": 4.0})

    np.testing.assert_array_equal(scores_for_colours(scores, ["orange", "blue"]), [3.0, 6.0])


def test_agreement_scores() -> None:
    """Agreement is the winning colour's score, however the rest were spread."""
    scores = scores_for({"red": 10.0}, {"blue": 5.0, "green": 5.0})

    np.testing.assert_array_equal(agreement_scores(scores), [10.0, 5.0])


def test_agreed_samples_boundary() -> None:
    """A winning score of exactly 7.0 counts as agreed, just below does not."""
    scores = scores_for({"red": 7.0, "orange": 3.0}, {"red": 6.9, "orange": 3.1})

    np.testing.assert_array_equal(agreed_samples(scores), [True, False])


def test_defensible_answers_boundary() -> None:
    """An answer scoring exactly 2.0 is defensible, just below is not."""
    scores = scores_for({"red": 8.0, "orange": 2.0}, {"red": 8.1, "orange": 1.9})

    np.testing.assert_array_equal(defensible_answers(scores, ["orange", "orange"]), [True, False])


def test_accuracy_by_colour() -> None:
    """Samples are grouped by the name people chose, with a pass/fail for each."""
    predicted = np.array(["red", "red", "blue"])
    expected = np.array(["red", "blue", "blue"])

    accuracy = accuracy_by_colour(predicted, expected)

    assert list(accuracy) == ["red", "blue"]
    np.testing.assert_array_equal(accuracy["red"], [True])
    np.testing.assert_array_equal(accuracy["blue"], [False, True])


def test_accuracy_by_colour_skips_absent_colours() -> None:
    """Colours no sample was named with don't appear as empty groups."""
    predicted = np.array(["red"])
    expected = np.array(["red"])

    assert list(accuracy_by_colour(predicted, expected)) == ["red"]


def test_most_common_mistakes() -> None:
    """Mistakes come back as (expected, predicted) pairs, most frequent first."""
    predicted = np.array(["white", "white", "blue", "red"])
    expected = np.array(["grey", "grey", "green", "red"])

    mistakes = most_common_mistakes(predicted, expected, mistakes_to_show=5)

    assert mistakes == [(("grey", "white"), 2), (("green", "blue"), 1)]


def test_most_common_mistakes_respects_the_limit() -> None:
    """Only the requested number of mistakes come back."""
    predicted = np.array(["white", "blue"])
    expected = np.array(["grey", "green"])

    assert len(most_common_mistakes(predicted, expected, mistakes_to_show=1)) == 1


def test_clearest_failures_only_includes_agreed_wrong_answers() -> None:
    """Right answers and ambiguous samples are excluded, whatever we predicted."""
    scores = scores_for({"red": 9.0}, {"blue": 9.0}, {"green": 4.0})
    predicted = np.array(["red", "orange", "orange"])  # right, wrong, wrong-but-ambiguous
    expected = highest_scoring_colours(scores)

    failures = clearest_failures(scores, predicted, expected, failures_to_show=5)

    np.testing.assert_array_equal(failures, [1])


def test_clearest_failures_orders_by_agreement() -> None:
    """The failures people were most certain about come first."""
    scores = scores_for({"red": 8.0}, {"blue": 10.0})
    predicted = np.array(["orange", "orange"])
    expected = highest_scoring_colours(scores)

    failures = clearest_failures(scores, predicted, expected, failures_to_show=5)

    np.testing.assert_array_equal(failures, [1, 0])


def test_run_benchmark_pairs_answers_with_judgements(monkeypatch) -> None:
    """Without quantising, the loaded colours are named as they are."""
    colours = np.array([[255, 0, 0], [0, 0, 255]])
    scores = scores_for({"red": 9.0}, {"blue": 9.0})
    monkeypatch.setattr(evaluation, "load_benchmark", lambda: (colours, scores))
    monkeypatch.setattr(evaluation, "rgb_to_colour_names", lambda _: ["red", "purple"])

    results = run_benchmark()

    assert isinstance(results, BenchmarkResults)
    np.testing.assert_array_equal(results.colours, colours)
    np.testing.assert_array_equal(results.predicted, ["red", "purple"])
    np.testing.assert_array_equal(results.expected, ["red", "blue"])


def test_run_benchmark_quantises_when_asked(monkeypatch) -> None:
    """A bucket size rounds the sample colours down before they are named."""
    colours = np.array([[154, 255, 178]])
    scores = scores_for({"green": 9.0})
    monkeypatch.setattr(evaluation, "load_benchmark", lambda: (colours, scores))
    monkeypatch.setattr(evaluation, "rgb_to_colour_names", lambda quantised: ["green"] * len(quantised))

    results = run_benchmark(bucket_size=10)

    np.testing.assert_array_equal(results.colours, [[150, 250, 170]])
