import numpy as np

from dominant_colour import colour_analysis
from dominant_colour.colour_analysis import DominantColourFinder
from dominant_colour.constants import QUANTISE_BUCKET_SIZE


def quantise(pixels, bucket_size=QUANTISE_BUCKET_SIZE) -> np.ndarray:
    """Quantise an array of pixels and hand back the result."""
    finder = DominantColourFinder(pixels)
    finder.quantise_colours(bucket_size)

    return finder.quantised_pixels


def counted(colours, counts) -> DominantColourFinder:
    """A finder with its counted colours set directly, skipping quantise and count."""
    finder = DominantColourFinder(np.array(colours))
    finder.colours = np.array(colours)
    finder.counts = np.array(counts)

    return finder


def test_quantise_colours() -> None:
    """Each channel rounds down to the multiple of the bucket size below it."""
    quantised = quantise(np.array([[154, 255, 178]]))

    np.testing.assert_array_equal(quantised, np.array([[150, 250, 170]]))


def test_quantise_colours_max_rgb_values() -> None:
    """Quantising never pushes a channel above 255, which isn't a valid RGB value."""
    quantised = quantise(np.array([[255, 255, 255], [254, 251, 250]]))

    assert quantised.max() <= 255


def test_quantise_colours_bucket_size_20() -> None:
    """A bucket size other than the default groups values into wider steps."""
    quantised = quantise(np.array([[154, 255, 178]]), bucket_size=20)

    np.testing.assert_array_equal(quantised, np.array([[140, 240, 160]]))


def test_quantise_colours_shape() -> None:
    """Every pixel survives quantising"""
    pixels = np.array([[154, 255, 178], [0, 0, 0], [0, 0, 0]])

    assert quantise(pixels).shape == pixels.shape


def test_count_unique_colours() -> None:
    """Pixels sharing a bucket become one colour, counted once per pixel."""
    pixels = np.array([[[0, 0, 0], [0, 0, 0], [250, 250, 250], [253, 252, 251], [255, 255, 255]]], dtype=np.uint8)

    finder = DominantColourFinder(pixels)
    finder.quantise_colours(bucket_size=QUANTISE_BUCKET_SIZE)
    finder.count_unique_colours()

    counted = dict(zip(map(tuple, finder.colours), finder.counts))
    assert counted == {(0, 0, 0): 2, (250, 250, 250): 3}


def test_remove_ignored_colour_names(monkeypatch) -> None:
    """A named colour is dropped from both the colours and their counts."""
    monkeypatch.setattr(colour_analysis, "rgb_to_colour_names", lambda _: ["black", "yellow"])

    black, yellow = (0, 0, 0), (250, 250, 0)
    finder = counted([black, yellow], [2, 3])

    finder.remove_ignored_colour_names(["black"])

    np.testing.assert_array_equal(finder.colours, np.array([yellow]))
    np.testing.assert_array_equal(finder.counts, np.array([3]))


def test_remove_ignored_colour_names_multiple_colours(monkeypatch) -> None:
    """Every name in the list is dropped, not just the first."""
    monkeypatch.setattr(colour_analysis, "rgb_to_colour_names", lambda _: ["black", "white", "yellow"])

    black, white, yellow = (0, 0, 0), (250, 250, 250), (250, 250, 0)
    finder = counted([black, white, yellow], [2, 3, 1])

    finder.remove_ignored_colour_names(["black", "white"])

    np.testing.assert_array_equal(finder.colours, np.array([yellow]))
    np.testing.assert_array_equal(finder.counts, np.array([1]))


def test_remove_ignored_colour_names_empty(monkeypatch) -> None:
    """An empty ignore list is a no-op, which is what the CLI passes by default."""
    monkeypatch.setattr(colour_analysis, "rgb_to_colour_names", lambda _: ["black", "yellow"])

    black, yellow = (0, 0, 0), (250, 250, 0)
    finder = counted([black, yellow], [2, 3])

    finder.remove_ignored_colour_names([])

    np.testing.assert_array_equal(finder.colours, np.array([black, yellow]))
    np.testing.assert_array_equal(finder.counts, np.array([2, 3]))


def test_remove_ignored_colour_names_removes_all(monkeypatch) -> None:
    """Ignoring every colour empties both arrays rather than erroring."""
    monkeypatch.setattr(colour_analysis, "rgb_to_colour_names", lambda _: ["black", "yellow"])

    black, yellow = (0, 0, 0), (250, 250, 0)
    finder = counted([black, yellow], [2, 3])

    finder.remove_ignored_colour_names(["black", "yellow"])

    assert len(finder.colours) == 0
    assert len(finder.counts) == 0


def test_rank_top_colours() -> None:
    """Colours come back ordered by count, most frequent first."""
    black, white, yellow = (0, 0, 0), (250, 250, 250), (250, 250, 0)
    finder = counted([black, white, yellow], [1, 5, 3])

    assert finder.rank_top_colours(top_n=3) == [white, yellow, black]


def test_rank_top_colours_high_top_n() -> None:
    """Asking for more colours than the image has returns all of them, not an error."""
    black, yellow = (0, 0, 0), (250, 250, 0)
    finder = counted([black, yellow], [2, 3])

    assert finder.rank_top_colours(top_n=10) == [yellow, black]


def test_rank_top_colours_empty() -> None:
    """With no colours left, ranking returns an empty list for the CLI to report."""
    finder = counted([], [])

    assert finder.rank_top_colours(top_n=3) == []


def test_rank_top_colours_with_names(monkeypatch) -> None:
    """Each ranked colour is paired with its name, keeping the ranking order."""
    monkeypatch.setattr(colour_analysis, "rgb_to_colour_names", lambda _: ["white", "yellow"])

    black, white, yellow = (0, 0, 0), (250, 250, 250), (250, 250, 0)
    finder = counted([black, white, yellow], [1, 5, 3])

    assert finder.rank_top_colours_with_names(top_n=2) == [(white, "white"), (yellow, "yellow")]


def test_find_runs_the_whole_pipeline(monkeypatch) -> None:
    """The steps run in order, so colours come back ranked and named."""
    monkeypatch.setattr(colour_analysis, "rgb_to_colour_names", lambda colours: ["black"] * len(colours))

    pixels = np.array([[[0, 0, 0], [0, 0, 0], [250, 250, 250]]], dtype=np.uint8)

    assert DominantColourFinder(pixels).find() == [((0, 0, 0), "black")]
