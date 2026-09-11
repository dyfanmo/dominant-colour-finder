from pathlib import Path

import numpy as np

from dominant_colour import image_editing
from dominant_colour.image_editing import black_out, default_output_path


def name_pixels(*names_by_pixel):
    """Stub the naming so each pixel's colour family is set by the test."""
    columns = {"black": 0, "blue": 1, "red": 8}

    def fake_rgb_to_colour_columns(pixels):
        return np.array([columns[name] for name in names_by_pixel]).reshape(pixels.shape[:-1])

    def fake_colour_names_to_columns(colour_names):
        return [columns[name] for name in colour_names]

    return fake_rgb_to_colour_columns, fake_colour_names_to_columns


def stub_naming(monkeypatch, *names_by_pixel) -> None:
    """Point image_editing at the stubbed naming for this test."""
    to_columns, names_to_columns = name_pixels(*names_by_pixel)
    monkeypatch.setattr(image_editing, "rgb_to_colour_columns", to_columns)
    monkeypatch.setattr(image_editing, "colour_names_to_columns", names_to_columns)


def test_default_output_path_sits_alongside_the_original() -> None:
    """The edit lands in the same folder, marked as edited."""
    assert default_output_path("photos/holiday.jpg") == Path("photos/holiday_edited.png")


def test_default_output_path_is_always_png() -> None:
    """The output is PNG whatever the input was, since the edit must be lossless."""
    assert default_output_path("photo.webp").suffix == ".png"


def test_black_out_paints_matching_pixels_black(monkeypatch) -> None:
    """Pixels of the named colour go black, the rest keep their values."""
    stub_naming(monkeypatch, "blue", "red")
    blue, red = (30, 80, 200), (200, 30, 30)
    pixels = np.array([[blue, red]], dtype=np.uint8)

    edited, changed = black_out(pixels, ["blue"])

    np.testing.assert_array_equal(edited, [[(0, 0, 0), red]])
    assert changed == 1


def test_black_out_counts_every_changed_pixel(monkeypatch) -> None:
    """The count is pixels changed, not colours matched."""
    stub_naming(monkeypatch, "blue", "blue", "red")
    blue, red = (30, 80, 200), (200, 30, 30)
    pixels = np.array([[blue, blue, red]], dtype=np.uint8)

    _edited, changed = black_out(pixels, ["blue"])

    assert changed == 2


def test_black_out_removes_multiple_colours(monkeypatch) -> None:
    """Every named colour goes, not just the first."""
    stub_naming(monkeypatch, "blue", "red", "black")
    blue, red, black = (30, 80, 200), (200, 30, 30), (0, 0, 0)
    pixels = np.array([[blue, red, black]], dtype=np.uint8)

    edited, changed = black_out(pixels, ["blue", "red"])

    np.testing.assert_array_equal(edited, [[black, black, black]])
    assert changed == 2


def test_black_out_leaves_the_original_untouched(monkeypatch) -> None:
    """The input array isn't modified, so the caller can reuse it."""
    stub_naming(monkeypatch, "blue")
    pixels = np.array([[(30, 80, 200)]], dtype=np.uint8)

    black_out(pixels, ["blue"])

    np.testing.assert_array_equal(pixels, [[(30, 80, 200)]])
