from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from dominant_colour.image_loading import (
    has_transparency,
    load_image_as_pixels,
    transparent_to_white,
)


@pytest.mark.parametrize("mode, expected", [("RGBA", True), ("LA", True), ("RGB", False), ("L", False)])
def test_has_transparency(mode, expected) -> None:
    """Only modes with an alpha channel count as transparent."""
    assert has_transparency(Image.new("RGB", (2, 2)).convert(mode)) is expected


def test_transparent_to_white() -> None:
    """Transparent areas come out white, and the mode is RGB."""
    image = Image.new("RGBA", (2, 2), (255, 0, 0, 0))

    result = transparent_to_white(image)

    assert result.mode == "RGB"
    assert result.getpixel((0, 0)) == (255, 255, 255)


@pytest.mark.parametrize("extension", ["png", "bmp", "gif", "tiff"])
def test_load_image_as_pixels_lossless_formats(tmp_path: Path, extension) -> None:
    """Lossless formats give back exactly the colours that went in."""
    image_path = tmp_path / f"red.{extension}"
    Image.new("RGB", (4, 3), color=(255, 0, 0)).save(image_path)

    result = load_image_as_pixels(image_path)

    assert result.shape == (3, 4, 3)
    assert result.dtype == np.uint8
    assert tuple(result[0, 0]) == (255, 0, 0)


@pytest.mark.parametrize("extension", ["jpg", "webp"])
def test_load_image_as_pixels_lossy_formats(tmp_path: Path, extension) -> None:
    """Lossy formats shift the colours a little, so they come back close but not exact."""
    image_path = tmp_path / f"red.{extension}"
    Image.new("RGB", (4, 3), color=(255, 0, 0)).save(image_path)

    result = load_image_as_pixels(image_path)

    assert result.shape == (3, 4, 3)
    assert result.dtype == np.uint8
    assert np.allclose(result[0, 0], (255, 0, 0), atol=5)


@pytest.mark.parametrize(
    "mode, extension",
    [("L", "png"), ("P", "png"), ("1", "png"), ("RGBA", "png"), ("CMYK", "jpg")],
)
def test_load_image_as_pixels_converts_every_mode_to_three_channels(tmp_path: Path, mode, extension) -> None:
    """Whatever mode the file is in, we always get three channels back."""
    # CMYK is the odd one out - PNG can't store it, so it has to be a JPEG.
    image_path = tmp_path / f"{mode}.{extension}"
    Image.new("RGB", (4, 3), color=(255, 0, 0)).convert(mode).save(image_path)

    result = load_image_as_pixels(image_path)

    assert result.shape == (3, 4, 3)
    assert result.dtype == np.uint8


def test_load_image_as_pixels_flattens_transparency_onto_white(tmp_path: Path) -> None:
    """Transparent areas become white, matching how a viewer would show them."""
    image_path = tmp_path / "transparent.png"
    Image.new("RGBA", (2, 2), color=(255, 0, 0, 0)).save(image_path)

    result = load_image_as_pixels(image_path)

    assert result.shape == (2, 2, 3)
    assert tuple(result[0, 0]) == (255, 255, 255)


def test_load_image_as_pixels_raises_on_missing_file(tmp_path: Path) -> None:
    """A path that doesn't exist gives a clear error."""
    missing_path = tmp_path / "does_not_exist.jpg"

    with pytest.raises(FileNotFoundError):
        load_image_as_pixels(missing_path)


def test_load_image_as_pixels_raises_on_a_directory(tmp_path: Path) -> None:
    """Passing a folder rather than a file gives a clear error."""
    folder = tmp_path / "a_folder"
    folder.mkdir()

    with pytest.raises(FileNotFoundError):
        load_image_as_pixels(folder)


def test_load_image_as_pixels_raises_on_invalid_image(tmp_path: Path) -> None:
    """A file that isn't image data at all gives a clear error."""
    not_an_image = tmp_path / "not_an_image.jpg"
    not_an_image.write_text("not an image")

    with pytest.raises(ValueError):
        load_image_as_pixels(not_an_image)


def test_load_image_as_pixels_raises_on_an_empty_file(tmp_path: Path) -> None:
    """A zero-byte file isn't an image."""
    empty = tmp_path / "empty.png"
    empty.write_bytes(b"")

    with pytest.raises(ValueError):
        load_image_as_pixels(empty)


def test_load_image_as_pixels_raises_on_a_truncated_image(tmp_path: Path) -> None:
    """A real image header with the data cut short should fail clearly."""
    full_image = tmp_path / "full.png"
    Image.new("RGB", (60, 60), color=(255, 0, 0)).save(full_image)

    truncated = tmp_path / "truncated.png"
    truncated.write_bytes(full_image.read_bytes()[:100])

    with pytest.raises(ValueError):
        load_image_as_pixels(truncated)


def test_load_image_as_pixels_raises_on_a_huge_image(tmp_path: Path, monkeypatch) -> None:
    """Pillow refuses images past its decompression bomb limit, around 89 megapixels."""
    # Drop the limit so the test doesn't need a genuinely huge file.
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 10)

    image_path = tmp_path / "big.png"
    Image.new("RGB", (20, 20), color=(255, 0, 0)).save(image_path)

    with pytest.raises(ValueError):
        load_image_as_pixels(image_path)
