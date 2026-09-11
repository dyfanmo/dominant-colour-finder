from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from PIL.Image import DecompressionBombError

TRANSPARENT_MODES = ("RGBA", "LA", "PA")
WHITE = (255, 255, 255)


def has_transparency(image) -> bool:
    """True if the image has an alpha channel, so some pixels may be invisible."""
    return image.mode in TRANSPARENT_MODES


def transparent_to_white(image) -> Image.Image:
    """Put the image on a white background, so transparent areas read as white."""
    rgba_image = image.convert("RGBA")
    background = Image.new("RGB", rgba_image.size, WHITE)
    background.paste(rgba_image, mask=rgba_image.getchannel("A"))

    return background


def load_image_as_pixels(image_path: str | Path) -> NDArray:
    """Load an image file and return it as an (H, W, 3) RGB pixel array."""
    path = Path(image_path)

    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}")

    try:
        with Image.open(path) as image:
            rgb_image = transparent_to_white(image) if has_transparency(image) else image.convert("RGB")

            return np.array(rgb_image, dtype=np.uint8)
    except DecompressionBombError as exc:
        raise ValueError(f"Image is too large to process safely: {path}") from exc
    except OSError as exc:
        raise ValueError(f"File is not a valid image: {path}") from exc
