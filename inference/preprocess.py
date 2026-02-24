"""Image preprocessing for HOF model inference."""

from typing import Tuple

import numpy as np
from PIL import Image, ImageOps

from config import MODEL_IMAGE_SIZE


def _resize_with_padding(img_gray: Image.Image, size: int) -> Image.Image:
    """Resize while preserving aspect ratio, padding to square."""
    # contain() preserves aspect ratio; pad to exact square target
    contained = ImageOps.contain(img_gray, (size, size), method=Image.Resampling.BILINEAR)
    return ImageOps.pad(contained, (size, size), color=0, method=Image.Resampling.BILINEAR)


def preprocess_for_model(img: Image.Image, size: int = MODEL_IMAGE_SIZE) -> Tuple[np.ndarray, Image.Image, Image.Image]:
    """
    Convert an image to model-ready grayscale matrix.

    Returns
    -------
    tuple
        (image_array_2d, grayscale_image, resized_image)
    """
    img_gray = img.convert("L")
    img_resized = _resize_with_padding(img_gray, size=size)
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    return img_array, img_gray, img_resized

