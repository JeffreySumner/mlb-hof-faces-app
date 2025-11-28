"""
Image preprocessing utilities for Hall of Fame prediction.
"""

import numpy as np
from PIL import Image
from typing import Tuple


def preprocess_image(img: Image.Image) -> Tuple[np.ndarray, Image.Image, Image.Image]:
    """
    Preprocess image for model prediction.
    
    Args:
        img: PIL Image object (original image)
        
    Returns:
        Tuple of:
        - Flattened array ready for model (1024,)
        - Grayscale image (PIL Image)
        - Resized 32x32 image (PIL Image)
    """
    # Convert to grayscale
    img_gray = img.convert('L')
    
    # Resize to 32x32
    img_resized = img_gray.resize((32, 32))
    
    # Convert to array and normalize
    img_array = np.array(img_resized) / 255.0
    
    # Flatten for model input
    img_flat = img_array.flatten()
    
    return img_flat, img_gray, img_resized


def array_to_image(arr: np.ndarray, size: Tuple[int, int] = (32, 32)) -> Image.Image:
    """
    Convert numpy array back to PIL Image for display.
    
    Args:
        arr: Numpy array (normalized 0-1)
        size: Target size
        
    Returns:
        PIL Image
    """
    # Denormalize
    arr_uint8 = (arr * 255).astype(np.uint8)
    
    # Reshape if needed
    if arr.ndim == 1:
        arr_uint8 = arr_uint8.reshape(size)
    
    return Image.fromarray(arr_uint8, mode='L')
