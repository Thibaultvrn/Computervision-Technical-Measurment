"""Image preprocessing module for white sheet detection."""

import cv2
import numpy as np
from typing import Tuple


def grayscale(frame: np.ndarray) -> np.ndarray:
    """Convert BGR frame to grayscale."""
    if len(frame.shape) == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def blur(
    frame: np.ndarray,
    kernel_size: Tuple[int, int] = (5, 5)
) -> np.ndarray:
    """Apply Gaussian blur to reduce noise."""
    return cv2.GaussianBlur(frame, kernel_size, 0)


def edge_detect(
    frame: np.ndarray,
    low_threshold: int = 50,
    high_threshold: int = 150
) -> np.ndarray:
    """Detect edges using Canny algorithm."""
    return cv2.Canny(frame, low_threshold, high_threshold)


def threshold(
    frame: np.ndarray,
    thresh_value: int = 127,
    max_value: int = 255,
    method: int = cv2.THRESH_BINARY
) -> np.ndarray:
    """Apply binary threshold to image."""
    _, binary = cv2.threshold(frame, thresh_value, max_value, method)
    return binary


def detect_white_regions(
    frame: np.ndarray,
    min_brightness: int = 180,
    max_saturation: int = 60
) -> np.ndarray:
    """
    Detect white/bright regions (paper sheet).

    Args:
        frame: Input BGR image
        min_brightness: Minimum V value in HSV (brightness)
        max_saturation: Maximum S value in HSV (color saturation)

    Returns:
        Binary mask of white regions
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lower = np.array([0, 0, min_brightness])
    upper = np.array([180, max_saturation, 255])
    
    mask = cv2.inRange(hsv, lower, upper)
    return mask


def detect_bright_object(
    frame: np.ndarray,
    brightness_percentile: float = 85
) -> np.ndarray:
    """
    Detect bright objects using adaptive brightness threshold.

    Args:
        frame: Input BGR image
        brightness_percentile: Percentile for brightness cutoff

    Returns:
        Binary mask of bright regions
    """
    gray = grayscale(frame)
    thresh_value = np.percentile(gray, brightness_percentile)
    _, mask = cv2.threshold(gray, thresh_value, 255, cv2.THRESH_BINARY)
    return mask


def adaptive_threshold(
    frame: np.ndarray,
    block_size: int = 11,
    constant: int = 2
) -> np.ndarray:
    """Apply adaptive threshold for varying illumination."""
    return cv2.adaptiveThreshold(
        frame, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size, constant
    )


def morphological_clean(
    frame: np.ndarray,
    kernel_size: int = 3,
    iterations: int = 1
) -> np.ndarray:
    """Apply morphological operations to clean binary image."""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    cleaned = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=iterations)
    return cleaned


def preprocess_for_white_sheet(
    frame: np.ndarray,
    blur_kernel: Tuple[int, int] = (5, 5),
    min_brightness: int = 170,
    max_saturation: int = 70,
    morph_size: int = 5
) -> np.ndarray:
    """
    Preprocessing pipeline optimized for white paper sheet detection.

    Args:
        frame: Input BGR frame
        blur_kernel: Gaussian blur kernel size
        min_brightness: Minimum brightness threshold (0-255)
        max_saturation: Maximum saturation (0-255, low = white)
        morph_size: Morphological kernel size

    Returns:
        Binary image ready for contour detection
    """
    blurred = blur(frame, blur_kernel)
    
    white_mask = detect_white_regions(blurred, min_brightness, max_saturation)
    
    kernel = np.ones((morph_size, morph_size), np.uint8)
    cleaned = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return cleaned


def preprocess_pipeline(
    frame: np.ndarray,
    blur_kernel: Tuple[int, int] = (5, 5),
    thresh_value: int = 127
) -> np.ndarray:
    """
    Complete preprocessing pipeline for sheet detection.

    Args:
        frame: Input BGR frame
        blur_kernel: Gaussian blur kernel size
        thresh_value: Binary threshold value

    Returns:
        Preprocessed binary image ready for contour detection
    """
    gray = grayscale(frame)
    blurred = blur(gray, blur_kernel)
    binary = threshold(blurred, thresh_value)
    cleaned = morphological_clean(binary)
    return cleaned
