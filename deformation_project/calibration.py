"""Calibration module for pixel-to-metric conversion."""

import numpy as np
from typing import Tuple, Optional


def pixel_to_metric(
    value_px: float,
    scale_px_to_mm: float
) -> float:
    """
    Convert pixel measurement to metric units.

    Args:
        value_px: Value in pixels
        scale_px_to_mm: Scale factor (mm per pixel)

    Returns:
        Value in millimeters
    """
    return value_px * scale_px_to_mm


def metric_to_pixel(
    value_mm: float,
    scale_px_to_mm: float
) -> float:
    """
    Convert metric measurement to pixels.

    Args:
        value_mm: Value in millimeters
        scale_px_to_mm: Scale factor (mm per pixel)

    Returns:
        Value in pixels
    """
    return value_mm / scale_px_to_mm if scale_px_to_mm != 0 else 0.0


def compute_scale_from_reference(
    reference_length_px: float,
    reference_length_mm: float
) -> float:
    """
    Compute scale factor from known reference object.

    Args:
        reference_length_px: Reference length measured in pixels
        reference_length_mm: Known reference length in mm

    Returns:
        Scale factor (mm per pixel)
    """
    if reference_length_px <= 0:
        return 0.0
    return reference_length_mm / reference_length_px


def convert_contour_to_metric(
    contour: np.ndarray,
    scale_px_to_mm: float,
    origin_px: Optional[Tuple[float, float]] = None
) -> np.ndarray:
    """
    Convert contour coordinates from pixels to metric units.

    Args:
        contour: Nx2 array in pixel coordinates
        scale_px_to_mm: Scale factor (mm per pixel)
        origin_px: Optional origin offset in pixels

    Returns:
        Nx2 array in millimeter coordinates
    """
    if origin_px is not None:
        contour = contour - np.array(origin_px)
    return contour * scale_px_to_mm


def convert_metrics_to_mm(
    metrics: dict,
    scale_px_to_mm: float
) -> dict:
    """
    Convert pixel-based metrics dictionary to millimeters.

    Args:
        metrics: Dictionary with '_px' suffix keys
        scale_px_to_mm: Scale factor (mm per pixel)

    Returns:
        Dictionary with converted values and '_mm' suffix
    """
    result = {}
    for key, value in metrics.items():
        if key.endswith("_px") and isinstance(value, (int, float)):
            new_key = key.replace("_px", "_mm")
            result[new_key] = pixel_to_metric(value, scale_px_to_mm)
        else:
            result[key] = value
    return result


class CalibrationState:
    """Placeholder class for future camera calibration implementation."""

    def __init__(self):
        self.scale_px_to_mm: float = 1.0
        self.camera_matrix: Optional[np.ndarray] = None
        self.dist_coeffs: Optional[np.ndarray] = None
        self.is_calibrated: bool = False

    def set_scale(self, scale: float) -> None:
        """Set pixel-to-mm scale factor."""
        self.scale_px_to_mm = scale

    def undistort_points(self, points: np.ndarray) -> np.ndarray:
        """Placeholder for lens distortion correction."""
        if self.camera_matrix is None or self.dist_coeffs is None:
            return points
        return points

    def calibrate_from_checkerboard(
        self,
        images: list,
        pattern_size: Tuple[int, int],
        square_size_mm: float
    ) -> bool:
        """Placeholder for checkerboard calibration."""
        return False
