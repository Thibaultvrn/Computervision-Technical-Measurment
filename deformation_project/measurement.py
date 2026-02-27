"""Measurement module for vertical sheet deformation analysis."""

import numpy as np
from typing import Tuple, Optional


def contour_to_midline(
    contour: np.ndarray,
    num_points: int = 100,
    smooth_window: int = 5
) -> np.ndarray:
    """
    Compute smoothed midline from vertical sheet contour.

    Args:
        contour: Input Nx2 contour array
        num_points: Number of points in output midline
        smooth_window: Smoothing window size

    Returns:
        Midline as Mx2 array of (x, y) coordinates, sorted top to bottom
    """
    if len(contour) < 4:
        return np.array([]).reshape(0, 2)

    y_min, y_max = contour[:, 1].min(), contour[:, 1].max()
    y_range = y_max - y_min
    
    if y_range < 10:
        return np.array([]).reshape(0, 2)

    y_samples = np.linspace(y_min + 2, y_max - 2, num_points)
    midline_points = []

    for y in y_samples:
        y_band = 3
        mask = np.abs(contour[:, 1] - y) < y_band
        if np.sum(mask) < 2:
            continue
        
        x_values = contour[mask, 0]
        x_mid = (x_values.min() + x_values.max()) / 2
        midline_points.append([x_mid, y])

    if len(midline_points) < 3:
        return np.array([]).reshape(0, 2)

    midline = np.array(midline_points)

    if smooth_window > 1 and len(midline) > smooth_window:
        kernel = np.ones(smooth_window) / smooth_window
        midline[:, 0] = np.convolve(midline[:, 0], kernel, mode='same')
        edge = smooth_window // 2
        midline = midline[edge:-edge] if edge > 0 else midline

    return midline


def compute_reference_line(midline: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute straight reference line between top and bottom fixation points.

    Args:
        midline: Mx2 array of midline points

    Returns:
        Tuple of (top_point, bottom_point) as 1D arrays
    """
    if len(midline) < 2:
        return np.array([0, 0]), np.array([0, 0])

    top_point = midline[0]
    bottom_point = midline[-1]
    return top_point, bottom_point


def compute_max_deflection(
    midline: np.ndarray,
    reference_line: Optional[Tuple[np.ndarray, np.ndarray]] = None
) -> Tuple[float, int, np.ndarray]:
    """
    Compute maximum lateral deflection from reference line.

    Args:
        midline: Mx2 array of midline points
        reference_line: Tuple of (top_point, bottom_point) or None to compute

    Returns:
        Tuple of (max_deflection_pixels, index_of_max, point_on_reference)
    """
    if len(midline) < 2:
        return 0.0, 0, np.array([0, 0])

    if reference_line is None:
        top, bottom = compute_reference_line(midline)
    else:
        top, bottom = reference_line

    direction = bottom - top
    length = np.linalg.norm(direction)
    
    if length < 1e-10:
        return 0.0, 0, top

    direction_norm = direction / length
    deflections = []

    for point in midline:
        to_point = point - top
        proj_length = np.dot(to_point, direction_norm)
        proj_point = top + proj_length * direction_norm
        deflection = point[0] - proj_point[0]
        deflections.append(deflection)

    deflections = np.array(deflections)
    abs_deflections = np.abs(deflections)
    max_idx = np.argmax(abs_deflections)

    t = (midline[max_idx, 1] - top[1]) / (bottom[1] - top[1]) if (bottom[1] - top[1]) != 0 else 0
    ref_point = top + t * (bottom - top)

    return float(deflections[max_idx]), int(max_idx), ref_point


def compute_curvature(midline: np.ndarray) -> np.ndarray:
    """
    Compute local curvature along midline.

    Curvature k = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)

    Args:
        midline: Mx2 array of midline points

    Returns:
        Array of curvature values
    """
    if len(midline) < 5:
        return np.array([])

    dx = np.gradient(midline[:, 0])
    dy = np.gradient(midline[:, 1])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    denominator = (dx**2 + dy**2)**1.5
    denominator = np.where(denominator < 1e-10, 1e-10, denominator)

    curvature = np.abs(dx * ddy - dy * ddx) / denominator
    return curvature


def compute_max_curvature(midline: np.ndarray) -> Tuple[float, int]:
    """
    Compute maximum curvature and its location.

    Args:
        midline: Mx2 array of midline points

    Returns:
        Tuple of (max_curvature, index)
    """
    curvature = compute_curvature(midline)
    if len(curvature) == 0:
        return 0.0, 0
    
    max_idx = np.argmax(curvature)
    return float(curvature[max_idx]), int(max_idx)


def compute_mean_curvature(midline: np.ndarray) -> float:
    """
    Compute mean curvature along midline.

    Args:
        midline: Mx2 array of midline points

    Returns:
        Mean curvature value in pixels^-1
    """
    curvature = compute_curvature(midline)
    return float(np.mean(curvature)) if len(curvature) > 0 else 0.0


def compute_curvature_radius(curvature: float) -> float:
    """
    Convert curvature to radius of curvature.

    Args:
        curvature: Curvature value (1/pixels)

    Returns:
        Radius in pixels (inf if curvature ~= 0)
    """
    if abs(curvature) < 1e-10:
        return float('inf')
    return 1.0 / curvature


def compute_deflection_profile(
    midline: np.ndarray,
    reference_x: Optional[float] = None
) -> np.ndarray:
    """
    Compute deflection profile along sheet height.

    Args:
        midline: Mx2 array of midline points
        reference_x: Reference x position

    Returns:
        Mx2 array of (y_position, deflection)
    """
    if len(midline) < 2:
        return np.array([]).reshape(0, 2)

    if reference_x is None:
        reference_x = (midline[0, 0] + midline[-1, 0]) / 2

    deflections = midline[:, 0] - reference_x
    return np.column_stack([midline[:, 1], deflections])


def compute_bending_moment_proxy(midline: np.ndarray) -> np.ndarray:
    """
    Compute proxy for bending moment (proportional to second derivative).

    Args:
        midline: Mx2 array of midline points

    Returns:
        Array of bending moment proxy values
    """
    if len(midline) < 3:
        return np.array([])

    d2x = np.gradient(np.gradient(midline[:, 0]))
    return d2x


def compute_arc_length(midline: np.ndarray) -> float:
    """
    Compute total arc length of midline.

    Args:
        midline: Mx2 array of midline points

    Returns:
        Arc length in pixels
    """
    if len(midline) < 2:
        return 0.0

    diffs = np.diff(midline, axis=0)
    segment_lengths = np.sqrt(np.sum(diffs**2, axis=1))
    return float(np.sum(segment_lengths))


def compute_all_metrics(midline: np.ndarray) -> dict:
    """
    Compute all deformation metrics from midline.

    Args:
        midline: Mx2 array of midline points

    Returns:
        Dictionary containing all computed metrics
    """
    if len(midline) < 3:
        return {}

    max_defl, max_idx, ref_point = compute_max_deflection(midline)
    max_curv, curv_idx = compute_max_curvature(midline)
    radius = compute_curvature_radius(max_curv)

    return {
        "max_deflection_px": max_defl,
        "max_deflection_idx": max_idx,
        "max_curvature": max_curv * 1000,
        "curvature_idx": curv_idx,
        "radius_px": radius if radius < 1e6 else float('inf'),
        "ref_point": ref_point,
        "midline_points": len(midline)
    }
