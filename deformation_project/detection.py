"""Contour detection module for vertical sheet extraction."""

import cv2
import numpy as np
from typing import Optional, List, Tuple


def find_contours(binary_frame: np.ndarray) -> List[np.ndarray]:
    """Find all contours in binary image."""
    contours, _ = cv2.findContours(
        binary_frame,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    return contours


def extract_central_vertical_contour(
    binary_frame: np.ndarray,
    min_area: int = 2000,
    min_aspect_ratio: float = 1.2,
    center_weight: float = 2.0
) -> Optional[np.ndarray]:
    """
    Extract vertical contour closest to image center.

    Prioritizes tall objects near the center of the frame.

    Args:
        binary_frame: Binary input image
        min_area: Minimum contour area
        min_aspect_ratio: Minimum height/width ratio
        center_weight: Weight for center proximity in scoring

    Returns:
        Contour as Nx2 numpy array, or None if no valid contour
    """
    contours = find_contours(binary_frame)

    if not contours:
        return None

    frame_h, frame_w = binary_frame.shape[:2]
    frame_center_x = frame_w / 2
    frame_center_y = frame_h / 2

    best_contour = None
    best_score = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = h / w if w > 0 else 0

        if aspect_ratio < min_aspect_ratio:
            continue

        contour_center_x = x + w / 2
        contour_center_y = y + h / 2

        dist_from_center = np.sqrt(
            ((contour_center_x - frame_center_x) / frame_w) ** 2 +
            ((contour_center_y - frame_center_y) / frame_h) ** 2
        )
        center_score = 1.0 / (1.0 + dist_from_center * center_weight)

        score = area * aspect_ratio * center_score

        if score > best_score:
            best_score = score
            best_contour = contour

    if best_contour is None:
        return None

    return best_contour.reshape(-1, 2)


def extract_vertical_sheet_contour(
    binary_frame: np.ndarray,
    min_area: int = 1000,
    min_aspect_ratio: float = 1.5,
    max_aspect_ratio: float = 20.0
) -> Optional[np.ndarray]:
    """
    Extract contour of vertical sheet (tall and narrow object).

    Args:
        binary_frame: Binary input image
        min_area: Minimum contour area
        min_aspect_ratio: Minimum height/width ratio
        max_aspect_ratio: Maximum height/width ratio

    Returns:
        Contour as Nx2 numpy array, or None if no valid contour
    """
    contours = find_contours(binary_frame)

    if not contours:
        return None

    best_contour = None
    best_score = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = h / w if w > 0 else 0

        if min_aspect_ratio <= aspect_ratio <= max_aspect_ratio:
            score = area * aspect_ratio
            if score > best_score:
                best_score = score
                best_contour = contour

    if best_contour is None:
        return None

    return best_contour.reshape(-1, 2)


def extract_largest_contour(
    binary_frame: np.ndarray,
    min_area: int = 100
) -> Optional[np.ndarray]:
    """
    Extract largest contour from binary frame with noise filtering.

    Args:
        binary_frame: Binary input image
        min_area: Minimum contour area to consider

    Returns:
        Contour as Nx2 numpy array, or None if no valid contour found
    """
    contours = find_contours(binary_frame)

    if not contours:
        return None

    areas = np.array([cv2.contourArea(c) for c in contours])
    valid_mask = areas > min_area

    if not np.any(valid_mask):
        return None

    valid_indices = np.where(valid_mask)[0]
    largest_idx = valid_indices[np.argmax(areas[valid_mask])]
    largest_contour = contours[largest_idx]

    return largest_contour.reshape(-1, 2)


def approximate_contour(
    contour: np.ndarray,
    epsilon_factor: float = 0.01
) -> np.ndarray:
    """
    Approximate contour with reduced point count.

    Args:
        contour: Input Nx2 contour array
        epsilon_factor: Approximation precision (fraction of perimeter)

    Returns:
        Approximated contour as Nx2 array
    """
    contour_cv = contour.reshape(-1, 1, 2).astype(np.int32)
    epsilon = epsilon_factor * cv2.arcLength(contour_cv, closed=True)
    approx = cv2.approxPolyDP(contour_cv, epsilon, closed=True)
    return approx.reshape(-1, 2)


def get_contour_bounds(contour: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Get bounding rectangle of contour.

    Args:
        contour: Input Nx2 contour array

    Returns:
        Tuple (x, y, width, height)
    """
    contour_cv = contour.reshape(-1, 1, 2).astype(np.int32)
    return cv2.boundingRect(contour_cv)


def sort_contour_points(
    contour: np.ndarray,
    axis: int = 1
) -> np.ndarray:
    """
    Sort contour points along specified axis.

    Args:
        contour: Input Nx2 contour array
        axis: Sort axis (0=x, 1=y)

    Returns:
        Sorted contour array
    """
    sorted_indices = np.argsort(contour[:, axis])
    return contour[sorted_indices]


def extract_edge_profile(
    contour: np.ndarray,
    side: str = "left"
) -> np.ndarray:
    """
    Extract one edge of the sheet contour (for lateral deformation).

    Args:
        contour: Input Nx2 contour array
        side: Edge to extract ('left', 'right', 'top', 'bottom')

    Returns:
        Edge profile as Nx2 array
    """
    sorted_contour = sort_contour_points(contour, axis=1)

    if side == "left":
        unique_y = np.unique(sorted_contour[:, 1])
        edge = np.array([
            [sorted_contour[sorted_contour[:, 1] == y, 0].min(), y]
            for y in unique_y
        ])
    elif side == "right":
        unique_y = np.unique(sorted_contour[:, 1])
        edge = np.array([
            [sorted_contour[sorted_contour[:, 1] == y, 0].max(), y]
            for y in unique_y
        ])
    elif side == "top":
        unique_x = np.unique(sorted_contour[:, 0])
        edge = np.array([
            [x, sorted_contour[sorted_contour[:, 0] == x, 1].min()]
            for x in unique_x
        ])
    else:  # bottom
        unique_x = np.unique(sorted_contour[:, 0])
        edge = np.array([
            [x, sorted_contour[sorted_contour[:, 0] == x, 1].max()]
            for x in unique_x
        ])

    return edge
