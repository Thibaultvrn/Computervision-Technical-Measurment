"""Resolution-independent visualization module for adaptive overlays."""

import cv2
import numpy as np
from typing import Tuple, Dict, Any, Optional


def get_scale_params(frame: np.ndarray) -> Dict[str, float]:
    """
    Compute dynamic scale parameters based on frame dimensions.

    Args:
        frame: Input frame (H, W, 3) or (H, W)

    Returns:
        Dictionary with adaptive sizes:
            font_scale: Proportional font size
            thickness: Line thickness for drawings
            thin_thickness: Thinner line for subtle elements
            circle_radius: Marker circle radius
            small_offset: Small spacing (e.g., padding)
            large_offset: Large spacing (e.g., line spacing)
    """
    h, w = frame.shape[:2]
    base_dim = min(h, w)

    return {
        "font_scale": base_dim / 800.0,
        "thickness": max(1, int(base_dim / 300)),
        "thin_thickness": max(1, int(base_dim / 500)),
        "circle_radius": max(3, int(base_dim / 100)),
        "small_offset": max(5, int(base_dim * 0.02)),
        "large_offset": max(15, int(base_dim * 0.05)),
    }


def get_text_position(
    frame: np.ndarray,
    x_ratio: float = 0.02,
    y_ratio: float = 0.05
) -> Tuple[int, int]:
    """
    Compute text position relative to frame dimensions.

    Args:
        frame: Input frame
        x_ratio: Horizontal position as fraction of width
        y_ratio: Vertical position as fraction of height

    Returns:
        (x, y) pixel coordinates
    """
    h, w = frame.shape[:2]
    return int(w * x_ratio), int(h * y_ratio)


def overlay_contour(
    frame: np.ndarray,
    contour: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    scale_params: Optional[Dict[str, float]] = None
) -> np.ndarray:
    """
    Draw contour overlay with adaptive thickness.

    Args:
        frame: Input BGR frame
        contour: Nx2 contour array
        color: BGR color tuple
        scale_params: Dynamic scale parameters (computed if None)

    Returns:
        Frame with contour overlay
    """
    output = frame.copy()
    if contour is None or len(contour) == 0:
        return output

    params = scale_params or get_scale_params(frame)
    thickness = params["thickness"]

    contour_cv = contour.reshape(-1, 1, 2).astype(np.int32)
    cv2.drawContours(output, [contour_cv], -1, color, thickness)
    return output


def overlay_midline(
    frame: np.ndarray,
    midline: np.ndarray,
    color: Tuple[int, int, int] = (255, 0, 0),
    scale_params: Optional[Dict[str, float]] = None
) -> np.ndarray:
    """
    Draw midline overlay with adaptive thickness.

    Args:
        frame: Input BGR frame
        midline: Mx2 midline array
        color: BGR color tuple
        scale_params: Dynamic scale parameters (computed if None)

    Returns:
        Frame with midline overlay
    """
    output = frame.copy()
    if midline is None or len(midline) < 2:
        return output

    params = scale_params or get_scale_params(frame)
    thickness = params["thickness"]

    points = midline.astype(np.int32)
    cv2.polylines(output, [points], isClosed=False, color=color, thickness=thickness)
    return output


def overlay_deflection_marker(
    frame: np.ndarray,
    midline: np.ndarray,
    max_idx: int,
    reference_x: Optional[float] = None,
    color: Tuple[int, int, int] = (0, 0, 255),
    scale_params: Optional[Dict[str, float]] = None
) -> np.ndarray:
    """
    Draw deflection measurement marker with adaptive sizing.

    Args:
        frame: Input BGR frame
        midline: Mx2 midline array
        max_idx: Index of maximum deflection point
        reference_x: Reference x position
        color: BGR color tuple
        scale_params: Dynamic scale parameters (computed if None)

    Returns:
        Frame with deflection marker
    """
    output = frame.copy()
    if midline is None or len(midline) <= max_idx:
        return output

    params = scale_params or get_scale_params(frame)
    thickness = params["thickness"]
    radius = params["circle_radius"]

    if reference_x is None:
        reference_x = (midline[0, 0] + midline[-1, 0]) / 2

    point = midline[max_idx]
    ref_point = (int(reference_x), int(point[1]))
    defl_point = (int(point[0]), int(point[1]))

    cv2.line(output, ref_point, defl_point, color, thickness)
    cv2.circle(output, defl_point, radius, color, -1)

    return output


def display_metrics(
    frame: np.ndarray,
    values_dict: Dict[str, Any],
    color: Tuple[int, int, int] = (255, 255, 255),
    scale_params: Optional[Dict[str, float]] = None,
    x_ratio: float = 0.02,
    y_ratio: float = 0.06
) -> np.ndarray:
    """
    Display metric values with adaptive text sizing and positioning.

    Args:
        frame: Input BGR frame
        values_dict: Dictionary of metric names and values
        color: BGR text color
        scale_params: Dynamic scale parameters (computed if None)
        x_ratio: Horizontal position as fraction of width
        y_ratio: Starting vertical position as fraction of height

    Returns:
        Frame with metrics overlay
    """
    output = frame.copy()
    if not values_dict:
        return output

    params = scale_params or get_scale_params(frame)
    font_scale = params["font_scale"]
    thickness = max(1, int(params["thickness"] * 0.6))
    line_spacing = params["large_offset"]

    h, w = frame.shape[:2]
    x = int(w * x_ratio)
    y = int(h * y_ratio)

    font = cv2.FONT_HERSHEY_SIMPLEX

    for i, (key, value) in enumerate(values_dict.items()):
        if isinstance(value, float):
            text = f"{key}: {value:.2f}"
        else:
            text = f"{key}: {value}"

        text_y = y + int(i * line_spacing)
        cv2.putText(output, text, (x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)

    return output


def draw_reference_line(
    frame: np.ndarray,
    start_point: np.ndarray,
    end_point: np.ndarray,
    color: Tuple[int, int, int] = (100, 100, 100),
    scale_params: Optional[Dict[str, float]] = None
) -> np.ndarray:
    """
    Draw reference line between two fixation points.

    Args:
        frame: Input BGR frame
        start_point: Top fixation point (x, y)
        end_point: Bottom fixation point (x, y)
        color: BGR color tuple
        scale_params: Dynamic scale parameters (computed if None)

    Returns:
        Frame with reference line
    """
    output = frame.copy()
    params = scale_params or get_scale_params(frame)
    thickness = params["thin_thickness"]

    pt1 = (int(start_point[0]), int(start_point[1]))
    pt2 = (int(end_point[0]), int(end_point[1]))
    cv2.line(output, pt1, pt2, color, thickness)
    return output


def draw_roi(
    frame: np.ndarray,
    roi: Tuple[int, int, int, int],
    color: Tuple[int, int, int] = (255, 255, 0),
    scale_params: Optional[Dict[str, float]] = None
) -> np.ndarray:
    """
    Draw ROI rectangle on frame.

    Args:
        frame: Input BGR frame
        roi: Tuple (x, y, width, height)
        color: BGR color tuple
        scale_params: Dynamic scale parameters

    Returns:
        Frame with ROI overlay
    """
    output = frame.copy()
    params = scale_params or get_scale_params(frame)
    thickness = params["thin_thickness"]

    x, y, w, h = roi
    cv2.rectangle(output, (x, y), (x + w, y + h), color, thickness)
    return output


def overlay_deflection_line(
    frame: np.ndarray,
    midline_point: np.ndarray,
    ref_point: np.ndarray,
    color: Tuple[int, int, int] = (0, 0, 255),
    scale_params: Optional[Dict[str, float]] = None
) -> np.ndarray:
    """
    Draw deflection measurement line from midline to reference.

    Args:
        frame: Input BGR frame
        midline_point: Point on midline (x, y)
        ref_point: Corresponding point on reference line (x, y)
        color: BGR color tuple
        scale_params: Dynamic scale parameters

    Returns:
        Frame with deflection line
    """
    output = frame.copy()
    params = scale_params or get_scale_params(frame)
    thickness = params["thickness"]
    radius = params["circle_radius"]

    pt1 = (int(ref_point[0]), int(ref_point[1]))
    pt2 = (int(midline_point[0]), int(midline_point[1]))

    cv2.line(output, pt1, pt2, color, thickness)
    cv2.circle(output, pt2, radius, color, -1)

    return output


def create_visualization(
    frame: np.ndarray,
    contour: Optional[np.ndarray],
    midline: Optional[np.ndarray],
    metrics: Dict[str, Any],
    roi: Optional[Tuple[int, int, int, int]] = None,
    contour_color: Tuple[int, int, int] = (0, 255, 0),
    midline_color: Tuple[int, int, int] = (255, 0, 0),
    text_color: Tuple[int, int, int] = (255, 255, 255)
) -> np.ndarray:
    """
    Create complete visualization with adaptive overlays.

    Args:
        frame: Input BGR frame
        contour: Nx2 contour array
        midline: Mx2 midline array
        metrics: Dictionary of computed metrics
        roi: Optional ROI rectangle (x, y, w, h)
        contour_color: BGR color for contour
        midline_color: BGR color for midline
        text_color: BGR color for text

    Returns:
        Frame with all visualizations
    """
    scale_params = get_scale_params(frame)
    output = frame.copy()

    if roi is not None:
        output = draw_roi(output, roi, scale_params=scale_params)

    if contour is not None:
        output = overlay_contour(output, contour, contour_color, scale_params)

    if midline is not None and len(midline) > 1:
        output = draw_reference_line(
            output, midline[0], midline[-1],
            color=(100, 100, 100), scale_params=scale_params
        )

        output = overlay_midline(output, midline, midline_color, scale_params)

        if "max_deflection_idx" in metrics and "ref_point" in metrics:
            max_idx = metrics["max_deflection_idx"]
            ref_point = metrics["ref_point"]
            output = overlay_deflection_line(
                output, midline[max_idx], ref_point,
                scale_params=scale_params
            )

    display_dict = {
        k: v for k, v in metrics.items()
        if not k.endswith("_idx") and k != "ref_point"
    }
    output = display_metrics(output, display_dict, text_color, scale_params)

    return output


def create_window(window_name: str) -> None:
    """
    Create resizable OpenCV window that preserves aspect ratio.

    Args:
        window_name: Name identifier for the window
    """
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)


def set_fullscreen(window_name: str, fullscreen: bool) -> None:
    """
    Toggle window fullscreen mode.

    Args:
        window_name: Name of the window
        fullscreen: True for fullscreen, False for normal
    """
    flag = cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, flag)


def show_frame(window_name: str, frame: np.ndarray, wait_ms: int = 1) -> int:
    """
    Display frame and return key press.

    Args:
        window_name: Name of display window
        frame: Frame to display
        wait_ms: Milliseconds to wait for key (1 for ~60 FPS compatible)

    Returns:
        Key code of pressed key (-1 if none)
    """
    cv2.imshow(window_name, frame)
    return cv2.waitKey(wait_ms) & 0xFF


def cleanup_windows() -> None:
    """Close all OpenCV windows."""
    cv2.destroyAllWindows()
