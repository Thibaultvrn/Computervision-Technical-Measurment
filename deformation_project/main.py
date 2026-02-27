"""Main application for vertical sheet deformation measurement."""

import cv2
import numpy as np
from typing import Optional, Tuple

from config import get_default_config, SystemConfig
from acquisition import open_camera, read_frame, release_camera, get_capture_properties
from preprocessing import preprocess_for_white_sheet
from detection import extract_central_vertical_contour
from measurement import contour_to_midline, compute_all_metrics
from calibration import convert_metrics_to_mm
from visualization import (
    create_visualization, show_frame, cleanup_windows,
    create_window, set_fullscreen
)
from utils import TimeSeriesBuffer, FPSCounter, get_timestamp, validate_contour


class DetectionParams:
    """Adjustable detection parameters."""
    def __init__(self):
        self.min_brightness = 170
        self.max_saturation = 70
        self.min_area = 3000
        self.min_aspect_ratio = 1.2


def select_roi(frame: np.ndarray, window_name: str) -> Optional[Tuple[int, int, int, int]]:
    """Allow user to select ROI with mouse."""
    roi = cv2.selectROI(window_name, frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(window_name)
    create_window(window_name)
    
    if roi[2] > 0 and roi[3] > 0:
        return roi
    return None


def apply_roi(frame: np.ndarray, roi: Tuple[int, int, int, int]) -> np.ndarray:
    """Extract ROI region from frame."""
    x, y, w, h = roi
    return frame[y:y+h, x:x+w]


def offset_contour(contour: np.ndarray, roi: Tuple[int, int, int, int]) -> np.ndarray:
    """Offset contour coordinates back to full frame."""
    if contour is None:
        return None
    offset = np.array([roi[0], roi[1]])
    return contour + offset


def process_frame(
    frame: np.ndarray,
    params: DetectionParams,
    roi: Optional[Tuple[int, int, int, int]] = None
) -> tuple:
    """
    Process single frame through detection and measurement pipeline.

    Args:
        frame: Input BGR frame
        params: Detection parameters
        roi: Optional ROI (x, y, w, h)

    Returns:
        Tuple of (contour, midline, metrics) in full frame coordinates
    """
    process_region = apply_roi(frame, roi) if roi else frame

    binary = preprocess_for_white_sheet(
        process_region,
        blur_kernel=(7, 7),
        min_brightness=params.min_brightness,
        max_saturation=params.max_saturation,
        morph_size=5
    )

    contour = extract_central_vertical_contour(
        binary,
        min_area=params.min_area,
        min_aspect_ratio=params.min_aspect_ratio,
        center_weight=2.0
    )

    if not validate_contour(contour, min_points=20):
        return None, None, {}

    if roi:
        contour = offset_contour(contour, roi)

    midline = contour_to_midline(contour, num_points=80, smooth_window=7)
    
    if len(midline) < 5:
        return contour, None, {}
    
    metrics = compute_all_metrics(midline)

    return contour, midline, metrics


def run_measurement_loop(
    source: int | str = 0
) -> None:
    """Main measurement loop with real-time visualization."""
    config = get_default_config()
    params = DetectionParams()
    
    cap = open_camera(
        source,
        target_fps=config.camera.target_fps,
        frame_size=(config.camera.frame_width, config.camera.frame_height)
    )

    props = get_capture_properties(cap)
    print(f"Camera: {props['width']}x{props['height']} @ {props['fps']:.1f} FPS")

    time_series = TimeSeriesBuffer(max_length=5000)
    fps_counter = FPSCounter()
    window_name = "Sheet Deformation Measurement"
    is_fullscreen = False
    roi = None
    show_binary = False

    create_window(window_name)

    try:
        while True:
            success, frame = read_frame(cap)
            if not success:
                break

            timestamp = get_timestamp()
            contour, midline, metrics = process_frame(frame, params, roi)

            if metrics:
                time_series.append(timestamp, {
                    "deflection": metrics.get("max_deflection_px", 0),
                    "curvature": metrics.get("max_curvature", 0)
                })

            fps = fps_counter.tick()
            display_metrics = {k: v for k, v in metrics.items() 
                            if k not in ("ref_point", "curvature_idx")}
            display_metrics["FPS"] = fps

            if config.calibration.scale_px_to_mm != 1.0:
                display_metrics = convert_metrics_to_mm(
                    display_metrics,
                    config.calibration.scale_px_to_mm
                )

            if show_binary:
                process_region = apply_roi(frame, roi) if roi else frame
                binary = preprocess_for_white_sheet(
                    process_region,
                    min_brightness=params.min_brightness,
                    max_saturation=params.max_saturation
                )
                output = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            else:
                output = create_visualization(
                    frame, contour, midline, metrics, roi,
                    contour_color=config.visualization.contour_color,
                    midline_color=config.visualization.midline_color,
                    text_color=config.visualization.text_color
                )

            key = show_frame(window_name, output)

            if key == 27:  # ESC
                break
            elif key == ord('f'):
                is_fullscreen = not is_fullscreen
                set_fullscreen(window_name, is_fullscreen)
            elif key == ord('r'):
                print("Select ROI - drag rectangle, then press ENTER or SPACE")
                new_roi = select_roi(frame, window_name)
                if new_roi:
                    roi = new_roi
                    print(f"ROI set: {roi}")
                else:
                    roi = None
                    print("ROI cleared")
            elif key == ord('b'):
                show_binary = not show_binary
                print(f"Binary view: {'ON' if show_binary else 'OFF'}")
            elif key == ord('+') or key == ord('='):
                params.min_brightness = min(255, params.min_brightness + 5)
                print(f"Brightness threshold: {params.min_brightness}")
            elif key == ord('-'):
                params.min_brightness = max(50, params.min_brightness - 5)
                print(f"Brightness threshold: {params.min_brightness}")
            elif key == ord('s'):
                data = time_series.to_dict()
                np.savez_compressed("data/measurement_data.npz", **data)
                print(f"Saved {len(time_series.timestamps)} samples")
            elif key == ord('c'):
                time_series.clear()
                print("Buffer cleared")

    finally:
        release_camera(cap)
        cleanup_windows()

    print(f"Recorded {len(time_series.timestamps)} measurements")


def main():
    """Application entry point."""
    print("=== White Sheet Deformation Measurement ===")
    print("")
    print("Position the white sheet in front of the camera (30-50cm)")
    print("")
    print("Controls:")
    print("  R = Select ROI (recommended for better detection)")
    print("  B = Toggle binary mask view (debug)")
    print("  +/- = Adjust brightness threshold")
    print("  F = Toggle fullscreen")
    print("  S = Save data")
    print("  C = Clear buffer")
    print("  ESC = Quit")
    print("")

    run_measurement_loop(source=0)


if __name__ == "__main__":
    main()
