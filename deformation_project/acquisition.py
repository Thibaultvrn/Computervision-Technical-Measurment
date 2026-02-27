"""Video acquisition module for camera and IP stream capture."""

import cv2
import numpy as np
from typing import Union, Optional, Tuple


def open_camera(
    source: Union[int, str],
    target_fps: Optional[int] = None,
    frame_size: Optional[Tuple[int, int]] = None
) -> cv2.VideoCapture:
    """
    Open video capture from webcam index or IP stream URL.

    Args:
        source: Webcam index (int) or IP stream URL (str)
        target_fps: Target frames per second (locked if supported)
        frame_size: Tuple of (width, height) for frame resolution

    Returns:
        Configured cv2.VideoCapture object
    """
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video source: {source}")

    if frame_size is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_size[1])

    if target_fps is not None:
        cap.set(cv2.CAP_PROP_FPS, target_fps)

    return cap


def read_frame(cap: cv2.VideoCapture) -> Tuple[bool, Optional[np.ndarray]]:
    """
    Read single frame from video capture.

    Args:
        cap: OpenCV VideoCapture object

    Returns:
        Tuple of (success_flag, frame_array)
    """
    return cap.read()


def get_capture_properties(cap: cv2.VideoCapture) -> dict:
    """
    Get current capture device properties.

    Args:
        cap: OpenCV VideoCapture object

    Returns:
        Dictionary with fps, width, height properties
    """
    return {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    }


def release_camera(cap: cv2.VideoCapture) -> None:
    """Release video capture resources."""
    cap.release()
