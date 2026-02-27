"""Configuration parameters for deformation measurement system."""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class CameraConfig:
    """Camera acquisition settings."""
    source: int | str = 0
    target_fps: int = 30
    frame_width: int = 640
    frame_height: int = 480


@dataclass
class PreprocessingConfig:
    """Image preprocessing parameters."""
    blur_kernel_size: Tuple[int, int] = (5, 5)
    canny_low: int = 50
    canny_high: int = 150
    threshold_value: int = 127
    threshold_max: int = 255


@dataclass
class DetectionConfig:
    """Contour detection parameters."""
    min_contour_area: int = 100
    approx_epsilon_factor: float = 0.01


@dataclass
class CalibrationConfig:
    """Calibration parameters."""
    scale_px_to_mm: float = 1.0
    reference_length_mm: float = 100.0


@dataclass
class VisualizationConfig:
    """Visualization settings."""
    contour_color: Tuple[int, int, int] = (0, 255, 0)
    midline_color: Tuple[int, int, int] = (255, 0, 0)
    text_color: Tuple[int, int, int] = (255, 255, 255)
    line_thickness: int = 2
    font_scale: float = 0.6


@dataclass
class SystemConfig:
    """Combined system configuration."""
    camera: CameraConfig = None
    preprocessing: PreprocessingConfig = None
    detection: DetectionConfig = None
    calibration: CalibrationConfig = None
    visualization: VisualizationConfig = None

    def __post_init__(self):
        self.camera = self.camera or CameraConfig()
        self.preprocessing = self.preprocessing or PreprocessingConfig()
        self.detection = self.detection or DetectionConfig()
        self.calibration = self.calibration or CalibrationConfig()
        self.visualization = self.visualization or VisualizationConfig()


def get_default_config() -> SystemConfig:
    """Return default system configuration."""
    return SystemConfig()
