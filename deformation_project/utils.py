"""Utility functions for timing, data storage, and signal processing."""

import time
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class TimeSeriesBuffer:
    """Buffer for storing time series data (deformation vs time)."""

    max_length: int = 1000
    timestamps: List[float] = field(default_factory=list)
    values: Dict[str, List[float]] = field(default_factory=dict)

    def append(self, timestamp: float, metrics: Dict[str, float]) -> None:
        """Add new measurement to buffer."""
        self.timestamps.append(timestamp)

        for key, value in metrics.items():
            if key not in self.values:
                self.values[key] = []
            self.values[key].append(value)

        if len(self.timestamps) > self.max_length:
            self.timestamps.pop(0)
            for key in self.values:
                self.values[key].pop(0)

    def get_array(self, key: str) -> np.ndarray:
        """Get metric values as numpy array."""
        return np.array(self.values.get(key, []))

    def get_time_array(self) -> np.ndarray:
        """Get timestamps as numpy array."""
        return np.array(self.timestamps)

    def clear(self) -> None:
        """Clear all stored data."""
        self.timestamps.clear()
        self.values.clear()

    def to_dict(self) -> Dict[str, np.ndarray]:
        """Export all data as dictionary of arrays."""
        result = {"time": self.get_time_array()}
        for key in self.values:
            result[key] = self.get_array(key)
        return result


class FPSCounter:
    """Frame rate counter for performance monitoring."""

    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.timestamps: List[float] = []

    def tick(self) -> float:
        """Record frame and return current FPS."""
        now = time.perf_counter()
        self.timestamps.append(now)

        if len(self.timestamps) > self.window_size:
            self.timestamps.pop(0)

        if len(self.timestamps) < 2:
            return 0.0

        elapsed = self.timestamps[-1] - self.timestamps[0]
        return (len(self.timestamps) - 1) / elapsed if elapsed > 0 else 0.0


class Timer:
    """Context manager for timing code blocks."""

    def __init__(self):
        self.start_time: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start_time


def smooth_signal(
    signal: np.ndarray,
    window_size: int = 5
) -> np.ndarray:
    """
    Apply moving average smoothing to signal.

    Args:
        signal: Input 1D array
        window_size: Smoothing window size

    Returns:
        Smoothed signal array
    """
    if len(signal) < window_size:
        return signal
    kernel = np.ones(window_size) / window_size
    return np.convolve(signal, kernel, mode='valid')


def resample_signal(
    signal: np.ndarray,
    timestamps: np.ndarray,
    target_rate: float
) -> tuple:
    """
    Resample signal to uniform sampling rate for FFT.

    Args:
        signal: Input signal array
        timestamps: Original timestamps
        target_rate: Target sampling rate (Hz)

    Returns:
        Tuple of (resampled_signal, uniform_timestamps)
    """
    if len(signal) < 2:
        return signal, timestamps

    duration = timestamps[-1] - timestamps[0]
    n_samples = int(duration * target_rate)
    uniform_time = np.linspace(timestamps[0], timestamps[-1], n_samples)
    resampled = np.interp(uniform_time, timestamps, signal)

    return resampled, uniform_time


def save_measurement_data(
    filepath: str,
    data: Dict[str, np.ndarray]
) -> None:
    """
    Save measurement data to numpy compressed file.

    Args:
        filepath: Output file path
        data: Dictionary of numpy arrays to save
    """
    np.savez_compressed(filepath, **data)


def load_measurement_data(filepath: str) -> Dict[str, np.ndarray]:
    """
    Load measurement data from numpy file.

    Args:
        filepath: Input file path

    Returns:
        Dictionary of loaded arrays
    """
    loaded = np.load(filepath)
    return {key: loaded[key] for key in loaded.files}


def validate_contour(
    contour: Optional[np.ndarray],
    min_points: int = 10
) -> bool:
    """
    Validate contour has sufficient points for analysis.

    Args:
        contour: Input contour array
        min_points: Minimum required points

    Returns:
        True if contour is valid
    """
    if contour is None:
        return False
    return len(contour) >= min_points


def get_timestamp() -> float:
    """Return high-precision timestamp in seconds."""
    return time.perf_counter()
