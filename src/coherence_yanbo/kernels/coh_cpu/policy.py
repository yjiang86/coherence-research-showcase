"""Explicit local policy for the CPU coherence kernel."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


_VALID_MODES = frozenset(("direct", "fft", "fft_auto"))


def require_positive_float64(value: object, name: str) -> np.float64:
    """Return a finite positive binary64 scalar without accepting implicit types."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (float, np.float64)):
        raise TypeError(f"{name} must be a float64 scalar")
    result = np.float64(value)
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _require_positive_int(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


@dataclass(frozen=True, slots=True)
class ConvolutionPolicy:
    """Numerical and memory policy with no environment-derived behavior."""

    mode: str
    t_step: float
    wpara: float
    fft_kernel_threshold: int = 2048
    fft_signal_threshold: int = 131072
    fft_max_temp_bytes: int = 1 << 30

    def __post_init__(self) -> None:
        if not isinstance(self.mode, str) or self.mode not in _VALID_MODES:
            raise ValueError(f"mode must be one of {sorted(_VALID_MODES)!r}")
        object.__setattr__(self, "t_step", require_positive_float64(self.t_step, "t_step"))
        object.__setattr__(self, "wpara", require_positive_float64(self.wpara, "wpara"))
        object.__setattr__(
            self,
            "fft_kernel_threshold",
            _require_positive_int(self.fft_kernel_threshold, "fft_kernel_threshold"),
        )
        object.__setattr__(
            self,
            "fft_signal_threshold",
            _require_positive_int(self.fft_signal_threshold, "fft_signal_threshold"),
        )
        object.__setattr__(
            self,
            "fft_max_temp_bytes",
            _require_positive_int(self.fft_max_temp_bytes, "fft_max_temp_bytes"),
        )
