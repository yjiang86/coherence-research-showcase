"""Independent direct NumPy oracle for the frozen CPU coherence formula."""

from __future__ import annotations

import numpy as np


def direct_reference(
    lhs: np.ndarray,
    rhs: np.ndarray,
    frequency_hz: float,
    t_step: float,
    wpara: float,
) -> tuple[np.ndarray, int]:
    lhs_work = np.asarray(lhs, dtype=np.complex128)
    rhs_work = np.asarray(rhs, dtype=np.complex128)
    sigma = wpara / (2.0 * np.pi * frequency_hz)
    amp = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
    support = np.arange(-wpara / frequency_hz, wpara / frequency_hz, t_step, dtype=np.float64)
    smoothing = amp * np.exp(-(support**2) / (2.0 * sigma**2))
    smoothing = smoothing / smoothing.sum(dtype=np.float64)

    def same(values: np.ndarray) -> np.ndarray:
        full = np.convolve(values, smoothing, mode="full")
        start = smoothing.size // 2
        return full[start : start + values.size]

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        p1 = same(np.abs(lhs_work) ** 2)
        p2 = same(np.abs(rhs_work) ** 2)
        cross = lhs_work * np.conjugate(rhs_work)
        cross_smoothed = same(cross.real) + 1j * same(cross.imag)
        quotient = np.abs(cross_smoothed) ** 2 / np.maximum(p1 * p2, np.finfo(np.float64).eps)
    quotient = np.where(np.isfinite(quotient), quotient, np.float64(0.0))
    return np.array(np.clip(quotient, 0.0, 1.0), dtype=np.float32, order="C", copy=True), int(support.size)
