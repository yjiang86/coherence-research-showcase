"""Public single- and multi-RHS CPU coherence API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator

import numpy as np

from .kernel import build_smoothing, same_convolution_row_halo
from .policy import ConvolutionPolicy, require_positive_float64


@dataclass(frozen=True, slots=True)
class CoherenceResult:
    """One immutable coherence buffer and the smoothing support used for it."""

    values: np.ndarray
    support_length: int


def _require_policy(policy: object) -> ConvolutionPolicy:
    if not isinstance(policy, ConvolutionPolicy):
        raise TypeError("policy must be a ConvolutionPolicy")
    return policy


def _validate_signal(value: object, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy array")
    if value.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if value.size <= 0:
        raise ValueError(f"{name} must be non-empty")
    if value.dtype not in (np.dtype(np.complex64), np.dtype(np.complex128)):
        raise TypeError(f"{name} must have complex64 or complex128 dtype")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain only finite values")
    return value


def _finalize(
    lhs_smoothed_power: np.ndarray,
    rhs_smoothed_power: np.ndarray,
    cross_smoothed: np.ndarray,
) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        denominator_safe = np.maximum(
            lhs_smoothed_power * rhs_smoothed_power,
            np.finfo(np.float64).eps,
        )
    quotient = np.abs(cross_smoothed) ** 2 / denominator_safe
    quotient = np.where(np.isfinite(quotient), quotient, np.float64(0.0))
    clipped = np.array(np.clip(quotient, 0.0, 1.0), dtype=np.float32, order="C", copy=True)
    immutable_payload = clipped.tobytes(order="C")
    return np.frombuffer(immutable_payload, dtype=np.float32)


def _smoothed_power(
    signal: np.ndarray,
    smoothing: np.ndarray,
    policy: ConvolutionPolicy,
) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        power = np.abs(signal) ** 2
    return same_convolution_row_halo(power, smoothing, policy)


def _compute_rhs(
    lhs_snapshot: np.ndarray,
    rhs: object,
    smoothing: np.ndarray,
    lhs_smoothed_power: np.ndarray,
    policy: ConvolutionPolicy,
) -> np.ndarray:
    rhs_array = _validate_signal(rhs, "rhs")
    if rhs_array.size != lhs_snapshot.size:
        raise ValueError("lhs and rhs must have the same positive length")
    rhs_work = np.asarray(rhs_array, dtype=np.complex128)
    rhs_smoothed_power = _smoothed_power(rhs_work, smoothing, policy)
    with np.errstate(over="ignore", invalid="ignore"):
        cross = lhs_snapshot * np.conjugate(rhs_work)
    cross_smoothed = (
        same_convolution_row_halo(cross.real, smoothing, policy)
        + 1j * same_convolution_row_halo(cross.imag, smoothing, policy)
    )
    return _finalize(lhs_smoothed_power, rhs_smoothed_power, cross_smoothed)


def _prepare_lhs(
    lhs: object,
    frequency_hz: object,
    policy: object,
) -> tuple[np.ndarray, np.float64, ConvolutionPolicy, np.ndarray, np.ndarray, int]:
    checked_policy = _require_policy(policy)
    lhs_array = _validate_signal(lhs, "lhs")
    frequency = require_positive_float64(frequency_hz, "frequency_hz")
    support, smoothing = build_smoothing(frequency, checked_policy.t_step, checked_policy.wpara)
    # The generator must never combine cached lhs power with caller mutations after a yield.
    lhs_snapshot = np.array(lhs_array, dtype=np.complex128, order="C", copy=True)
    lhs_smoothed_power = _smoothed_power(lhs_snapshot, smoothing, checked_policy)
    return lhs_snapshot, frequency, checked_policy, smoothing, lhs_smoothed_power, int(support.size)


def compute_single(
    lhs: np.ndarray,
    rhs: np.ndarray,
    frequency_hz: float,
    policy: ConvolutionPolicy,
) -> CoherenceResult:
    """Compute one frozen CPU coherence result with an exact-length float32 output."""
    lhs_snapshot, _, checked_policy, smoothing, lhs_smoothed_power, support_length = _prepare_lhs(
        lhs,
        frequency_hz,
        policy,
    )
    values = _compute_rhs(lhs_snapshot, rhs, smoothing, lhs_smoothed_power, checked_policy)
    return CoherenceResult(values=values, support_length=support_length)


def iter_multi_rhs(
    lhs: np.ndarray,
    rhs_iterable: Iterable[np.ndarray],
    frequency_hz: float,
    policy: ConvolutionPolicy,
) -> Iterator[CoherenceResult]:
    """Lazily compute one RHS at a time while retaining only shared lhs state."""
    lhs_snapshot, _, checked_policy, smoothing, lhs_smoothed_power, support_length = _prepare_lhs(
        lhs,
        frequency_hz,
        policy,
    )
    for rhs in rhs_iterable:
        values = _compute_rhs(lhs_snapshot, rhs, smoothing, lhs_smoothed_power, checked_policy)
        yield CoherenceResult(values=values, support_length=support_length)
