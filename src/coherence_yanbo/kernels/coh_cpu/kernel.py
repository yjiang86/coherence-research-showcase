"""Pure NumPy smoothing and same-convolution primitives for CPU coherence."""

from __future__ import annotations

import numpy as np

from .policy import ConvolutionPolicy, require_positive_float64


def build_smoothing(
    frequency_hz: object,
    t_step: object,
    wpara: object,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the frozen normalized Morlet smoothing kernel."""
    frequency = require_positive_float64(frequency_hz, "frequency_hz")
    step = require_positive_float64(t_step, "t_step")
    width = require_positive_float64(wpara, "wpara")
    sigma = width / (2.0 * np.pi * frequency)
    amp = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
    support = np.arange(-width / frequency, width / frequency, step, dtype=np.float64)
    smoothing = amp * np.exp(-(support**2) / (2.0 * sigma**2))
    smoothing = np.ascontiguousarray(smoothing / smoothing.sum(dtype=np.float64))
    return support, smoothing


def _next_power_of_two(value: int) -> int:
    return 1 << (max(1, int(value)) - 1).bit_length()


def _fft_temp_bytes(signal_length: int, kernel_length: int) -> int:
    """Conservative working estimate for one real FFT convolution."""
    return 64 * _next_power_of_two(signal_length + kernel_length - 1)


def selected_mode(
    policy: ConvolutionPolicy,
    signal_length: int,
    kernel_length: int,
) -> str:
    """Resolve an explicit mode without reading process configuration."""
    if policy.mode != "fft_auto":
        return policy.mode
    if kernel_length < policy.fft_kernel_threshold:
        return "direct"
    if signal_length < policy.fft_signal_threshold:
        return "direct"
    if _fft_temp_bytes(signal_length, kernel_length) > policy.fft_max_temp_bytes:
        return "direct"
    return "fft"


def same_convolution_row_halo(
    row_halo: np.ndarray,
    smoothing: np.ndarray,
    policy: ConvolutionPolicy,
) -> np.ndarray:
    """Convolve one caller-supplied row/halo segment and retain its exact length.

    The caller supplies any halo explicitly. This primitive has no chunk index,
    merge, completion, or provenance interpretation.
    """
    if row_halo.ndim != 1 or smoothing.ndim != 1:
        raise ValueError("same convolution requires one-dimensional arrays")
    if row_halo.size <= 0 or smoothing.size <= 0:
        raise ValueError("same convolution requires non-empty arrays")
    if selected_mode(policy, int(row_halo.size), int(smoothing.size)) == "direct":
        full = np.convolve(row_halo, smoothing, mode="full")
    else:
        full_length = int(row_halo.size + smoothing.size - 1)
        fft_length = _next_power_of_two(full_length)
        if np.iscomplexobj(row_halo) or np.iscomplexobj(smoothing):
            full = np.fft.ifft(
                np.fft.fft(row_halo, n=fft_length) * np.fft.fft(smoothing, n=fft_length)
            )[:full_length]
        else:
            full = np.fft.irfft(
                np.fft.rfft(row_halo, n=fft_length) * np.fft.rfft(smoothing, n=fft_length),
                n=fft_length,
            )[:full_length]

    # Both paths crop a full convolution at the kernel midpoint, including even kernels.
    start = int(smoothing.size // 2)
    return np.array(full[start : start + row_halo.size], order="C", copy=True)


def kernel_owned_array_payload_upper_bound_bytes(
    signal_length: int,
    support_length: int,
    policy: ConvolutionPolicy,
) -> int:
    """Bound package-created ndarray payload bytes for one active RHS.

    This analytic bound covers one active ``compute_single`` call or one active
    ``iter_multi_rhs`` step. It includes package-visible NumPy output and
    temporary payloads, including the current immutable float32 output payload
    regardless of its bytes backing. The cumulative allocation ledger is at
    least the peak live payload referenced by ndarrays created by this package.

    Excluded exactly are caller input buffers; prior yielded results retained by
    the caller; Python and ndarray object metadata; allocator overhead and
    fragmentation; NumPy internal C, convolution, and FFT workspace; and
    interpreter, import, thread, runtime, and RSS memory. No bound on those
    excluded categories is asserted.

    With ``N`` as signal length, ``K`` as support length, and
    ``F = next_pow2(N + K - 1)``, direct mode uses ``512*N + 128*K`` and FFT
    mode uses ``512*N + 128*K + 256*F``. ``fft_auto`` uses its selected mode.
    """
    if isinstance(signal_length, bool) or not isinstance(signal_length, (int, np.integer)):
        raise TypeError("signal_length must be a positive integer")
    if isinstance(support_length, bool) or not isinstance(support_length, (int, np.integer)):
        raise TypeError("support_length must be a positive integer")
    if not isinstance(policy, ConvolutionPolicy):
        raise TypeError("policy must be a ConvolutionPolicy")
    n = int(signal_length)
    k = int(support_length)
    if n <= 0 or k <= 0:
        raise ValueError("signal_length and support_length must be positive")
    direct_bound = 512 * n + 128 * k
    if selected_mode(policy, n, k) == "direct":
        return direct_bound
    return direct_bound + 256 * _next_power_of_two(n + k - 1)
