"""Self-contained CPU implementation of the frozen Morlet CWT primitive."""

from __future__ import annotations

import hashlib
import math
import numbers
import threading
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np


_SAFE_EXCEPTION_TYPES = (ValueError, TypeError, RuntimeError, FloatingPointError, OverflowError)
_SAFE_EXCEPTION_ARGS = (type(None), bool, int, float, str, bytes)


class CpuCwtKernelError(ValueError):
    """Raised when a request violates the standalone CPU-kernel contract."""


def _require_positive_real(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise CpuCwtKernelError(f"{name} must be a finite positive real scalar")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise CpuCwtKernelError(f"{name} must be a finite positive real scalar")
    return result


def _require_integral(value: object, name: str, *, positive: bool) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise CpuCwtKernelError(f"{name} must be an integral scalar")
    result = int(value)
    if (positive and result <= 0) or (not positive and result < 0):
        qualifier = "positive" if positive else "nonnegative"
        raise CpuCwtKernelError(f"{name} must be {qualifier}")
    return result


def _normalize_signal(signal: object) -> np.ndarray:
    values = np.asarray(signal)
    if values.ndim != 1 or values.size == 0:
        raise CpuCwtKernelError("signal must be a non-empty one-dimensional array")
    if values.dtype.kind not in {"i", "u", "f"}:
        raise CpuCwtKernelError("signal must contain signed/unsigned integers or floats")
    normalized = np.ascontiguousarray(values, dtype=np.float64)
    if not bool(np.all(np.isfinite(normalized))):
        raise CpuCwtKernelError("signal must contain only finite values")
    return normalized


def _next_pow2(value: int) -> int:
    return 1 << (max(1, int(value)) - 1).bit_length()


def _same_crop(full: np.ndarray, signal_length: int, kernel_length: int) -> np.ndarray:
    start = kernel_length // 2
    return full[start : start + signal_length]


def _freeze_failure(exc: BaseException) -> tuple[type[BaseException], tuple[object, ...]] | None:
    if type(exc) not in _SAFE_EXCEPTION_TYPES or any(type(arg) not in _SAFE_EXCEPTION_ARGS for arg in exc.args):
        return None
    return type(exc), tuple(exc.args)


def _rebuild_failure(frozen: tuple[type[BaseException], tuple[object, ...]] | None) -> BaseException:
    if frozen is None:
        return CpuCwtKernelError("kernel FFT single-flight producer failed")
    exception_type, args = frozen
    try:
        rebuilt = exception_type(*args)
    except BaseException:
        return CpuCwtKernelError("kernel FFT single-flight producer failed")
    if type(rebuilt) is not exception_type or rebuilt.args != args:
        return CpuCwtKernelError("kernel FFT single-flight producer failed")
    return rebuilt


class _Flight:
    def __init__(self) -> None:
        self.event = threading.Event()
        self.result: tuple[np.ndarray, np.ndarray] | None = None
        self.failure: tuple[type[BaseException], tuple[object, ...]] | None = None
        self.failed = False


class KernelFftCache:
    """Per-engine, byte-capped LRU cache with independent per-key flights."""

    def __init__(self, cap_bytes: int) -> None:
        self._cap_bytes = cap_bytes
        self._items: OrderedDict[tuple[str, str, int, int], tuple[np.ndarray, np.ndarray, int]] = OrderedDict()
        self._bytes = 0
        self._flights: dict[tuple[str, str, int, int], _Flight] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _key(kernel: np.ndarray, n_fft: int) -> tuple[str, str, int, int]:
        canonical = np.ascontiguousarray(kernel, dtype=np.complex128)
        return (
            hashlib.sha256(canonical.tobytes(order="C")).hexdigest(),
            canonical.dtype.str,
            int(canonical.size),
            int(n_fft),
        )

    def spectra(self, kernel: np.ndarray, n_fft: int) -> tuple[np.ndarray, np.ndarray]:
        canonical = np.ascontiguousarray(kernel, dtype=np.complex128)
        key = self._key(canonical, n_fft)
        with self._lock:
            existing = self._items.pop(key, None)
            if existing is not None:
                self._items[key] = existing
                return existing[0], existing[1]
            flight = self._flights.get(key)
            if flight is None:
                flight = _Flight()
                self._flights[key] = flight
                producer = True
            else:
                producer = False

        if not producer:
            flight.event.wait()
            if flight.failed:
                raise _rebuild_failure(flight.failure)
            if flight.result is None:
                raise CpuCwtKernelError("kernel FFT single-flight completed without a result")
            return flight.result

        try:
            real_fft = np.fft.rfft(canonical.real, n=n_fft)
            imag_fft = np.fft.rfft(canonical.imag, n=n_fft)
            item_bytes = int(real_fft.nbytes + imag_fft.nbytes)
            with self._lock:
                existing = self._items.pop(key, None)
                if existing is not None:
                    self._items[key] = existing
                    result = (existing[0], existing[1])
                elif self._cap_bytes <= 0 or item_bytes > self._cap_bytes:
                    result = (real_fft, imag_fft)
                else:
                    while self._items and self._bytes + item_bytes > self._cap_bytes:
                        _, (_, _, evicted_bytes) = self._items.popitem(last=False)
                        self._bytes -= evicted_bytes
                    self._items[key] = (real_fft, imag_fft, item_bytes)
                    self._bytes += item_bytes
                    result = (real_fft, imag_fft)
                flight.result = result
                self._flights.pop(key, None)
                flight.event.set()
            return result
        except BaseException as exc:
            frozen = _freeze_failure(exc)
            with self._lock:
                flight.failure = frozen
                flight.failed = True
                self._flights.pop(key, None)
                flight.event.set()
            raise


@dataclass(frozen=True)
class CpuCwtKernelConfig:
    mode: str = "fft_auto"
    fft_kernel_threshold: int = 2048
    fft_signal_threshold: int = 131072
    fft_max_temp_mb: float = 1024.0
    ola_factor: int = 2
    cache_cap_bytes: int = 256 * 1024 * 1024

    def __post_init__(self) -> None:
        if self.mode not in {"np", "fft", "fft_fused", "fft_auto", "ola"}:
            raise CpuCwtKernelError(f"unsupported convolution mode: {self.mode!r}")
        object.__setattr__(self, "fft_kernel_threshold", _require_integral(self.fft_kernel_threshold, "fft_kernel_threshold", positive=True))
        object.__setattr__(self, "fft_signal_threshold", _require_integral(self.fft_signal_threshold, "fft_signal_threshold", positive=True))
        object.__setattr__(self, "ola_factor", _require_integral(self.ola_factor, "ola_factor", positive=True))
        object.__setattr__(self, "cache_cap_bytes", _require_integral(self.cache_cap_bytes, "cache_cap_bytes", positive=False))
        object.__setattr__(self, "fft_max_temp_mb", _require_positive_real(self.fft_max_temp_mb, "fft_max_temp_mb"))


class CpuCwtKernel:
    """A standalone Morlet transform engine with a private FFT spectrum cache."""

    def __init__(self, config: CpuCwtKernelConfig | None = None) -> None:
        self.config = CpuCwtKernelConfig() if config is None else config
        if not isinstance(self.config, CpuCwtKernelConfig):
            raise CpuCwtKernelError("config must be a CpuCwtKernelConfig")
        self._cache = KernelFftCache(self.config.cache_cap_bytes)

    @staticmethod
    def build_morlet(curr_freq: object, t_step: object = 0.001, wavelet: object = "morlet", wpara: object = 7) -> np.ndarray:
        if not isinstance(wavelet, str) or wavelet.lower() != "morlet":
            raise CpuCwtKernelError("wavelet must be morlet")
        frequency = _require_positive_real(curr_freq, "curr_freq")
        step = _require_positive_real(t_step, "t_step")
        width = _require_integral(wpara, "wpara", positive=True)
        sigma = width / (2.0 * np.pi * frequency)
        amp = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
        support = np.arange(-width / (2.0 * frequency), width / (2.0 * frequency) + step, step)
        kernel = (
            np.sqrt((2.0 * np.pi * (1.0 / frequency)) / step)
            * amp
            * np.exp(-(support**2) / (2.0 * sigma**2))
            * np.exp(2.0j * np.pi * frequency * support)
        )
        return np.ascontiguousarray(kernel, dtype=np.complex128)

    def _auto_mode(self, signal_length: int, kernel_length: int) -> str:
        n_fft = _next_pow2(signal_length + kernel_length - 1)
        fits_memory = 24 * n_fft <= self.config.fft_max_temp_mb * 1024.0 * 1024.0
        if kernel_length >= self.config.fft_kernel_threshold and signal_length >= self.config.fft_signal_threshold and fits_memory:
            return "fft"
        return "np"

    def _select_mode(self, signal_length: int, kernel_length: int) -> str:
        if self.config.mode == "fft_auto":
            return self._auto_mode(signal_length, kernel_length)
        if self.config.mode == "fft_fused":
            # Preserve frozen fft_auto dispatch; fuse only the branch it selects.
            return "fft_fused" if self._auto_mode(signal_length, kernel_length) == "fft" else "np"
        return self.config.mode

    def _direct(self, signal: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        real_full = np.convolve(signal, kernel.real, mode="full")
        imag_full = np.convolve(signal, kernel.imag, mode="full")
        return _same_crop(real_full, signal.size, kernel.size) + 1.0j * _same_crop(imag_full, signal.size, kernel.size)

    def _fft(self, signal: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        n_full = signal.size + kernel.size - 1
        n_fft = _next_pow2(n_full)
        real_kernel_fft, imag_kernel_fft = self._cache.spectra(kernel, n_fft)
        # Separate signal transforms deliberately retain the frozen two-real-path algebra.
        real_full = np.fft.irfft(np.fft.rfft(signal, n=n_fft) * real_kernel_fft, n=n_fft)[:n_full]
        imag_full = np.fft.irfft(np.fft.rfft(signal, n=n_fft) * imag_kernel_fft, n=n_fft)[:n_full]
        return _same_crop(real_full, signal.size, kernel.size) + 1.0j * _same_crop(imag_full, signal.size, kernel.size)

    def _fft_fused(self, signal: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Reuse one signal spectrum, trading fewer FFTs for a longer-lived temporary."""
        n_full = signal.size + kernel.size - 1
        n_fft = _next_pow2(n_full)
        real_kernel_fft, imag_kernel_fft = self._cache.spectra(kernel, n_fft)
        signal_fft = np.fft.rfft(signal, n=n_fft)
        real_full = np.fft.irfft(signal_fft * real_kernel_fft, n=n_fft)[:n_full]
        imag_full = np.fft.irfft(signal_fft * imag_kernel_fft, n=n_fft)[:n_full]
        return _same_crop(real_full, signal.size, kernel.size) + 1.0j * _same_crop(imag_full, signal.size, kernel.size)

    def _ola(self, signal: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        kernel_length = kernel.size
        block_length = max(1, self.config.ola_factor * kernel_length)
        n_fft = _next_pow2(block_length + kernel_length - 1)
        block_length = max(1, n_fft - kernel_length + 1)
        real_kernel_fft, imag_kernel_fft = self._cache.spectra(kernel, n_fft)
        n_full = signal.size + kernel_length - 1
        real_full = np.zeros(n_full, dtype=np.float64)
        imag_full = np.zeros(n_full, dtype=np.float64)
        for start in range(0, signal.size, block_length):
            stop = min(signal.size, start + block_length)
            width = stop - start + kernel_length - 1
            block = signal[start:stop]
            real_full[start : start + width] += np.fft.irfft(np.fft.rfft(block, n=n_fft) * real_kernel_fft, n=n_fft)[:width]
            imag_full[start : start + width] += np.fft.irfft(np.fft.rfft(block, n=n_fft) * imag_kernel_fft, n=n_fft)[:width]
        return _same_crop(real_full, signal.size, kernel_length) + 1.0j * _same_crop(imag_full, signal.size, kernel_length)

    def transform_single(
        self,
        signal: object,
        curr_freq: object,
        *,
        t_step: object = 0.001,
        wavelet: object = "morlet",
        wpara: object = 7,
    ) -> tuple[np.ndarray, int]:
        values = _normalize_signal(signal)
        kernel = self.build_morlet(curr_freq, t_step=t_step, wavelet=wavelet, wpara=wpara)
        mode = self._select_mode(values.size, kernel.size)
        if mode == "np":
            transformed = self._direct(values, kernel)
        elif mode == "fft":
            transformed = self._fft(values, kernel)
        elif mode == "fft_fused":
            transformed = self._fft_fused(values, kernel)
        else:
            transformed = self._ola(values, kernel)
        result = np.ascontiguousarray(transformed, dtype=np.complex128)
        if not bool(np.all(np.isfinite(result))):
            raise CpuCwtKernelError(f"{mode} produced non-finite output")
        return result, int(kernel.size)
