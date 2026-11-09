import ast
import hashlib
import math
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from coherence_yanbo.kernels.cwt_cpu import CpuCwtKernel, CpuCwtKernelConfig, CpuCwtKernelError
from coherence_yanbo.kernels.cwt_cpu import kernel as production_kernel
from coherence_yanbo.kernels.cwt_cpu.kernel import KernelFftCache


def independent_next_pow2(value):
    value = max(1, int(value))
    return 1 << (value - 1).bit_length()


def independent_morlet(freq, t_step=0.001, wpara=7):
    sigma = wpara / (2.0 * np.pi * freq)
    amp = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
    support = np.arange(-wpara / (2.0 * freq), wpara / (2.0 * freq) + t_step, t_step)
    return np.ascontiguousarray(
        np.sqrt((2.0 * np.pi * (1.0 / freq)) / t_step)
        * amp
        * np.exp(-(support**2) / (2.0 * sigma**2))
        * np.exp(2.0j * np.pi * freq * support),
        dtype=np.complex128,
    )


def independent_direct(signal, freq, t_step=0.001, wpara=7):
    kernel = independent_morlet(freq, t_step, wpara)
    full = np.convolve(signal, kernel.real, "full") + 1.0j * np.convolve(signal, kernel.imag, "full")
    start = kernel.size // 2
    return np.ascontiguousarray(full[start : start + len(signal)], dtype=np.complex128), kernel.size


def independent_fft(signal, freq, t_step=0.001, wpara=7):
    kernel = independent_morlet(freq, t_step, wpara)
    full_length = len(signal) + len(kernel) - 1
    n_fft = independent_next_pow2(full_length)
    real_full = np.fft.irfft(np.fft.rfft(signal, n=n_fft) * np.fft.rfft(kernel.real, n=n_fft), n=n_fft)[:full_length]
    imag_full = np.fft.irfft(np.fft.rfft(signal, n=n_fft) * np.fft.rfft(kernel.imag, n=n_fft), n=n_fft)[:full_length]
    start = len(kernel) // 2
    return np.ascontiguousarray(real_full[start : start + len(signal)] + 1.0j * imag_full[start : start + len(signal)], dtype=np.complex128), len(kernel)


def dynamic_atol(reference):
    return 2.0e-12 * max(1.0, float(np.max(np.abs(reference))))


@pytest.mark.parametrize("frequency,expected_length", [(1.0, 7001), (8.0, 876), (150.0, 48)])
def test_morlet_support_and_direct_oracle(frequency, expected_length):
    signal = np.ascontiguousarray(np.linspace(-1.0, 1.0, 257, dtype=np.float64))
    actual, support_length = CpuCwtKernel(CpuCwtKernelConfig(mode="np")).transform_single(signal, frequency)
    expected, expected_support = independent_direct(signal, frequency)
    assert support_length == expected_support == expected_length
    assert actual.dtype == np.dtype(np.complex128)
    assert actual.shape == signal.shape and actual.flags.c_contiguous
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("length,frequency", [(257, 150.0), (1025, 8.0)])
def test_fft_matches_independent_frozen_fft_and_direct_bound(length, frequency):
    signal = np.random.default_rng(length).standard_normal(length).astype(np.float64)
    engine = CpuCwtKernel(CpuCwtKernelConfig(mode="fft"))
    actual, support_length = engine.transform_single(signal, frequency)
    expected_fft, fft_support = independent_fft(signal, frequency)
    direct, direct_support = independent_direct(signal, frequency)
    assert support_length == fft_support == direct_support
    np.testing.assert_array_equal(actual, expected_fft)
    assert float(np.max(np.abs(actual - direct))) <= dynamic_atol(direct)
    repeat, _ = engine.transform_single(signal, frequency)
    np.testing.assert_array_equal(actual, repeat)


def test_ola_matches_direct_bound_and_is_not_selected_by_auto():
    signal = np.random.default_rng(91).standard_normal(2049).astype(np.float64)
    direct, support = independent_direct(signal, 8.0)
    ola, actual_support = CpuCwtKernel(CpuCwtKernelConfig(mode="ola", ola_factor=2)).transform_single(signal, 8.0)
    assert actual_support == support
    np.testing.assert_allclose(ola, direct, rtol=0.0, atol=dynamic_atol(direct))
    automatic = CpuCwtKernel(CpuCwtKernelConfig(mode="fft_auto", fft_kernel_threshold=1, fft_signal_threshold=1))
    assert automatic._select_mode(signal.size, support) == "fft"


def test_auto_threshold_and_memory_boundaries():
    kernel = CpuCwtKernel.build_morlet(1.0)
    signal_length = 131072
    n_fft = independent_next_pow2(signal_length + kernel.size - 1)
    required_mb = 24 * n_fft / (1024.0 * 1024.0)
    exact = CpuCwtKernel(CpuCwtKernelConfig(fft_max_temp_mb=required_mb))
    too_small = CpuCwtKernel(CpuCwtKernelConfig(fft_max_temp_mb=np.nextafter(required_mb, 0.0)))
    assert exact._select_mode(signal_length, kernel.size) == "fft"
    assert too_small._select_mode(signal_length, kernel.size) == "np"
    assert exact._select_mode(signal_length - 1, kernel.size) == "np"
    assert exact._select_mode(signal_length, 2047) == "np"


@pytest.mark.parametrize(
    "signal",
    [np.array([], dtype=np.float64), np.array([[1.0]]), np.array([True]), np.array(["x"]), np.array([1 + 2j]), np.array([np.nan])],
)
def test_signal_rejection(signal):
    with pytest.raises(CpuCwtKernelError):
        CpuCwtKernel().transform_single(signal, 8.0)


@pytest.mark.parametrize("value", [True, 0, -1, float("nan"), float("inf"), "8"])
def test_frequency_and_step_rejection(value):
    with pytest.raises(CpuCwtKernelError):
        CpuCwtKernel().transform_single([1, 2], value)
    with pytest.raises(CpuCwtKernelError):
        CpuCwtKernel().transform_single([1, 2], 8.0, t_step=value)


@pytest.mark.parametrize("wpara", [True, 7.0, "7", 0, -1])
def test_wavelet_and_wpara_rejection(wpara):
    with pytest.raises(CpuCwtKernelError):
        CpuCwtKernel().transform_single([1, 2], 8.0, wpara=wpara)
    with pytest.raises(CpuCwtKernelError):
        CpuCwtKernel().transform_single([1, 2], 8.0, wavelet="mexican_hat")


@pytest.mark.parametrize("kwargs", [{"fft_kernel_threshold": True}, {"fft_signal_threshold": 1.5}, {"ola_factor": 0}, {"cache_cap_bytes": -1}, {"fft_max_temp_mb": float("inf")}])
def test_config_rejection(kwargs):
    with pytest.raises(CpuCwtKernelError):
        CpuCwtKernelConfig(**kwargs)


@pytest.mark.parametrize(
    "signal",
    [
        np.array([-3, 0, 5], dtype=np.int16),
        np.array([0, 7, 13], dtype=np.uint16),
        np.array([-0.5, 0.25, 1.75], dtype=np.float32),
        np.array([-2.0, 1.0, 4.0], dtype=np.float64),
    ],
)
def test_finite_real_signal_normalization_matches_float64_direct_oracle(signal):
    original = signal.copy()
    expected, support_length = independent_direct(np.ascontiguousarray(signal, dtype=np.float64), 150.0)
    actual, actual_support = CpuCwtKernel(CpuCwtKernelConfig(mode="np")).transform_single(signal, 150.0)
    assert actual_support == support_length
    assert actual.dtype == np.dtype(np.complex128) and actual.flags.c_contiguous
    assert actual.shape == signal.shape
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(signal, original)


def test_cache_exact_cap_and_one_byte_under_nonpersistence():
    kernel = np.array([1 + 2j, 3 - 4j], dtype=np.complex128)
    n_fft = 32
    item_bytes = 2 * np.fft.rfft(kernel.real, n=n_fft).nbytes
    admitted = KernelFftCache(item_bytes)
    admitted.spectra(kernel, n_fft)
    assert admitted._bytes == item_bytes and len(admitted._items) == 1
    rejected = KernelFftCache(item_bytes - 1)
    rejected.spectra(kernel, n_fft)
    assert rejected._bytes == 0 and rejected._items == {}


def test_cache_lru_order_and_key():
    n_fft = 32
    item_bytes = 2 * np.fft.rfft(np.array([1.0]), n=n_fft).nbytes
    cache = KernelFftCache(2 * item_bytes)
    kernels = [np.array([i + 1j], dtype=np.complex128) for i in range(3)]
    cache.spectra(kernels[0], n_fft)
    cache.spectra(kernels[1], n_fft)
    cache.spectra(kernels[0], n_fft)
    cache.spectra(kernels[2], n_fft)
    keys = [cache._key(kernel, n_fft) for kernel in kernels]
    assert list(cache._items) == [keys[0], keys[2]]
    assert cache._bytes == 2 * item_bytes
    assert keys[0][0] == hashlib.sha256(kernels[0].tobytes(order="C")).hexdigest()
    assert keys[0][1:] == ("<c16", 1, n_fft)


def test_cache_zero_and_same_key_single_flight(monkeypatch):
    kernel = np.array([1 + 2j, 3 - 4j], dtype=np.complex128)
    cache = KernelFftCache(0)
    original = np.fft.rfft
    entered = threading.Event()
    release = threading.Event()
    calls = 0
    lock = threading.Lock()

    def delayed(values, *args, **kwargs):
        nonlocal calls
        with lock:
            calls += 1
            first = calls == 1
        if first:
            entered.set()
            assert release.wait(5.0)
        return original(values, *args, **kwargs)

    monkeypatch.setattr(np.fft, "rfft", delayed)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(cache.spectra, kernel, 32)
        assert entered.wait(5.0)
        second = executor.submit(cache.spectra, kernel, 32)
        release.set()
        one, two = first.result(5.0), second.result(5.0)
    assert calls == 2 and cache._items == {} and cache._flights == {}
    np.testing.assert_array_equal(one[0], two[0])


def test_cache_allows_unrelated_key_producers_to_overlap(monkeypatch):
    cache = KernelFftCache(1024 * 1024)
    one = np.array([1 + 2j], dtype=np.complex128)
    two = np.array([-3 + 4j], dtype=np.complex128)
    original = np.fft.rfft
    barrier = threading.Barrier(2)

    def overlapping(values, *args, **kwargs):
        if float(values[0]) in {1.0, -3.0}:
            barrier.wait(5.0)
        return original(values, *args, **kwargs)

    monkeypatch.setattr(np.fft, "rfft", overlapping)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(cache.spectra, item, 32) for item in (one, two)]
        [future.result(5.0) for future in futures]
    assert len(cache._items) == 2


class _WaiterGate:
    """Test-only event that confirms all waiters joined one live flight."""

    def __init__(self, expected_waiters):
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._expected_waiters = expected_waiters
        self._waiters = 0
        self.waiters_ready = threading.Event()

    def wait(self, timeout=None):
        with self._lock:
            self._waiters += 1
            if self._waiters == self._expected_waiters:
                self.waiters_ready.set()
        assert self._event.wait(timeout)
        return True

    def set(self):
        self._event.set()


def _run_failure_fanout(monkeypatch, source_error):
    cache = KernelFftCache(1024 * 1024)
    kernel = np.array([1 + 2j], dtype=np.complex128)
    original = np.fft.rfft
    entered, release = threading.Event(), threading.Event()
    call_lock = threading.Lock()
    calls = 0
    gate = _WaiterGate(expected_waiters=3)

    class TestFlight:
        def __init__(self):
            self.event = gate
            self.result = None
            self.failure = None
            self.failed = False

    monkeypatch.setattr(production_kernel, "_Flight", TestFlight)

    def failing(values, *args, **kwargs):
        nonlocal calls
        with call_lock:
            calls += 1
            first = calls == 1
        if first:
            entered.set()
            assert release.wait(5.0)
            raise source_error
        return original(values, *args, **kwargs)

    monkeypatch.setattr(np.fft, "rfft", failing)
    with ThreadPoolExecutor(max_workers=4) as executor:
        producer = executor.submit(cache.spectra, kernel, 32)
        assert entered.wait(5.0)
        waiters = [executor.submit(cache.spectra, kernel, 32) for _ in range(3)]
        assert gate.waiters_ready.wait(5.0)
        release.set()
        errors = []
        for future in [producer, *waiters]:
            with pytest.raises(BaseException) as captured:
                future.result(5.0)
            errors.append(captured.value)
    assert cache._flights == {}
    cache.spectra(kernel, 32)
    assert len(cache._items) == 1
    return errors


def test_cache_safe_failure_fanout_clear_and_retry(monkeypatch):
    source_error = ValueError("planned failure")
    producer, *waiters = _run_failure_fanout(monkeypatch, source_error)
    assert producer is source_error
    assert len(waiters) == 3
    assert all(type(error) is ValueError and error.args == source_error.args for error in waiters)
    assert all(error is not producer for error in waiters)
    assert len({id(error) for error in waiters}) == 3


def test_cache_unsafe_failure_fanout_fails_closed_and_retries(monkeypatch):
    class UnsafeFailure(Exception):
        pass

    source_error = UnsafeFailure("planned unsafe failure")
    producer, *waiters = _run_failure_fanout(monkeypatch, source_error)
    assert producer is source_error
    assert len(waiters) == 3
    assert all(type(error) is CpuCwtKernelError for error in waiters)
    assert all(error.args == ("kernel FFT single-flight producer failed",) for error in waiters)
    assert len({id(error) for error in waiters}) == 3


@pytest.mark.parametrize("mode", ["np", "fft", "ola"])
def test_114_balanced_segment_reassembly_matches_independent_oracles(mode):
    samples = 1369  # Deliberately not divisible by 114.
    signal = np.random.default_rng(114).standard_normal(samples).astype(np.float64)
    boundaries = [(i * samples // 114, (i + 1) * samples // 114) for i in range(114)]
    assert all(start < stop for start, stop in boundaries)
    assert boundaries[0][0] == 0 and boundaries[-1][1] == samples
    assert sum(stop - start for start, stop in boundaries) == samples
    assert len({stop - start for start, stop in boundaries}) >= 2
    frequencies = (150.0, 8.0, 10.0)
    independent = np.empty((len(frequencies), samples), dtype=np.complex128)
    candidate = np.empty_like(independent)
    engine = CpuCwtKernel(CpuCwtKernelConfig(mode=mode, ola_factor=2))
    for row, frequency in enumerate(frequencies):
        _, kernel_length = independent_direct(signal, frequency)
        halo = kernel_length // 2
        for start, stop in boundaries:
            lo, hi = max(0, start - halo), min(samples, stop + halo)
            direct_segment, direct_support = independent_direct(signal[lo:hi], frequency)
            assert direct_support == kernel_length
            if mode == "fft":
                expected_segment, expected_support = independent_fft(signal[lo:hi], frequency)
                assert expected_support == kernel_length
            else:
                expected_segment = direct_segment
            independent[row, start:stop] = expected_segment[start - lo : stop - lo]
            segment, segment_support = engine.transform_single(signal[lo:hi], frequency)
            assert segment_support == kernel_length
            if mode == "fft":
                np.testing.assert_array_equal(segment, expected_segment)
                assert float(np.max(np.abs(segment - direct_segment))) <= dynamic_atol(direct_segment)
            elif mode == "ola":
                np.testing.assert_allclose(segment, direct_segment, rtol=0.0, atol=dynamic_atol(direct_segment))
            else:
                np.testing.assert_array_equal(segment, direct_segment)
            candidate[row, start:stop] = segment[start - lo : stop - lo]
    if mode == "ola":
        np.testing.assert_allclose(candidate, independent, rtol=0.0, atol=dynamic_atol(independent))
    else:
        np.testing.assert_array_equal(candidate, independent)


def test_production_import_surface_is_self_contained():
    package_root = Path(__file__).resolve().parents[3] / "src" / "coherence_yanbo" / "kernels" / "cwt_cpu"
    init_tree = ast.parse((package_root / "__init__.py").read_text(encoding="utf-8"))
    kernel_tree = ast.parse((package_root / "kernel.py").read_text(encoding="utf-8"))
    kernel_imports = [
        (type(node).__name__, getattr(node, "module", None), getattr(node, "level", None), tuple((alias.name, alias.asname) for alias in node.names))
        for node in ast.walk(kernel_tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    assert kernel_imports == [
        ("ImportFrom", "__future__", 0, (("annotations", None),)),
        ("Import", None, None, (("hashlib", None),)),
        ("Import", None, None, (("math", None),)),
        ("Import", None, None, (("numbers", None),)),
        ("Import", None, None, (("threading", None),)),
        ("ImportFrom", "collections", 0, (("OrderedDict", None),)),
        ("ImportFrom", "dataclasses", 0, (("dataclass", None),)),
        ("Import", None, None, (("numpy", "np"),)),
    ]
    init_imports = [
        (type(node).__name__, node.module, node.level, tuple((alias.name, alias.asname) for alias in node.names))
        for node in ast.walk(init_tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    assert init_imports == [
        ("ImportFrom", "kernel", 1, (("CpuCwtKernel", None), ("CpuCwtKernelConfig", None), ("CpuCwtKernelError", None)))
    ]
