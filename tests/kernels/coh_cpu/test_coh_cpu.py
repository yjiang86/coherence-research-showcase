from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from coherence_yanbo.kernels.coh_cpu import (
    ConvolutionPolicy,
    compute_single,
    iter_multi_rhs,
    kernel_owned_array_payload_upper_bound_bytes,
)
from coherence_yanbo.kernels.coh_cpu.kernel import selected_mode

from .reference import direct_reference


def _policy(mode: str = "direct", **changes: object) -> ConvolutionPolicy:
    values: dict[str, object] = {
        "mode": mode,
        "t_step": 0.2,
        "wpara": 1.0,
        "fft_kernel_threshold": 1,
        "fft_signal_threshold": 1,
        "fft_max_temp_bytes": 1 << 30,
    }
    values.update(changes)
    return ConvolutionPolicy(**values)


@pytest.mark.parametrize("t_step, expected_parity", [(0.4, 1), (0.25, 0)])
@pytest.mark.parametrize("length", [1, 3])
def test_odd_even_support_and_short_inputs(t_step: float, expected_parity: int, length: int) -> None:
    lhs = np.linspace(1.0, 2.0, length).astype(np.complex128)
    rhs = np.linspace(2.0, 1.0, length).astype(np.complex128)
    policy = _policy("direct", t_step=t_step)

    result = compute_single(lhs, rhs, 1.0, policy)
    expected, support_length = direct_reference(lhs, rhs, 1.0, policy.t_step, policy.wpara)

    assert result.support_length == support_length
    assert result.support_length % 2 == expected_parity
    assert result.values.shape == (length,)
    np.testing.assert_allclose(result.values, expected, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_complex_input_types_match_direct_oracle(dtype: np.dtype) -> None:
    rng = np.random.default_rng(12)
    lhs = (rng.normal(size=17) + 1j * rng.normal(size=17)).astype(dtype)
    rhs = (rng.normal(size=17) + 1j * rng.normal(size=17)).astype(dtype)
    policy = _policy("direct")

    result = compute_single(lhs, rhs, 2.5, policy)
    expected, _ = direct_reference(lhs, rhs, 2.5, policy.t_step, policy.wpara)

    assert result.values.dtype == np.float32
    np.testing.assert_allclose(result.values, expected, atol=1e-6, rtol=1e-5)


def test_compute_single_output_is_frozen_contiguous_and_non_aliasing() -> None:
    lhs = np.arange(8, dtype=np.float64).astype(np.complex128)
    rhs = np.roll(lhs, 2)

    result = compute_single(lhs, rhs, 2.0, _policy("fft"))

    assert result.values.dtype == np.float32
    assert result.values.flags.c_contiguous
    assert not result.values.flags.writeable
    assert isinstance(result.values.base, bytes)
    assert not np.shares_memory(result.values, lhs)
    assert not np.shares_memory(result.values, rhs)
    with pytest.raises(ValueError):
        result.values.setflags(write=True)
    with pytest.raises(ValueError):
        result.values[0] = 0.5


def test_zero_and_finite_subnormal_inputs_are_valid() -> None:
    policy = _policy("direct")
    zeros = np.zeros(6, dtype=np.complex64)
    zero_result = compute_single(zeros, zeros, 2.0, policy)
    np.testing.assert_array_equal(zero_result.values, np.zeros(6, dtype=np.float32))

    tiny = np.nextafter(np.float64(0.0), np.float64(1.0))
    lhs = np.full(6, tiny + 1j * tiny, dtype=np.complex128)
    rhs = np.full(6, tiny - 1j * tiny, dtype=np.complex128)
    result = compute_single(lhs, rhs, 2.0, policy)
    expected, _ = direct_reference(lhs, rhs, 2.0, policy.t_step, policy.wpara)
    np.testing.assert_allclose(result.values, expected, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_nonfinite_inputs_are_rejected_before_compute(invalid: float) -> None:
    valid = np.ones(5, dtype=np.complex128)
    invalid_lhs = valid.copy()
    invalid_lhs[2] = invalid + 0j
    with pytest.raises(ValueError, match="finite"):
        compute_single(invalid_lhs, valid, 2.0, _policy())
    invalid_rhs = valid.copy()
    invalid_rhs[2] = invalid + 0j
    with pytest.raises(ValueError, match="finite"):
        compute_single(valid, invalid_rhs, 2.0, _policy())


def test_finite_intermediate_overflow_uses_final_nonfinite_zero_rule() -> None:
    lhs = np.full(5, np.complex128(1e308 + 0j))
    rhs = np.full(5, np.complex128(1e308 + 0j))
    direct_policy = _policy("direct")
    fft_policy = _policy("fft")
    expected, _ = direct_reference(lhs, rhs, 2.0, direct_policy.t_step, direct_policy.wpara)

    direct = compute_single(lhs, rhs, 2.0, direct_policy).values
    fft = compute_single(lhs, rhs, 2.0, fft_policy).values

    np.testing.assert_array_equal(expected, np.zeros(5, dtype=np.float32))
    np.testing.assert_array_equal(direct, expected)
    np.testing.assert_array_equal(fft, expected)


def test_direct_fft_auto_parity_and_legacy_conformance() -> None:
    rng = np.random.default_rng(19)
    lhs = (rng.normal(size=31) + 1j * rng.normal(size=31)).astype(np.complex128)
    rhs = (rng.normal(size=31) + 1j * rng.normal(size=31)).astype(np.complex128)
    direct_policy = _policy("direct")
    fft_policy = _policy("fft")
    auto_policy = _policy("fft_auto")
    expected, support_length = direct_reference(lhs, rhs, 2.0, direct_policy.t_step, direct_policy.wpara)

    results = [
        compute_single(lhs, rhs, 2.0, policy)
        for policy in (direct_policy, fft_policy, auto_policy)
    ]
    for result in results:
        assert result.values.shape == (31,)
        assert result.values.dtype == np.float32
        assert result.support_length == support_length
        np.testing.assert_allclose(result.values, expected, atol=1e-6, rtol=1e-5)
    np.testing.assert_allclose(results[0].values, results[1].values, atol=1e-6, rtol=1e-5)
    np.testing.assert_allclose(results[1].values, results[2].values, atol=1e-6, rtol=1e-5)


def test_multi_rhs_matches_independent_single_calls_and_preserves_order() -> None:
    rng = np.random.default_rng(23)
    lhs = (rng.normal(size=13) + 1j * rng.normal(size=13)).astype(np.complex64)
    rhs_values = [
        (rng.normal(size=13) + 1j * rng.normal(size=13)).astype(np.complex64)
        for _ in range(8)
    ]
    policy = _policy("fft")

    one_rhs = list(iter_multi_rhs(lhs, iter(rhs_values[:1]), 2.0, policy))
    multi = list(iter_multi_rhs(lhs, iter(rhs_values), 2.0, policy))
    singles = [compute_single(lhs, rhs, 2.0, policy) for rhs in rhs_values]
    permuted_indices = [7, 3, 0, 6, 1, 5, 2, 4]
    permuted = list(iter_multi_rhs(lhs, (rhs_values[i] for i in permuted_indices), 2.0, policy))

    one_rhs_bound = kernel_owned_array_payload_upper_bound_bytes(
        lhs.size,
        one_rhs[0].support_length,
        policy,
    )
    eight_rhs_bound = kernel_owned_array_payload_upper_bound_bytes(
        lhs.size,
        multi[0].support_length,
        policy,
    )
    assert one_rhs_bound == eight_rhs_bound
    assert one_rhs[0].values.nbytes <= one_rhs_bound
    np.testing.assert_allclose(one_rhs[0].values, singles[0].values, atol=1e-6, rtol=1e-5)
    assert len(multi) == 8
    for actual, expected in zip(multi, singles):
        assert actual.support_length == expected.support_length
        assert actual.values.nbytes <= eight_rhs_bound
        assert actual.values.dtype == np.float32
        assert actual.values.flags.c_contiguous
        assert not actual.values.flags.writeable
        assert isinstance(actual.values.base, bytes)
        assert not np.shares_memory(actual.values, lhs)
        for rhs in rhs_values:
            assert not np.shares_memory(actual.values, rhs)
        with pytest.raises(ValueError):
            actual.values.setflags(write=True)
        with pytest.raises(ValueError):
            actual.values[0] = 0.5
        np.testing.assert_allclose(actual.values, expected.values, atol=1e-6, rtol=1e-5)
    assert len({id(result.values.base) for result in multi}) == len(multi)
    for actual, index in zip(permuted, permuted_indices):
        np.testing.assert_allclose(actual.values, singles[index].values, atol=1e-6, rtol=1e-5)


def test_multi_rhs_is_lazy_and_yields_before_second_conversion() -> None:
    lhs = np.arange(6, dtype=np.float64).astype(np.complex128)
    first_rhs = lhs[::-1].copy()
    second_rhs = np.ones(6, dtype=np.complex128)

    class GuardedRhs:
        def __init__(self) -> None:
            self.requests = 0
            self.first_yield_observed = False

        def __iter__(self) -> "GuardedRhs":
            return self

        def __next__(self) -> np.ndarray:
            self.requests += 1
            if self.requests == 1:
                return first_rhs
            if self.requests == 2:
                if not self.first_yield_observed:
                    raise AssertionError("second RHS was requested before the first result yielded")
                return second_rhs
            raise StopIteration

    guarded = GuardedRhs()
    iterator = iter_multi_rhs(lhs, guarded, 2.0, _policy())
    first = next(iterator)
    assert guarded.requests == 1
    guarded.first_yield_observed = True
    second = next(iterator)
    assert guarded.requests == 2
    assert first.values.shape == second.values.shape == (6,)


def test_multi_rhs_uses_lhs_snapshot_after_first_yield() -> None:
    lhs = np.arange(8, dtype=np.float64).astype(np.complex128)
    lhs_before_mutation = np.array(lhs, dtype=np.complex128, order="C", copy=True)
    first_rhs = np.roll(lhs_before_mutation, 1)
    second_rhs = np.roll(lhs_before_mutation, 3)
    policy = _policy("fft")

    iterator = iter_multi_rhs(lhs, (first_rhs, second_rhs), 2.0, policy)
    first = next(iterator)
    lhs[:] = np.complex128(-100.0 + 50.0j)
    second = next(iterator)
    expected = compute_single(lhs_before_mutation, second_rhs, 2.0, policy)

    np.testing.assert_allclose(second.values, expected.values, atol=1e-6, rtol=1e-5)
    for result in (first, second):
        assert not np.shares_memory(result.values, lhs)
        assert not np.shares_memory(result.values, lhs_before_mutation)


def test_outputs_are_frozen_distinct_and_do_not_share_memory() -> None:
    lhs = np.arange(8, dtype=np.float64).astype(np.complex128)
    rhs_values = [np.roll(lhs, offset) for offset in (1, 2)]
    outputs = [result.values for result in iter_multi_rhs(lhs, rhs_values, 2.0, _policy("fft"))]

    for values in outputs:
        assert values.flags.c_contiguous
        assert not values.flags.writeable
        assert isinstance(values.base, bytes)
        assert not np.shares_memory(values, lhs)
        for rhs in rhs_values:
            assert not np.shares_memory(values, rhs)
        with pytest.raises(ValueError):
            values.setflags(write=True)
        with pytest.raises(ValueError):
            values[0] = 0.5
    assert outputs[0] is not outputs[1]
    assert outputs[0].base is not outputs[1].base
    assert not np.shares_memory(outputs[0], outputs[1])
    with pytest.raises(FrozenInstanceError):
        compute_single(lhs, rhs_values[0], 2.0, _policy()).support_length = 99


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ConvolutionPolicy("unknown", 0.2, 1.0),
        lambda: ConvolutionPolicy("direct", 0.2, 1.0, fft_kernel_threshold=0),
        lambda: ConvolutionPolicy("direct", 0.2, 1.0, fft_signal_threshold=False),
        lambda: ConvolutionPolicy("direct", 0.2, 1.0, fft_max_temp_bytes=0),
        lambda: ConvolutionPolicy(mode="direct", t_step=1, wpara=1.0),
        lambda: ConvolutionPolicy(mode="direct", t_step=0.0, wpara=1.0),
        lambda: ConvolutionPolicy(mode="direct", t_step=0.2, wpara=0.0),
        lambda: ConvolutionPolicy("direct", 0.2, 1.0, wpara=np.inf),
    ],
)
def test_invalid_policy_values_are_rejected(factory: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        factory()


@pytest.mark.parametrize(
    "lhs, rhs, frequency",
    [
        ([1 + 0j], np.ones(1, dtype=np.complex128), 2.0),
        (np.ones((2, 2), dtype=np.complex128), np.ones(4, dtype=np.complex128), 2.0),
        (np.ones(2, dtype=np.float64), np.ones(2, dtype=np.complex128), 2.0),
        (np.ones(2, dtype=np.complex128), np.ones(3, dtype=np.complex128), 2.0),
        (np.ones(2, dtype=np.complex128), np.ones(2, dtype=np.complex128), 2),
        (np.ones(2, dtype=np.complex128), np.ones(2, dtype=np.complex128), np.nan),
    ],
)
def test_invalid_shape_type_and_frequency_are_rejected(lhs: object, rhs: object, frequency: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        compute_single(lhs, rhs, frequency, _policy())


def test_kernel_owned_payload_bound_exact_modes() -> None:
    signal_length = 13
    support_length = 5
    fft_length = 32
    direct_expected = 512 * signal_length + 128 * support_length
    fft_expected = direct_expected + 256 * fft_length

    direct = kernel_owned_array_payload_upper_bound_bytes(
        signal_length,
        support_length,
        _policy("direct"),
    )
    fft = kernel_owned_array_payload_upper_bound_bytes(
        signal_length,
        support_length,
        _policy("fft"),
    )
    auto_direct = kernel_owned_array_payload_upper_bound_bytes(
        signal_length,
        support_length,
        _policy("fft_auto", fft_kernel_threshold=support_length + 1),
    )
    auto_fft = kernel_owned_array_payload_upper_bound_bytes(
        signal_length,
        support_length,
        _policy("fft_auto"),
    )

    assert direct == direct_expected
    assert fft == fft_expected
    assert auto_direct == direct_expected
    assert auto_fft == fft_expected


@pytest.mark.parametrize(
    "signal_length, support_length, expected_error",
    [
        (0, 5, ValueError),
        (-1, 5, ValueError),
        (5, 0, ValueError),
        (5, -1, ValueError),
        (1.0, 5, TypeError),
        (5, 1.0, TypeError),
        (True, 5, TypeError),
        (5, False, TypeError),
    ],
)
def test_kernel_owned_payload_bound_rejects_invalid_lengths(
    signal_length: object,
    support_length: object,
    expected_error: type[Exception],
) -> None:
    with pytest.raises(expected_error):
        kernel_owned_array_payload_upper_bound_bytes(
            signal_length,
            support_length,
            _policy("direct"),
        )


def test_fft_auto_requires_thresholds_and_temp_cap() -> None:
    signal_length = 16
    kernel_length = 8

    assert selected_mode(
        _policy("fft_auto", fft_kernel_threshold=kernel_length + 1),
        signal_length,
        kernel_length,
    ) == "direct"
    assert selected_mode(
        _policy("fft_auto", fft_signal_threshold=signal_length + 1),
        signal_length,
        kernel_length,
    ) == "direct"
    assert selected_mode(
        _policy("fft_auto", fft_max_temp_bytes=1),
        signal_length,
        kernel_length,
    ) == "direct"
    assert selected_mode(
        _policy("fft_auto"),
        signal_length,
        kernel_length,
    ) == "fft"
