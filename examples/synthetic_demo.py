"""Small, deterministic numerical illustration; no experimental data or timing claim."""

from __future__ import annotations

import json

import numpy as np

from coherence_yanbo.kernels.coh_cpu import ConvolutionPolicy, compute_single, iter_multi_rhs
from coherence_yanbo.kernels.cwt_cpu import CpuCwtKernel, CpuCwtKernelConfig


def main() -> None:
    rng = np.random.default_rng(23)
    step = 0.004
    time = np.arange(512, dtype=np.float64) * step
    lhs = np.sin(2.0 * np.pi * 8.0 * time) + 0.1 * rng.normal(size=time.size)
    rhs = np.sin(2.0 * np.pi * 8.0 * time + 0.4) + 0.1 * rng.normal(size=time.size)
    direct = CpuCwtKernel(CpuCwtKernelConfig(mode="np"))
    fft = CpuCwtKernel(CpuCwtKernelConfig(mode="fft"))
    lhs_direct, _ = direct.transform_single(lhs, 8.0, t_step=step)
    rhs_direct, _ = direct.transform_single(rhs, 8.0, t_step=step)
    lhs_fft, _ = fft.transform_single(lhs, 8.0, t_step=step)
    np.testing.assert_allclose(lhs_fft, lhs_direct, rtol=1e-11, atol=1e-11)

    policy = ConvolutionPolicy(mode="direct", t_step=step, wpara=7.0)
    single = compute_single(lhs_direct, rhs_direct, 8.0, policy)
    shared = list(iter_multi_rhs(lhs_direct, [rhs_direct, lhs_direct], 8.0, policy))
    np.testing.assert_array_equal(shared[0].values, single.values)
    assert single.values.shape == lhs.shape
    assert np.isfinite(single.values).all()
    assert not single.values.flags.writeable

    print(json.dumps({
        "input": "deterministic synthetic signals generated in memory",
        "samples": int(time.size),
        "frequency_hz": 8.0,
        "cwt_direct_fft_max_absolute_difference": float(np.max(np.abs(lhs_fft - lhs_direct))),
        "shared_lhs_matches_single": True,
        "coherence_output_dtype": str(single.values.dtype),
        "coherence_output_immutable": not single.values.flags.writeable,
        "scope": "numerical illustration only; no biological or throughput claim",
    }, indent=2))


if __name__ == "__main__":
    main()
