# Validation of this curated export

The curated export was checked on 2026-11-09 UTC using Python 3.11.9, NumPy 2.2.6 and pytest 9.1.1. Numerical-library thread counts were set to one. No experimental data were loaded.

| Check | Actual result |
| --- | --- |
| `python -B -m pytest -q -p no:cacheprovider` | 84 selected existing synthetic tests passed; three RuntimeWarnings in the deliberate finite-intermediate-overflow regression case were retained. |
| `python examples/synthetic_demo.py` | Passed using this source tree. Direct/FFT maximum absolute discrepancy was approximately 1.16e-12; the lazy shared-input result exactly matched the single-comparison result. |
| `python -m pip wheel --no-deps --no-cache-dir --wheel-dir <local-output> .` | Built the distribution successfully. Build dependencies were resolved in the packaging tool's isolated environment. |
| Install the built wheel into an isolated target with `pip install --no-deps --no-compile --no-index --target <local-target> <wheel>` | Passed; package imports were verified to come from that installed target, then the synthetic example passed again. |

The temporary installation was removed after the check. Build artifacts and private preparation evidence are not part of the Git delivery.

The scope is limited to the exported numerical modules and synthetic examples. Real experimental data, Nextflow, HPC, GPU execution, throughput benchmarking, full-pipeline deployment and scientific cohort acceptance were **not run or established** by these checks. No independent semantic review of the full research pipeline is claimed.
