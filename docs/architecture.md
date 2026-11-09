# Architecture and exported boundary

The broader system uses three layers:

1. **Shared numerical core:** extraction contracts, wavelet transforms, coherence and reusable numerical helpers.
2. **Platform orchestration:** Nextflow/HPC batch processing and a separate platform-driver boundary.
3. **Project analysis:** input conventions, analysis questions, quality control and downstream statistics.

This export includes only two CPU numerical modules and their existing tests. It contains no runnable production workflow or real-data input adapter.

## CWT

`coherence_yanbo.kernels.cwt_cpu` constructs a Morlet kernel and computes an exact-length complex transform. Explicit configuration selects direct convolution, FFT convolution, a signal-spectrum-sharing FFT variant, automatic dispatch or overlap-add. A per-engine byte-capped cache reuses kernel spectra and coordinates concurrent requests for the same cache key.

## Coherence

`coherence_yanbo.kernels.coh_cpu` smooths powers and the cross-spectrum, normalizes their ratio, and returns an immutable float32 coherence buffer. An explicit convolution policy controls direct or FFT evaluation. The multi-right-hand-side interface retains shared left-hand-side state and yields comparisons lazily.

The numerical convention clips finite coherence outputs to the interval [0, 1]. This is an implementation convention to inspect and test; it does not by itself establish a scientifically accepted estimator for every possible input.

## Orchestration boundary

In the broader project, input extraction, CWT, coherence and downstream reductions are separate stages in a versioned workflow. Task identity, resource settings, input/output contracts and numerical acceptance are tracked separately. Cluster-specific launchers, scheduler queues, hostnames, storage paths and private run records are deliberately absent from this export.

The synthetic example runs in one local Python process. It does not invoke Nextflow or a scheduler and provides no evidence about cluster throughput, real-data completion or production deployment.
