# Coherence: spectral analysis research software

A compact, private code showcase of a neuroscience analysis system developed from a MATLAB origin through collaborative Python, Nextflow and HPC engineering. This repository exposes a small executable CPU core and synthetic tests so that a reviewer can inspect concrete numerical and software contracts.

## Research and engineering problem

Electrophysiology analysis must connect signal extraction, a continuous wavelet transform (CWT), spectral coherence, and downstream statistics while controlling memory use, numerical conventions and provenance. The broader project separates a shared numerical core from platform orchestration and project-specific analysis.

```text
Input adapters -> extraction -> Morlet CWT -> smoothed coherence -> analysis
                                  |                 |
                            CPU core included in this showcase

Nextflow/HPC orchestration and project adapters: architecture described only
```

The selected implementation contains a standalone Morlet CWT with direct, FFT and overlap-add convolution paths, a byte-capped kernel-spectrum cache, explicit coherence policies, and a lazy interface that reuses left-hand-side state across multiple comparisons. It preserves exact-length outputs and validates inputs. The accompanying synthetic tests cover direct numerical oracles, boundary conditions, immutable outputs and cache behavior.

## Contributions and provenance

Yanbo Jiang's contribution includes the scientific analysis direction, MATLAB-origin workflow, analysis and maintenance work, and supervision of AI-assisted development. Research collaborators extended the Python/Nextflow/HPC lineage. This showcase does not claim sole authorship of that lineage or that every source line was written manually by Yanbo.

The extracted CPU modules and existing tests retain their source bytes. New showcase documentation, packaging and the small example were prepared with AI assistance. See [attribution](docs/attribution.md) for the contribution boundary and dependency licenses.

## Run locally

Requires Python 3.10 or later and NumPy. No data download, GPU, scheduler or network service is needed by the example or tests.

```sh
python -m pip install -e ".[test]"
python examples/synthetic_demo.py
python -m pytest -q
```

The example generates deterministic synthetic signals in memory, compares direct and FFT paths, and prints the measured discrepancy. It is a numerical illustration, not a throughput benchmark or an experimental finding. Existing tests exercise the included modules only.

## Scope and current status

- Included: the CPU CWT and coherence modules, their existing synthetic tests, one portable synthetic example, and concise architecture/provenance documentation.
- Described only: extraction, project adapters, downstream statistics, and Nextflow/HPC orchestration.
- Excluded: experimental and participant data, private identifiers, cluster configurations, credentials, private reviews, raw communications, binaries, and the original repository history.
- This snapshot does not establish full-pipeline runtime acceptance, scientific cohort acceptance, biological findings, GPU performance or production readiness. Ongoing development in the full project remains separate.

See [validation](docs/validation.md) for the checks actually performed on this export and [architecture](docs/architecture.md) for the broader design.

## Access and licensing

This is a private review copy. No open-source license or permission to redistribute the underlying research software is granted here. Rights remain with the relevant contributors and rights holders; see [LICENSE](LICENSE) and [third-party notices](THIRD_PARTY_NOTICES.md). Public access requires a separate rights and disclosure decision.
