# Dual-Inertial Bregman Methods

Numerical experiments accompanying **Dual-Inertial Bregman Methods with Energy-Based Safeguarding**, by Kangming Chen.

This repository contains implementations of the entropy-regression and nonnegative least-squares experiments. It separates the original seed-1 manuscript protocol from the additional paired 20-seed benchmarks. The additional benchmarks are not all experiments included in the manuscript.

## Quick start

Use Python 3.9 or later. From the repository root:

```sh
python -m venv .venv
```

Activate the environment (`source .venv/bin/activate` on macOS/Linux; `.venv\Scripts\Activate.ps1` in Windows PowerShell), then:

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python run_paper.py --check
python run_paper.py
```

The final command redraws the four seed-1 performance/geometry figures from the supplied histories. New files go to `results/generated/paper/`; reference data are not overwritten. No dataset downloads or network access are needed after installing dependencies.

For the originally tested package versions, use `requirements-tested.txt` instead of `requirements.txt`. Recorded runs used Python 3.9.12, NumPy 1.26.3, SciPy 1.11.4, and Matplotlib 3.5.1 on an Apple M1 Pro, with one numerical-library thread. Fresh timings will differ between machines. The broader version ranges in `requirements.txt` are compatibility targets, not a claim that every combination has been tested.

## Experiments

| Protocol | Settings | Methods | Budget |
|---|---|---|---|
| Manuscript, seed 1 | KL regression: 192-by-64 and 768-by-256 | BPG, AEPG, ABPG-g, DI-BPG, S-DI-BPG | 3,000 outer updates; 5 timing repeats |
| Manuscript, seed 1 | NNLS: 480-by-320; correlation 0.3 and 0.95 | BPPA, V-iBPPA, DI-BPPA, S-DI-BPPA | 500 proximal calls; 3 timing repeats |
| Additional, seeds 1–20 | Same regression sizes | Same regression methods | 3,000 outer updates; 3 timing repeats |
| Additional, seeds 1–20 | Same NNLS settings | Same BPPA family, projected FISTA, gradient-restarted FISTA | 500 proximal calls or 10,000 FISTA steps; 3 timing repeats |

The regression stepsize is `lambda = 0.99/L`; ABPG-g starts with base parameter `1/L` and performs gain adaptation. NNLS uses `lambda = 1` and Newton infinity-residual tolerance `1e-12`. The NNLS oracle is a numerical solve to this tolerance, not a symbolic exact solve.

`DI-BFB` and `S-DI-BFB` are retained in internal identifiers and archived filenames; they denote **DI-BPG** and **S-DI-BPG**. `V-iEPPA (q=2, heuristic)` denotes the plotted **V-iBPPA** implementation. This is the practical entropy exponent-two rule, not a theorem-certified global quadratic-scaling guarantee for entropy.

### Recompute the manuscript performance runs

```sh
python run_paper.py --rerun
```

This reruns the original performance protocol. Geometry panels still use the supplied audited seed-1 histories; they are not silently relabeled as newly generated diagnostics. The additional regression runner below computes fresh geometry histories for every generated DI instance.

### Additional multi-seed benchmarks

Start with short checks:

```sh
python -m experiments.regression --smoke --output results/generated/regression-smoke
python -m experiments.nnls --smoke --output results/generated/nnls-smoke
```

Run the full declared grids explicitly:

```sh
python -m experiments.regression --run
python -m experiments.nnls --run
```

Both commands may take substantial time. Run them serially, without other heavy computations. A shared file lock prevents these two runners from timing concurrently when they share an output parent directory. New benchmark output directories must be empty: cached or partial runs are not silently mixed with new timings. Use `--output` to select a fresh directory. Seeds, budgets, and timing repetitions are configurable; see `--help`. Changing them changes the protocol.

To replot a completed full run:

```sh
python -m experiments.regression --plot-only results/generated/regression
python -m experiments.nnls --summarize --output results/generated/nnls
```

The regression plot suite includes a predeclared seed-1 local-energy panel, so it expects seed 1, both sizes, and the full 3,000-update horizon. Smaller custom runs still save their traces and summaries.

## Reading the results

- The manuscript regression performance plots show **current-iterate gaps**. The manuscript NNLS plots and additional primary performance plots show **best-accepted-point gaps**. Additional regression current-iterate plots are supplied separately.
- All gaps are normalized by the gap at the shared initial point. Reference gaps and geometric diagnostics are evaluated outside update timing.
- In the multi-seed benchmarks, objective-based incumbent selection is charged to every method. Update timing includes safeguard evaluation, gain/backtracking searches, rejected trials, and inner Newton solves. It excludes instance construction, reference solves, offline diagnostics, and plotting.
- Original-space objective evaluations used for incumbent selection are not counted as gradient calls. Regression summaries additionally report function and matrix-vector counts. NNLS FISTA spectral setup is reported separately from update time.
- Gradient steps and exact-proximal calls are **not interchangeable units**. In particular, the NNLS FISTA and BPPA families have different fixed horizons.
- Each multi-seed timing history uses the median of three within-instance repeats. Shaded bands are the 25th–75th percentiles across instances (**IQR**, not confidence intervals). Time curves do not extend beyond the observed common horizon.
- Target-time medians are conditional on reaching the target and must be read together with the success counts. Failures remain in the output logs.
- NNLS near-reference gaps can be slightly negative due to reference-solver and floating-point error. Raw signs are retained; nonpositive values are omitted from logarithmic plots. Such values do not certify additional accuracy.
- Local-energy fits are exploratory finite-horizon diagnostics, not a numerical proof of eventual linear convergence. Sparse NNLS references can lie on the entropy-domain boundary, outside the interior-minimizer local theorem.

See [docs/PROTOCOL.md](docs/PROTOCOL.md) for problem generation, baseline settings, and reference accuracy, and [docs/REFERENCES.md](docs/REFERENCES.md) for algorithm sources.

The supplied additional regression figure is available in [results/reference/regression/regression_performance.pdf](results/reference/regression/regression_performance.pdf). To regenerate the target-time CSV and a compact LaTeX regression table from the recorded summaries:

```sh
python make_tables.py
```

After a full additional regression run, the exploratory exponential-versus-power-law comparison can be rebuilt with:

```sh
python exploratory_models.py
```

For a more extensive numerical fidelity check (18 complete seed-1 trajectories, without repeated benchmark timing):

```sh
python verify_full.py
```

## Repository layout

```text
run_paper.py              Seed-1 checks, replotting, and performance reruns
_core/                    Internal numerical routines and plotting helpers
experiments/              Additional regression and NNLS benchmark runners
make_tables.py            Target-time CSV and compact LaTeX table export
exploratory_models.py     Descriptive local-energy model comparison
verify_full.py            Full seed-1 trajectory agreement checks
tests/                    Trajectory, oracle, geometry, and accounting tests
data/paper/               Seed-1 histories and reference instances
results/reference/        Recorded multi-seed summaries and PDF/PNG figures
results/generated/        New outputs (ignored by Git)
SOURCE_PROVENANCE.json    Source snapshot checksums and packaging changes
CITATION.cff              Citation metadata
```



## Citation and license

Please cite the manuscript when using these experiments; metadata are provided in `CITATION.cff`. No software license has been selected for this release. No license file is included.

The repository was prepared with AI assistance in implementation, documentation, and reproducibility checks. The author is responsible for verifying the code and reported results.
