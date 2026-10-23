# Packaging verification

Checked on 2026-10-23, using the pinned experiment environment on macOS.

- The automated contract tests passed, covering archived initial updates, NNLS reference data and oracle residuals, the entropy drift integral, safeguard/energy accounting, interpolation horizons, local-fit window exclusions, the recorded experiment grid, and target-table regeneration.
- `run_paper.py --check` passed for both regression sizes and both NNLS correlations.
- `verify_full.py` compared 18 complete seed-1 trajectories (34,018 states) against the supplied histories. Maximum absolute change in stored objective gaps was **0.0** in this environment. NNLS rejection histories also matched; the largest oracle infinity residual was approximately **9.895e-13**, below the specified 1e-12 tolerance.
- All four manuscript performance/geometry figures were redrawn from the supplied histories. The regression performance figure was visually checked.
- Short end-to-end regression and NNLS benchmarks completed, including the Euclidean FISTA comparisons. These smoke checks are not replacements for the recorded full benchmark results.
- Target-time tables were rebuilt from the recorded summaries. Exploratory local-model comparisons were recomputed from the original full regression histories and compared with the recorded table.

The full 20-seed timing grids were **not rerun during packaging**. Their recorded summaries and figures are supplied separately from newly generated output. The original manuscript, experiment sources and recorded input data were not edited. The distributable was also tested after copying it outside the research-project directory.

Runtime equality is not promised on other machines or dependency versions. Windows and Linux were not executed during this preparation.
