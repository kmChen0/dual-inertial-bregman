"""Replot recorded Section 6 results, check implementations, or rerun the protocol."""
import os
from _core import __doc__ as _runtime_setup
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'


import argparse
import csv
import importlib.util
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
from types import SimpleNamespace
from _core import regression as publication, nnls as proximal, geometry, paper_plots
s=SimpleNamespace(publication=publication,proximal=proximal,geometry=geometry,
                  REG_METHODS=paper_plots.REG_METHODS,NNLS_METHODS=paper_plots.NNLS_METHODS,
                  plot_performance=paper_plots.plot_performance)
OUT = HERE/'results/generated/paper'
OUT.mkdir(parents=True,exist_ok=True)
s.geometry.HERE = paper_plots.HERE = OUT
s.publication.OUT = OUT
instances = {rho: s.proximal.nnls_instance(1, rho, m=480, n=320)
             for rho in (.3, .95)}
s.proximal.nnls_instance = lambda seed, rho: instances[rho]


def read(path):
    with path.open() as stream:
        return [{key: float(value) if value else 0.
                 for key, value in row.items()} for row in csv.DictReader(stream)]


def check():
    details = {}
    for n in (64, 256):
        data = s.publication.old.orth.instance(1, 3*n, n)
        for method in s.REG_METHODS:
            rows, _, failure = s.publication.regression_run(
                method, data, K=5, step_factor=1. if method == 'ABPG-g' else .99)
            assert failure is None
            saved = read(HERE/f'data/paper/regression/n{n}_{method}.csv')
            np.testing.assert_allclose([r['gap'] for r in rows],
                                       [r['gap'] for r in saved[:6]], rtol=1e-10, atol=1e-13)
    for rho, data in instances.items():
        projected = np.where(data[4] > 1e-12, data[5], np.minimum(data[5], 0))
        details[str(rho)] = dict(reference_kkt=float(np.max(np.abs(projected))))
        assert details[str(rho)]['reference_kkt'] < 1e-10
        for method in s.NNLS_METHODS:
            rows, info = s.proximal.nnls_run(method, 1, rho, budget=5, tol=1e-12)
            saved = read(HERE/f'data/paper/nnls/rho{rho}_{method}_run0.csv')
            np.testing.assert_allclose([r['gap'] for r in rows],
                                       [r['gap'] for r in saved[:6]], rtol=1e-10, atol=1e-13)
            assert info['maximum_oracle_residual'] <= 1e-12
    print(json.dumps(dict(first_five_updates_match=True, nnls=details), indent=2))


def performance(rerun=False):
    for block, settings in (('regression', (64, 256)), ('nnls', (.3, .95))):
        cases = {}
        for value in settings:
            runs, clocks = {}, {}
            methods = s.REG_METHODS if block == 'regression' else s.NNLS_METHODS
            repeats = 5 if block == 'regression' else 3
            data = s.publication.old.orth.instance(1, 3*value, value) if block == 'regression' else None
            for method in methods:
                if rerun:
                    clocks[method] = []
                    for repeat in range(repeats):
                        if block == 'regression':
                            rows, clock, failure = s.publication.regression_run(
                                method, data, K=3000,
                                step_factor=1. if method == 'ABPG-g' else .99,
                                record=(repeat == 0))
                            assert failure is None
                        else:
                            rows, info = s.proximal.nnls_run(method, 1, value, budget=500, tol=1e-12)
                            assert info['maximum_oracle_residual'] <= 1e-12
                            clock = [r['update_seconds'] for r in rows]
                        if repeat == 0:
                            runs[method] = rows
                            s.publication.save(f'{block}_{value}_{method}', rows)
                        clocks[method].append(clock)
                    s.publication.save(f'{block}_{value}_{method}_times', [
                        dict(k=k, **{f'rep{j}': clocks[method][j][k] for j in range(repeats)})
                        for k in range(len(clocks[method][0]))])
                elif block == 'regression':
                    runs[method] = read(HERE/f'data/paper/regression/n{value}_{method}.csv')
                    timing = read(HERE/f'data/paper/regression/n{value}_{method}_times.csv')
                    clocks[method] = [[r[f'rep{j}'] for r in timing] for j in range(repeats)]
                else:
                    recorded = [read(HERE/f'data/paper/nnls/rho{value}_{method}_run{j}.csv')
                                for j in range(repeats)]
                    runs[method] = recorded[0]
                    clocks[method] = [[r['update_seconds'] for r in trace] for trace in recorded]
            cases[value] = runs, clocks
        s.plot_performance(block, cases)
        geometry = {value: (
            read(HERE/f'data/paper/{block}/geometry_{value}.csv'),
            cases[value][0]['S-DI-BFB' if block == 'regression' else 'S-DI-BPPA'], {})
            for value in settings}
        # Geometry figures always use the archived, independently audited traces.
        s.geometry.plot(block, geometry)
    print(f'Figures written to {OUT}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check', action='store_true', help='Check all methods against saved initial updates.')
    mode.add_argument('--rerun', action='store_true', help='Recompute trajectories and serial update timings.')
    args = parser.parse_args()
    check() if args.check else performance(args.rerun)
