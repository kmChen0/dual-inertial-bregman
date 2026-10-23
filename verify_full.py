"""Compare complete seed-1 trajectories with the supplied reference histories.

This reruns one trajectory per setting/method, not the full timing protocol.
It does not regenerate the 20-seed benchmark or modify reference data.
"""
import json
from pathlib import Path
from _core import __doc__ as _runtime_setup
from experiments import regression as r, nnls as n
import numpy as np

ROOT = Path(__file__).resolve().parent


def main():
    results = []
    for size in (64, 256):
        data = r.orth.instance(1, 3*size, size)
        for method in r.METHODS:
            rows, _, failure, _ = r.run(method, data, 3000, True)
            if failure is not None:
                raise RuntimeError(failure)
            recorded = r.read_csv(ROOT/f'data/paper/regression/n{size}_{method}.csv')
            current = np.asarray([x['gap'] for x in rows])
            old = np.asarray([x['gap'] for x in recorded])
            np.testing.assert_allclose(current, old, rtol=1e-10, atol=1e-13)
            results.append(dict(problem='regression', setting=size, method=r.LABEL[method],
                                compared_states=len(rows), max_absolute_gap_change=float(np.max(abs(current-old)))))
            print('PASS regression', size, r.LABEL[method], flush=True)
    for rho in (.3, .95):
        data = n.INSTANCE(1, rho, m=480, n=320)
        n.ENV['nnls_instance'] = lambda *args: data
        for method in n.METHODS[:4]:
            internal = n.SOURCE_METHOD.get(method, method)
            rows, info = n.ENV['nnls_run'](internal, 1, rho, 500, tol=1e-12)
            recorded = n.read_csv(ROOT/f'data/paper/nnls/rho{rho}_{internal}_run0.csv')
            current = np.asarray([x['gap'] for x in rows])
            old = np.asarray([x['gap'] for x in recorded])
            np.testing.assert_allclose(current, old, rtol=1e-10, atol=1e-13)
            if info['maximum_oracle_residual'] > 1e-12:
                raise AssertionError('Oracle residual exceeded tolerance')
            np.testing.assert_array_equal([x['rejected'] for x in rows], [x['rejected'] for x in recorded])
            results.append(dict(problem='NNLS', setting=rho, method=method,
                                compared_states=len(rows), maximum_oracle_residual=info['maximum_oracle_residual'],
                                max_absolute_gap_change=float(np.max(abs(current-old)))))
            print('PASS NNLS', rho, method, flush=True)
    output = ROOT/'results/generated/full_trajectory_check.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(complete_trajectories=len(results), results=results), indent=2)+'\n')
    print('Saved:', output)


if __name__ == '__main__':
    main()
