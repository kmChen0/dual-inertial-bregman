"""Numerical contracts for the standalone distribution."""
import unittest
import numpy as np
from _core import geometry, nnls
from experiments import regression, nnls as nnls_benchmark


class NumericalContracts(unittest.TestCase):
    def test_entropy_drift_integral(self):
        self.assertLess(geometry.check_formulas(), 1e-9)

    def test_nnls_oracle(self):
        rng = np.random.default_rng(8)
        matrix = rng.normal(size=(6, 6))
        gram = matrix.T @ matrix
        un, _, residual = nnls.nnls_prox(
            rng.normal(size=6), np.zeros(6), gram, rng.normal(size=6), 1., 1e-12)
        self.assertLessEqual(residual, 1e-12)
        self.assertTrue(np.all(np.isfinite(un)))

    def test_nnls_archived_initial_updates(self):
        nnls_benchmark.checks()

    def test_fista_diagonal_solution(self):
        matrix = np.eye(3)
        target = np.array([1., -1., 2.])
        optimum = np.maximum(target, 0.)
        gradient = optimum-target
        def gap(x):
            delta = x-optimum
            return float(.5*(delta@delta)+gradient@delta)
        data = matrix, target, matrix, target, optimum, gradient, gap
        for restart in (False, True):
            rows, _ = nnls_benchmark.fista_run(data, 5, restart)
            self.assertLess(abs(rows[-1]['gap']), 1e-14)


if __name__ == '__main__':
    unittest.main(verbosity=2)
