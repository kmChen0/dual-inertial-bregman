"""Contract tests for the isolated multi-instance preview."""
import unittest
import numpy as np
from experiments import regression as r


class PreviewTests(unittest.TestCase):
    def test_saved_seed1_first_five(self):
        for n in (64, 256):
            data = r.orth.instance(1, 3*n, n)
            for method in r.METHODS:
                rows, clocks, failure, states = r.run(method, data, 5, True)
                self.assertIsNone(failure)
                saved = r.read_csv(r.SOURCE/'data/paper/regression'/f'n{n}_{method}.csv')
                np.testing.assert_allclose([x['gap'] for x in rows],
                    [x['gap'] for x in saved[:6]], rtol=1e-10, atol=1e-13)
                self.assertEqual(len(clocks), len(rows))
                if method in r.DUAL:
                    self.assertEqual(len(states), len(rows))

    def test_diagnostics_and_accounting(self):
        data = r.orth.instance(9, 48, 16)
        for method in r.DUAL:
            rows, _, failure, states = r.run(method, data, 80, True)
            self.assertIsNone(failure)
            rows, geo, meta = r.diagnostics(method, rows, states, data)
            self.assertEqual(len(rows), 81)
            self.assertTrue(all(x['best_gap'] <= x['gap']+1e-12 for x in rows))
            self.assertTrue(all(x['energy'] >= -1e-12 for x in rows))
            if method == 'S-DI-BFB':
                self.assertEqual(rows[-1]['accepted']+rows[-1]['rejected'],80)
                self.assertLess(meta['maximum_energy_increase_scaled'],1e-10)
            else:
                self.assertLess(meta['maximum_reconstruction_residual'],1e-10)
                self.assertEqual(len(geo),79)

    def test_tail_fit_rejects_short_windows(self):
        rows=[dict(k=j, epoch=1, epoch_j=j, gap=10.**(-j), relative_gap=10.**(-j),
                   energy=10.**(-j)) for j in range(9)]
        fit=r.tail_fit(rows,'relative_gap')
        self.assertEqual(fit['status'],'insufficient_window')

    def test_tail_fit_rejects_flat_windows(self):
        rows=[dict(k=j,epoch=1,epoch_j=j,relative_gap=1e-5*(1-j*1e-4),
                   energy=1-j*1e-4) for j in range(100)]
        self.assertEqual(r.tail_fit(rows,'energy')['status'],'insufficient_dynamic_range')

    def test_interpolation_no_extrapolation(self):
        xx=np.array([0.,1.,2.]); yy=np.array([1.,.1,.01])
        got=r.step_interpolate(xx,yy,np.array([0.,.5,1.,2.,3.]))
        np.testing.assert_allclose(got[:4],[1.,1.,.1,.01])
        self.assertTrue(np.isnan(got[-1]))


if __name__=='__main__':
    unittest.main(verbosity=2)
