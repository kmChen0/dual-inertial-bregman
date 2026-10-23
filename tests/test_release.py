"""Release layout and source-data integrity contracts."""
import csv
import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ReleaseContracts(unittest.TestCase):
    def test_reference_checksums(self):
        checksums=json.loads((ROOT/'DATA_SHA256.json').read_text())
        self.assertTrue(checksums)
        for name, expected in checksums.items():
            self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),expected,name)

    def test_reference_grid(self):
        def read(name):
            with (ROOT/'results/reference'/name).open() as stream:
                return list(csv.DictReader(stream))
        regression = read('regression/per_instance_summary.csv')
        nnls = read('nnls/per_seed_summary.csv')
        self.assertEqual(len(regression), 200)
        self.assertEqual(len(nnls), 240)
        self.assertTrue(all(int(r['failure_count']) == 0 for r in regression))
        self.assertTrue(all(r['status'] == 'completed' for r in nnls))
        self.assertEqual({int(r['seed']) for r in regression}, set(range(1, 21)))

    def test_archive_has_no_machine_paths(self):
        for base in ('_core', 'experiments', 'tests', 'data', 'docs'):
            for path in (ROOT/base).rglob('*'):
                if path.suffix in ('.py', '.json', '.md', '.csv'):
                    text = path.read_text()
                    self.assertNotIn('/'+'Users/', text, str(path))

    def test_target_table_can_be_regenerated(self):
        from make_tables import target_rows
        rows = target_rows(ROOT/'results/reference')
        with (ROOT/'results/reference/tables/target_times.csv').open() as stream:
            recorded = list(csv.DictReader(stream))
        self.assertEqual(len(rows), len(recorded))
        for current, old in zip(rows, recorded):
            self.assertEqual(current['method'], old['method'])
            self.assertEqual(int(current['successful_instances']), int(old['successful_instances']))
            if old['median_update_hit_ms']:
                self.assertAlmostEqual(current['median_update_hit_ms'], float(old['median_update_hit_ms']), places=10)
            else:
                self.assertIsNone(current['median_update_hit_ms'])
