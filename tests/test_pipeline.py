"""Scientific sanity checks: split isolation, reproducibility and numerical stability."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from virtual_cell import generate, simulate, GENES


class SimulationTests(unittest.TestCase):
    def test_reproducible_and_finite(self):
        a, b = generate(8), generate(8)
        self.assertTrue(a.equals(b))
        self.assertEqual(len(a), 72)
        self.assertTrue(np.isfinite(a.select_dtypes("number")).all().all())
        self.assertTrue((a[[f"post_{g}" for g in GENES]] > 0).all().all())

    def test_step_size_stability(self):
        initial = np.ones((10, 4))
        rates = np.ones_like(initial)
        coarse = simulate(initial, rates, (.75, 0), dt=.02)
        fine = simulate(initial, rates, (.75, 0), dt=.01)
        np.testing.assert_allclose(coarse, fine, rtol=.005, atol=.002)

    def test_held_out_cells(self):
        from sklearn.model_selection import GroupShuffleSplit
        df = generate(20)
        standard = df[df.domain == "standard"]
        tr, te = next(GroupShuffleSplit(n_splits=1, test_size=.25, random_state=7204)
                      .split(standard, groups=standard.cell_id))
        train_ids = set(standard.iloc[tr].cell_id)
        test_ids = set(standard.iloc[te].cell_id)
        self.assertFalse(train_ids & test_ids)
        self.assertEqual(len(train_ids | test_ids), 20)


if __name__ == "__main__":
    unittest.main()
