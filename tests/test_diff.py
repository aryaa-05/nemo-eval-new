import unittest
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.storage.run_store import RunStore
from src.comparison.diff_engine import DiffEngine

class TestDiffEngine(unittest.TestCase):
    def test_diff(self):
        store = RunStore("runs")
        runs = store.list_runs()
        if len(runs) >= 2:
            ra = store.get_run(runs[-1]["run_id"])
            rb = store.get_run(runs[0]["run_id"])
            comp = DiffEngine.compare_runs(ra, rb)
            self.assertIn("delta_accuracy", comp)
            self.assertIn("mcnemar", comp)
            self.assertIn("a_wins_flips", comp)
            self.assertIn("b_wins_flips", comp)

if __name__ == "__main__":
    unittest.main()
