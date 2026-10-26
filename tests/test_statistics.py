import unittest
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.scoring.statistics import StatisticsEngine

class TestStatistics(unittest.TestCase):
    def test_wilson_ci(self):
        low, high = StatisticsEngine.wilson_confidence_interval(8, 10, confidence=0.95)
        self.assertTrue(0.40 <= low <= 0.60)
        self.assertTrue(0.85 <= high <= 1.0)

    def test_mcnemar(self):
        # Distinct results
        a = [True, True, True, False, False, True, False, True, False, True]
        b = [False, False, True, True, True, True, True, True, True, True]
        res = StatisticsEngine.mcnemar_test(a, b)
        self.assertIn("p_value", res)
        self.assertIn("statistic", res)
        self.assertEqual(res["a_better_than_b_count"], 2)
        self.assertEqual(res["b_better_than_a_count"], 4)

    def test_quality_gate(self):
        gate_pass = StatisticsEngine.evaluate_quality_gate(
            baseline_summary={"overall_accuracy": 0.50},
            candidate_summary={"overall_accuracy": 0.80},
            min_accuracy=0.65
        )
        self.assertEqual(gate_pass["status"], "PASS")

        gate_fail = StatisticsEngine.evaluate_quality_gate(
            baseline_summary={"overall_accuracy": 0.80},
            candidate_summary={"overall_accuracy": 0.60},
            min_accuracy=0.65
        )
        self.assertEqual(gate_fail["status"], "FAIL")

if __name__ == "__main__":
    unittest.main()
