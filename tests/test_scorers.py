import unittest
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.scoring.rule_scorers import RuleScorers
from src.engine.sandbox import CodeSandbox

class TestScorers(unittest.TestCase):
    def test_mcq_scorer(self):
        # Direct answer
        corr, ext = RuleScorers.score_mcq("The correct answer is B.", "B")
        self.assertTrue(corr)
        self.assertEqual(ext, "B")

        # Boxed answer (MMLU-Pro 10-option A-J)
        corr, ext = RuleScorers.score_mcq("After calculating, we find \\boxed{F}", "F", valid_options="ABCDEFGHIJ")
        self.assertTrue(corr)
        self.assertEqual(ext, "F")

        # Negative test
        corr, ext = RuleScorers.score_mcq("Option A is clearly the choice", "C")
        self.assertFalse(corr)
        self.assertEqual(ext, "A")

    def test_math_scorer(self):
        # GSM8K hash format
        corr, ext = RuleScorers.score_math("We do 50 + 22.\n#### 72", "72")
        self.assertTrue(corr)
        self.assertEqual(ext, "72")

        # Boxed LaTeX
        corr, ext = RuleScorers.score_math("Therefore, the answer is \\boxed{23}", "23")
        self.assertTrue(corr)
        self.assertEqual(ext, "23")

        # Fraction equality
        corr, ext = RuleScorers.score_math("The probability is 2/12", "1/6")
        self.assertTrue(corr)

    def test_code_sandbox(self):
        code = """def add(a, b):
    return a + b
"""
        test_code = """def check(candidate):
    assert candidate(2, 3) == 5
    assert candidate(-1, 1) == 0
"""
        passed, err = CodeSandbox.execute_code(code, test_code, "add", timeout_seconds=3.0)
        self.assertTrue(passed)
        self.assertIsNone(err)

if __name__ == "__main__":
    unittest.main()
