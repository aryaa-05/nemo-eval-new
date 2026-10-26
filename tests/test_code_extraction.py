import os
import sys
import unittest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)

from src.benchmarks.base import get_benchmark
from src.engine.answer_scoring import score_generation
from src.engine.solvers import CODE_INSTRUCTION, get_solver
from src.scoring.code_extraction import extract_python_code

TYPED_ITEM = {
    "task_id": "Synthetic/0",
    "prompt": 'from typing import List\n\n\ndef total(xs: List[int]) -> int:\n    """Sum the list."""\n',
    "entry_point": "total",
    "test": "def check(candidate):\n    assert candidate([1, 2, 3]) == 6\n    assert candidate([]) == 0\n",
}


def _score(item, text, finish_reason="stop"):
    bench = get_benchmark("humaneval", base_dir=REPO)
    return score_generation(bench, item, {"text": text, "finish_reason": finish_reason})


class TestExtractPythonCode(unittest.TestCase):
    def test_fenced_block_with_prose_around_it(self):
        reply = "Sure! Here is the function:\n\n```python\ndef f(x):\n    return x\n```\n\nIt returns its input."
        self.assertEqual(extract_python_code(reply), "def f(x):\n    return x\n")

    def test_last_block_defining_a_function_wins(self):
        reply = "```python\ndef f(x):\n    return 0\n```\nOops, fixed:\n```python\ndef f(x):\n    return x\n```\nExample:\n```\nf(1)\n```"
        self.assertIn("return x", extract_python_code(reply))
        self.assertNotIn("return 0", extract_python_code(reply))

    def test_unfenced_function_after_a_line_of_prose(self):
        reply = "Here is the solution:\ndef f(x):\n    return x\n"
        self.assertEqual(extract_python_code(reply), "def f(x):\n    return x\n")

    def test_bare_indented_body_is_attached_to_the_prompt(self):
        code = extract_python_code("    return sum(xs)\n", TYPED_ITEM["prompt"])
        self.assertIn("def total(xs: List[int])", code)
        self.assertIn("return sum(xs)", code)

    def test_imports_from_the_prompt_are_added_when_missing(self):
        code = extract_python_code("```python\ndef total(xs: List[int]) -> int:\n    return sum(xs)\n```", TYPED_ITEM["prompt"])
        self.assertTrue(code.startswith("from typing import List\n"))

    def test_imports_are_not_duplicated(self):
        reply = "```python\nfrom typing import List\n\ndef total(xs: List[int]) -> int:\n    return sum(xs)\n```"
        self.assertEqual(extract_python_code(reply, TYPED_ITEM["prompt"]).count("from typing import List"), 1)

    def test_unterminated_fence_still_yields_code(self):
        self.assertIn("return x", extract_python_code("```python\ndef f(x):\n    return x\n"))

    def test_prose_only_and_empty_replies_give_none(self):
        for reply in ("I am not able to help with that.", "", "   \n ", None):
            with self.subTest(reply=reply):
                self.assertIsNone(extract_python_code(reply))


class TestHumanEvalScoring(unittest.TestCase):
    def test_fenced_reply_with_prose_passes(self):
        reply = "Here you go:\n```python\nfrom typing import List\n\ndef total(xs: List[int]) -> int:\n    return sum(xs)\n```\nDone."
        scored = _score(TYPED_ITEM, reply)
        self.assertTrue(scored.is_correct, scored.error)

    def test_missing_import_in_reply_is_repaired_from_the_prompt(self):
        scored = _score(TYPED_ITEM, "```python\ndef total(xs: List[int]) -> int:\n    return sum(xs)\n```")
        self.assertTrue(scored.is_correct, scored.error)

    def test_body_only_reply_passes(self):
        self.assertTrue(_score(TYPED_ITEM, "    return sum(xs)\n").is_correct)

    def test_wrong_code_fails_with_a_real_assertion_error_not_a_syntax_error(self):
        scored = _score(TYPED_ITEM, "```python\ndef total(xs: List[int]) -> int:\n    return 0\n```")
        self.assertFalse(scored.is_correct)
        self.assertTrue(scored.error.startswith("AssertionError"), scored.error)
        self.assertFalse(scored.no_answer)

    def test_reasoning_is_stripped_before_code_is_run(self):
        reply = "<think>\nmaybe (B) ```python\nnot code\n```\n</think>\n```python\ndef total(xs: List[int]) -> int:\n    return sum(xs)\n```"
        self.assertTrue(_score(TYPED_ITEM, reply).is_correct)

    def test_reply_with_no_code_is_flagged_not_executed(self):
        scored = _score(TYPED_ITEM, "I cannot help with that request.")
        self.assertFalse(scored.is_correct)
        self.assertEqual(scored.error, "no_answer_found")

    def test_truncated_code_is_flagged_truncated(self):
        scored = _score(TYPED_ITEM, "```python\ndef total(xs: List[int]) -> int:\n    return sum(", finish_reason="length")
        self.assertEqual(scored.error, "truncated")

    def test_canonical_solutions_in_a_fenced_reply_pass_for_every_dataset_item(self):
        bench = get_benchmark("humaneval", base_dir=REPO)
        for item in bench.load_items(limit=15):
            with self.subTest(task=item["task_id"]):
                reply = f"```python\n{item['prompt']}{item['canonical_solution']}```"
                scored = score_generation(bench, item, {"text": reply, "finish_reason": "stop"})
                self.assertTrue(scored.is_correct, scored.error)


class TestHumanEvalPrompt(unittest.TestCase):
    def test_every_solver_asks_for_a_fenced_function_and_keeps_the_stub(self):
        item = get_benchmark("humaneval", base_dir=REPO).load_items(limit=1)[0]
        for solver in ("simple", "cot", "harbor"):
            with self.subTest(solver=solver):
                prompt = get_solver(solver).format_prompt(item, "humaneval")
                self.assertTrue(prompt.startswith(CODE_INSTRUCTION))
                self.assertTrue(prompt.endswith(item["prompt"]))

    def test_non_code_prompts_are_unchanged(self):
        item = {"question": "What is 2+2?", "options": ["3", "4"]}
        prompt = get_solver("simple").format_prompt(item, "gpqa")
        self.assertNotIn("python", prompt.lower())
        self.assertIn("A. 3\nB. 4", prompt)


if __name__ == "__main__":
    unittest.main()
