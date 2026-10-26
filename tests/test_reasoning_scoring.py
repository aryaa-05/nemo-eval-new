import json
import os
import sys
import unittest
from unittest import mock

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)

from src.benchmarks.base import get_benchmark
from src.comparison.diff_engine import DiffEngine
from src.engine.answer_scoring import score_generation
from src.engine.runner import EvaluationRunner
from src.gateway.model_client import UnifiedModelClient
from src.reporting.report_generator import ReportGenerator
from src.scoring.reasoning import UNCLOSED, strip_reasoning
from src.scoring.rule_scorers import RuleScorers

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "reasoning_cases.json")


def _load_cases():
    with open(FIXTURES, encoding="utf-8") as f:
        return json.load(f)["cases"]


class TestReasoningFixtures(unittest.TestCase):
    """Every fixture goes through the same score_generation() the runner uses."""

    def test_fixture_file_has_the_twelve_original_cases(self):
        ids = [c["id"] for c in _load_cases()]
        self.assertEqual(ids[:12], list(range(1, 13)))
        self.assertEqual(len(ids), len(set(ids)))

    def test_all_fixture_cases(self):
        for case in _load_cases():
            with self.subTest(case=case["id"], name=case["name"]):
                bench = get_benchmark(case["benchmark"], base_dir=REPO)
                gen_resp = {"text": case["text"], "finish_reason": case.get("finish_reason")}
                if case.get("error"):
                    gen_resp["error"] = case["error"]
                scored = score_generation(bench, {"answer": case["truth"]}, gen_resp)
                expect = case["expect"]
                self.assertEqual(scored.is_correct, expect["correct"])
                self.assertEqual(scored.extracted, expect["extracted"])
                self.assertEqual(scored.error, expect["error"])
                self.assertEqual(scored.truncated, expect["error"] == "truncated")
                self.assertEqual(scored.no_answer, expect["error"] == "no_answer_found")
                self.assertEqual(scored.api_error, str(expect["error"]).startswith("api_error"))


class TestStripReasoning(unittest.TestCase):
    def test_complete_block_is_removed(self):
        self.assertEqual(strip_reasoning("<think>a (B) b</think>\nAnswer: C"), ("Answer: C", "ok"))

    def test_multiple_blocks_and_case_insensitive_tags(self):
        self.assertEqual(strip_reasoning("<THINK>x</THINK>one<think>y</think> two"), ("one two", "ok"))

    def test_unclosed_block_means_no_answer(self):
        self.assertEqual(strip_reasoning("<think>still going Answer: C"), ("", UNCLOSED))

    def test_closer_without_opener_keeps_text_after_it(self):
        self.assertEqual(strip_reasoning("thoughts (B)\n</think>\nAnswer: C"), ("Answer: C", "ok"))

    def test_plain_text_is_unchanged_and_empty_is_safe(self):
        self.assertEqual(strip_reasoning("Answer: C"), ("Answer: C", "ok"))
        self.assertEqual(strip_reasoning(""), ("", "ok"))
        self.assertEqual(strip_reasoning(None), ("", "ok"))


class TestExtractorNeverReadsLettersFromProse(unittest.TestCase):
    def test_articles_and_pronouns_are_not_options(self):
        for text in ("I think a careful look is needed.", "a", "i", "Maybe a or b?", "Hope this is a help"):
            with self.subTest(text=text):
                self.assertIsNone(RuleScorers.extract_mcq_answer(text, "ABCDEFGHIJ"))

    def test_last_marker_wins(self):
        self.assertEqual(RuleScorers.extract_mcq_answer("Answer: A\nno wait\nAnswer: B", "ABCD"), "B")

    def test_out_of_range_letter_is_rejected(self):
        self.assertIsNone(RuleScorers.extract_mcq_answer("Answer: J", "ABCD"))
        self.assertEqual(RuleScorers.extract_mcq_answer("Answer: J", "ABCDEFGHIJ"), "J")

    def test_math_takes_the_last_boxed_and_handles_nested_braces(self):
        self.assertEqual(RuleScorers.extract_math_answer("\\boxed{9} then \\boxed{12}"), "12")
        self.assertEqual(RuleScorers.extract_math_answer("\\boxed{\\frac{1}{2}}"), "\\frac{1}{2}")


def _canned(text, finish_reason="stop", **extra):
    return {"text": text, "finish_reason": finish_reason, "latency_sec": 0.1, "prompt_tokens": 5,
            "completion_tokens": 5, "total_tokens": 10, "cost": 0.0, **extra}


class TestRunnerCountsAndAudit(unittest.TestCase):
    def _run(self):
        bench = get_benchmark("gpqa", base_dir=REPO)
        truths = [it["answer"] for it in bench.load_items(limit=4)]
        responses = [
            _canned(f"<think>\nmaybe (B)\n</think>\nAnswer: {truths[0]}"),          # correct after stripping
            _canned("<think>\nlong thought about (A) and (B)", finish_reason="length"),  # truncated
            _canned("<think>\nnever finishes", finish_reason="stop"),               # no answer
            _canned("no idea at all", finish_reason="stop"),                        # no answer
        ]
        runner = EvaluationRunner(base_dir=REPO)
        model = {"id": "vendor/reasoner", "name": "Reasoner", "provider": "mock"}
        with mock.patch.object(UnifiedModelClient, "generate", side_effect=responses):
            return runner.run_evaluation(model, ["gpqa"], limit_per_benchmark=4, show_progress=False)

    def test_counts_flags_and_accuracy(self):
        run = self._run()
        summary = run.benchmark_summaries["gpqa"]
        self.assertEqual((summary.correct_samples, summary.total_samples), (1, 4))
        self.assertEqual(summary.truncated_count, 1)
        self.assertEqual(summary.no_answer_count, 2)
        self.assertEqual((run.truncated_count, run.no_answer_count), (1, 2))
        self.assertEqual([it.error for it in run.item_results], [None, "truncated", "no_answer_found", "no_answer_found"])
        self.assertEqual([it.is_correct for it in run.item_results], [True, False, False, False])
        self.assertEqual([it.truncated for it in run.item_results], [False, True, False, False])

    def test_raw_response_is_kept_for_audit_and_scored_text_is_recorded(self):
        first = self._run().item_results[0]
        self.assertIn("<think>", first.raw_response)
        self.assertNotIn("<think>", first.scored_text)
        self.assertTrue(first.scored_text.startswith("Answer:"))

    def test_counts_appear_in_the_report(self):
        run = self._run()
        md = ReportGenerator.generate_markdown(DiffEngine.compare_runs(run, run))
        self.assertIn("1 truncated (hit max_tokens), 2 no answer found", md)

    def test_legacy_artifact_without_the_new_fields_still_loads(self):
        data = json.loads(self._run().model_dump_json())
        for key in ("truncated_count", "no_answer_count"):
            data.pop(key)
            for summary in data["benchmark_summaries"].values():
                summary.pop(key)
        for item in data["item_results"]:
            for key in ("scored_text", "reasoning_text", "finish_reason", "truncated"):
                item.pop(key, None)
        from src.storage.schema import RunArtifact
        loaded = RunArtifact.model_validate(data)
        self.assertEqual((loaded.truncated_count, loaded.no_answer_count), (0, 0))

    def test_simulated_runs_have_no_unscorable_responses(self):
        runner = EvaluationRunner(base_dir=REPO)
        model = {"id": "deepseek/deepseek-r1-distill-llama-70b", "name": "DS", "provider": "mock"}
        benches = ["mmlu_pro", "gsm8k", "gpqa", "kyc_india", "rbi_regulations", "image_processing"]
        run = runner.run_evaluation(model, benches, limit_per_benchmark=10, show_progress=False)
        self.assertEqual((run.truncated_count, run.no_answer_count), (0, 0))
        self.assertGreater(run.overall_accuracy, 0)


if __name__ == "__main__":
    unittest.main()
