import contextlib
import gc
import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)

from src.comparison.diff_engine import DiffEngine
from src.engine.runner import EvaluationRunner
from src.reporting.report_generator import ReportGenerator
from src.scoring.version import SCORER_VERSION
from src.storage.run_store import RunStore
from src.storage.schema import RunArtifact
from src.tools import rescore_run
from src.tools.rescore_run import RescoreError, rescore_artifact, rescored_run_id, save_rescored

MOCK = {"id": "meta/llama-3.1-8b-instruct", "name": "T", "provider": "mock"}
PENCILS = "72 - 27 = 45.\n\n**Answer: 45 pencils left.**"   # right, but the original scorer read the wrong number


def base_run(job="standin_t"):
    return EvaluationRunner(base_dir=REPO).run_evaluation(MOCK, ["mmlu_pro", "gsm8k"], job_name=job,
                                                         limit_per_benchmark=5, show_progress=False)


def tampered_run():
    """A run as an older scorer would have saved it: one right math reply scored wrong, one wrong reply scored right,
    plus an api_error, a truncated and a no_answer item, and a humaneval item with a stored verdict."""
    run = base_run()
    items = list(run.item_results)
    gsm = [i for i in items if i.benchmark == "gsm8k"]
    # 0: right reply, scored wrong by the old scorer (the wrong-to-right case)
    gsm[0] = gsm[0].model_copy(update={"raw_response": PENCILS, "ground_truth": "45", "is_correct": False, "score": 0.0,
                                       "extracted_answer": "27", "error": None, "finish_reason": "stop", "truncated": False})
    # 1: wrong reply, scored right by a lenient old scorer (the right-to-wrong case)
    gsm[1] = gsm[1].model_copy(update={"raw_response": "The total is 41.", "ground_truth": "40", "is_correct": True, "score": 1.0,
                                       "extracted_answer": "40", "error": None, "finish_reason": "stop", "truncated": False})
    # 2: failed call; 3: truncated; 4: no answer
    gsm[2] = gsm[2].model_copy(update={"raw_response": "API Error: HTTP 500", "ground_truth": "7", "is_correct": False,
                                       "extracted_answer": None, "error": "api_error: HTTP 500", "finish_reason": None, "truncated": False})
    gsm[3] = gsm[3].model_copy(update={"raw_response": "Let me think... the answer is 7", "ground_truth": "7", "is_correct": False,
                                       "extracted_answer": None, "error": "truncated", "finish_reason": "length", "truncated": True})
    gsm[4] = gsm[4].model_copy(update={"raw_response": "I cannot say.", "ground_truth": "7", "is_correct": False,
                                       "extracted_answer": None, "error": "no_answer_found", "finish_reason": "stop", "truncated": False})
    mmlu = [i for i in items if i.benchmark == "mmlu_pro"]
    he = mmlu[0].model_copy(update={"benchmark": "humaneval", "question_id": "he_01", "is_correct": True, "score": 1.0,
                                    "extracted_answer": "Executed", "error": None, "raw_response": "def f(): pass"})
    items = mmlu[1:] + gsm + [he]
    return finish(run, items)


def finish(run, items):
    """Recompute summaries and totals from `items`, the way the runner would have."""
    summaries = {}
    for b_id, s in run.benchmark_summaries.items():
        its = [i for i in items if i.benchmark == b_id]
        summaries[b_id] = s.model_copy(update={
            "total_samples": len(its), "correct_samples": sum(i.is_correct for i in its), "accuracy": round(sum(i.is_correct for i in its) / len(its), 4),
            "truncated_count": sum(i.truncated for i in its), "no_answer_count": sum(i.error == "no_answer_found" for i in its),
            "api_error_count": sum((i.error or "").startswith("api_error") for i in its)})
    he = [i for i in items if i.benchmark == "humaneval"]
    if he:
        summaries["humaneval"] = summaries["gsm8k"].model_copy(update={"benchmark_id": "humaneval", "benchmark_name": "HumanEval", "total_samples": len(he),
                                                                       "correct_samples": 1, "accuracy": 1.0, "truncated_count": 0, "no_answer_count": 0, "api_error_count": 0})
    total = sum(s.total_samples for s in summaries.values())
    correct = sum(s.correct_samples for s in summaries.values())
    return run.model_copy(update={"item_results": items, "benchmark_summaries": summaries, "scorer_version": "unversioned",
                                  "overall_accuracy": round(correct / total, 4)})


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


class TestRescoreArtifact(unittest.TestCase):
    def setUp(self):
        self.old = tampered_run()
        self.before = self.old.model_dump_json()
        self.new, self.report = rescore_artifact(self.old)

    def item(self, run, qid_prefix=None, benchmark="gsm8k", index=0):
        return [i for i in run.item_results if i.benchmark == benchmark][index]

    def test_the_input_run_object_is_not_modified(self):
        self.assertEqual(self.old.model_dump_json(), self.before)

    def test_the_new_artifact_is_labelled_rescored_with_the_version(self):
        n = self.new
        self.assertEqual(n.run_id, rescored_run_id(self.old.run_id))
        self.assertIn("rescored", n.run_id)
        self.assertIn(f"v{SCORER_VERSION}", n.run_id)
        self.assertNotEqual(n.run_id, self.old.run_id)
        self.assertEqual((n.scorer_version, n.rescored_from), (SCORER_VERSION, self.old.run_id))
        self.assertTrue(n.is_rescored and n.rescored_at)
        self.assertIn("RESCORED offline", n.description)
        self.assertIn("no model was called", n.description)

    def test_provenance_fields_are_kept(self):
        for field in ("job_name", "timestamp", "run_mode", "model_id", "model_name", "provider", "seed", "status", "mean_latency", "total_tokens"):
            with self.subTest(field=field):
                self.assertEqual(getattr(self.new, field), getattr(self.old, field))
        self.assertTrue(self.new.is_standin)   # the stand-in label is kept
        self.assertEqual(self.new.run_mode, "simulated")

    def test_a_right_reply_scored_wrong_before_is_now_right(self):
        old, new = self.item(self.old), self.item(self.new)
        self.assertFalse(old.is_correct)
        self.assertTrue(new.is_correct)
        self.assertEqual(new.extracted_answer, "45")
        self.assertEqual([c["question_id"] for c in self.report["wrong_to_right"]], [old.question_id])
        self.assertEqual(self.report["wrong_to_right"][0]["old_extracted"], "27")

    def test_a_wrong_reply_scored_right_before_is_now_wrong_and_reported(self):
        self.assertTrue(self.item(self.old, index=1).is_correct)
        self.assertFalse(self.item(self.new, index=1).is_correct)
        self.assertEqual([c["question_id"] for c in self.report["right_to_wrong"]], [self.item(self.old, index=1).question_id])

    def test_api_error_truncated_and_no_answer_verdicts_are_kept(self):
        for index, error in ((2, "api_error: HTTP 500"), (3, "truncated"), (4, "no_answer_found")):
            with self.subTest(error=error):
                old, new = self.item(self.old, index=index), self.item(self.new, index=index)
                self.assertEqual((new.is_correct, new.error, new.extracted_answer), (False, error, None))
                self.assertEqual(new.truncated, old.truncated)
        self.assertEqual((self.new.truncated_count, self.new.no_answer_count, self.new.api_error_count), (1, 1, 1))

    def test_the_kept_items_do_not_appear_among_the_changes(self):
        changed_ids = {c["question_id"] for c in self.report["changed"]}
        for index in (2, 3, 4):
            self.assertNotIn(self.item(self.old, index=index).question_id, changed_ids)

    def test_humaneval_is_kept_exactly_as_stored(self):
        old = [i for i in self.old.item_results if i.benchmark == "humaneval"]
        new = [i for i in self.new.item_results if i.benchmark == "humaneval"]
        self.assertEqual([i.model_dump() for i in old], [i.model_dump() for i in new])
        self.assertEqual(self.report["items_kept_as_stored"], 1)
        self.assertTrue(self.report["benchmarks"]["humaneval"]["kept_as_stored"])

    def test_summaries_and_totals_are_recomputed_from_the_items(self):
        for b_id, s in self.new.benchmark_summaries.items():
            its = [i for i in self.new.item_results if i.benchmark == b_id]
            self.assertEqual(s.correct_samples, sum(i.is_correct for i in its))
            self.assertEqual(s.total_samples, len(its))
            self.assertAlmostEqual(s.accuracy, round(sum(i.is_correct for i in its) / len(its), 4))
        total = len(self.new.item_results)
        self.assertAlmostEqual(self.new.overall_accuracy, round(sum(i.is_correct for i in self.new.item_results) / total, 4))
        self.assertEqual(self.report["overall_after"], self.new.overall_accuracy)

    def test_the_report_has_before_and_after_per_benchmark(self):
        b = self.report["benchmarks"]["gsm8k"]
        self.assertEqual((b["correct_before"], b["correct_after"], b["total"]), (1, 1, 5))   # +1 (pencils), -1 (the lenient one)
        self.assertEqual(b["accuracy_before"], self.old.benchmark_summaries["gsm8k"].accuracy)

    def test_latency_tokens_and_cost_are_untouched(self):
        for o, n in zip(self.old.item_results, self.new.item_results):
            self.assertEqual((o.latency_sec, o.total_tokens, o.cost, o.raw_response, o.ground_truth, o.attempt_log),
                             (n.latency_sec, n.total_tokens, n.cost, n.raw_response, n.ground_truth, n.attempt_log))

    def test_rescoring_a_current_run_changes_nothing(self):
        fresh = base_run()
        new, report = rescore_artifact(fresh)
        self.assertEqual(report["changed"], [])
        self.assertEqual(new.overall_accuracy, fresh.overall_accuracy)

    def test_rescoring_a_rescored_run_keeps_the_original_id(self):
        again, _ = rescore_artifact(self.new)
        self.assertEqual(again.rescored_from, self.old.run_id)

    def test_failed_and_incomplete_runs_are_refused(self):
        for status in ("failed", "incomplete"):
            with self.subTest(status=status), self.assertRaises(RescoreError):
                rescore_artifact(self.old.model_copy(update={"status": status}))

    def test_a_summary_that_disagrees_with_the_items_is_an_error(self):
        broken = self.old.model_copy(update={"item_results": self.old.item_results[:-2]})
        with self.assertRaises(RescoreError):
            rescore_artifact(broken)

    def test_a_rescored_run_can_be_compared_and_the_report_says_rescored(self):
        md = ReportGenerator.generate_markdown(DiffEngine.compare_runs(self.old, self.new))
        self.assertIn("RESCORED", md)
        self.assertIn("different scorer versions", md)
        self.assertIn(f"B = {SCORER_VERSION} (RESCORED)", md)

    def test_no_network_call_is_made(self):
        with mock.patch("requests.post", side_effect=AssertionError("network call")), \
                mock.patch("requests.get", side_effect=AssertionError("network call")), \
                mock.patch("requests.Session.request", side_effect=AssertionError("network call")):
            rescore_artifact(self.old)


class TestFilesAndCli(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(gc.collect)
        self.dir = os.path.join(self.tmp.name, "runs")
        self.store = RunStore(self.dir)
        self.old = tampered_run()
        self.original_path = self.store.save_run(self.old)

    def cli(self, *args):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = rescore_run.main(list(args) + ["--runs-dir", self.dir])
        return code, out.getvalue(), err.getvalue()

    def test_the_original_file_is_byte_identical_afterwards(self):
        before = sha(self.original_path)
        code, out, _ = self.cli(self.old.run_id)
        self.assertEqual(code, 0)
        self.assertEqual(sha(self.original_path), before)
        self.assertIn(rescored_run_id(self.old.run_id), out)

    def test_the_new_file_exists_is_labelled_and_is_not_registered(self):
        self.cli(self.old.run_id)
        new_id = rescored_run_id(self.old.run_id)
        self.assertTrue(os.path.isfile(os.path.join(self.dir, f"{new_id}.json")))
        loaded = self.store.get_run(new_id)
        self.assertEqual((loaded.rescored_from, loaded.scorer_version), (self.old.run_id, SCORER_VERSION))
        self.assertEqual([r["run_id"] for r in self.store.list_runs()], [self.old.run_id])   # the registry still lists only the original
        self.assertEqual(self.store.list_unusable(), [])

    def test_running_it_twice_overwrites_only_its_own_output(self):
        self.cli(self.old.run_id)
        code, _, _ = self.cli(self.old.run_id)
        self.assertEqual(code, 0)

    def test_it_never_overwrites_a_file_that_is_not_a_rescored_run(self):
        new, _ = rescore_artifact(self.old)
        victim = os.path.join(self.dir, f"{new.run_id}.json")
        with open(victim, "w", encoding="utf-8") as f:
            f.write(json.dumps({"run_id": new.run_id, "note": "someone else's file"}))
        before = sha(victim)
        with self.assertRaises(RescoreError):
            save_rescored(self.dir, new)
        code, _, err = self.cli(self.old.run_id)
        self.assertEqual(code, 1)
        self.assertIn("refusing to overwrite", err)
        self.assertEqual(sha(victim), before)

    def test_a_missing_run_is_an_error_and_writes_nothing(self):
        code, _, err = self.cli("run_does_not_exist")
        self.assertEqual(code, 1)
        self.assertIn("no run file", err)
        self.assertEqual(len([f for f in os.listdir(self.dir) if f.endswith(".json")]), 1)

    def test_a_failed_run_is_refused(self):
        failed = self.old.model_copy(update={"run_id": "run_failed_x", "status": "failed", "failure_reason": "HTTP 404"})
        self.store.save_run(failed)
        code, _, err = self.cli("run_failed_x")
        self.assertEqual(code, 1)
        self.assertIn("FAILED", err)

    def test_the_markdown_and_json_reports_list_the_changed_items(self):
        md_path, js_path = os.path.join(self.tmp.name, "r.md"), os.path.join(self.tmp.name, "r.json")
        self.cli(self.old.run_id, "--report-md", md_path, "--report-json", js_path)
        md = open(md_path, encoding="utf-8").read()
        self.assertIn("Wrong -> right", md)
        self.assertIn("Right -> wrong", md)
        self.assertIn("RESCORED", md)
        self.assertIn("No model was called", md)
        data = json.load(open(js_path, encoding="utf-8"))
        self.assertEqual(len(data[0]["wrong_to_right"]), 1)
        self.assertEqual(len(data[0]["right_to_wrong"]), 1)

    def test_several_runs_in_one_call(self):
        second = base_run()
        self.store.save_run(second)
        code, out, _ = self.cli(self.old.run_id, second.run_id)
        self.assertEqual(code, 0)
        self.assertIn(rescored_run_id(second.run_id), out)

    def test_the_saved_file_loads_as_the_same_model(self):
        self.cli(self.old.run_id)
        new, _ = rescore_artifact(self.old)
        loaded = self.store.get_run(new.run_id)
        self.assertIsInstance(loaded, RunArtifact)
        self.assertEqual(loaded.overall_accuracy, new.overall_accuracy)


if __name__ == "__main__":
    unittest.main()
