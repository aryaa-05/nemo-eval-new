import gc
import io
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

import requests
import yaml
from click.testing import CliRunner
from rich.console import Console

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)

import src.cli.nel as nel
from src.benchmarks.base import get_benchmark
from src.comparison.diff_engine import DiffEngine
from src.engine.runner import EvaluationRunner
from src.gateway.model_client import UnifiedModelClient
from src.reporting.report_generator import ReportGenerator
from src.storage.run_store import RunStore
from src.storage.schema import UNRELIABLE_THRESHOLD, RunArtifact

KEY = "sentinel-nvidia-key-unreliable"
LIVE_MODEL = {"id": "vendor/live-model", "name": "Live Model", "provider": "nvidia_nim"}


def simulated_run(n=10, job="standin_t"):
    """A clean simulated run with n mmlu_pro items (mmlu_pro has 15 rows)."""
    return EvaluationRunner(base_dir=REPO).run_evaluation(
        {"id": "meta/llama-3.1-8b-instruct", "name": "T", "provider": "mock"}, ["mmlu_pro"],
        job_name=job, limit_per_benchmark=n, show_progress=False)


def with_counts(run, api_error=0, truncated=0, no_answer=0):
    return run.model_copy(update={"api_error_count": api_error, "truncated_count": truncated, "no_answer_count": no_answer})


def reply(status=200, content="Answer: A"):
    r = mock.MagicMock()
    r.status_code = status
    r.headers = {}
    if status >= 400:
        r.raise_for_status.side_effect = requests.exceptions.HTTPError(f"{status} Client Error: Synthetic")
    else:
        r.raise_for_status.return_value = None
    r.json.return_value = {"choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                           "usage": {"prompt_tokens": 5, "completion_tokens": 3}}
    return r


class TestThreshold(unittest.TestCase):
    def test_the_threshold_is_ten_percent(self):
        self.assertEqual(UNRELIABLE_THRESHOLD, 0.10)

    def test_exactly_ten_percent_is_not_unreliable_but_more_is(self):
        run = simulated_run(10)
        for flagged, expected in ((0, False), (1, False), (2, True), (5, True), (10, True)):
            with self.subTest(flagged=flagged):
                art = with_counts(run, api_error=flagged)
                self.assertEqual(art.unreliable, expected)

    def test_each_kind_of_flag_counts_and_they_add_up(self):
        run = simulated_run(10)
        self.assertTrue(with_counts(run, api_error=2).unreliable)
        self.assertTrue(with_counts(run, truncated=2).unreliable)
        self.assertTrue(with_counts(run, no_answer=2).unreliable)
        self.assertTrue(with_counts(run, api_error=1, truncated=1).unreliable)       # 1 + 1 = 20%
        self.assertFalse(with_counts(run, api_error=0, truncated=1, no_answer=0).unreliable)

    def test_the_fraction_is_reported(self):
        run = with_counts(simulated_run(10), api_error=1, truncated=1, no_answer=1)
        self.assertEqual(run.unreliable_fraction, 0.3)

    def test_a_clean_simulated_run_is_not_unreliable(self):
        run = simulated_run(10)
        self.assertFalse(run.unreliable)
        self.assertEqual(run.unreliable_fraction, 0.0)

    def test_an_empty_run_is_not_unreliable(self):
        run = simulated_run(10).model_copy(update={"item_results": []})
        self.assertFalse(run.unreliable)
        self.assertEqual(run.unreliable_fraction, 0.0)

    def test_the_artifact_file_itself_says_unreliable(self):
        data = json.loads(with_counts(simulated_run(10), api_error=3).model_dump_json())
        self.assertIs(data["unreliable"], True)
        self.assertEqual(data["unreliable_fraction"], 0.3)

    def test_files_with_the_computed_keys_and_old_files_without_them_both_load(self):
        run = with_counts(simulated_run(10), api_error=3)
        again = RunArtifact.model_validate_json(run.model_dump_json())
        self.assertTrue(again.unreliable)
        data = json.loads(simulated_run(10).model_dump_json())
        for key in ("unreliable", "unreliable_fraction", "api_error_count", "truncated_count", "no_answer_count"):
            data.pop(key, None)
        self.assertFalse(RunArtifact.model_validate(data).unreliable)


class TestFlowFromRealCallFailures(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict(os.environ, {"NVIDIA_API_KEY": KEY})
        patcher.start()
        self.addCleanup(patcher.stop)
        sleeper = mock.patch.object(UnifiedModelClient, "_sleep")
        sleeper.start()
        self.addCleanup(sleeper.stop)

    def live_run(self, statuses):
        bench = get_benchmark("gpqa", base_dir=REPO)
        truths = [it["answer"] for it in bench.load_items(limit=len(statuses))]
        replies = [reply(s, content=f"Answer: {t}") for s, t in zip(statuses, truths)]
        with mock.patch("src.gateway.model_client.requests.post", side_effect=replies):
            return EvaluationRunner(base_dir=REPO).run_evaluation(
                LIVE_MODEL, ["gpqa"], job_name="real_job", limit_per_benchmark=len(statuses), show_progress=False)

    def test_one_failed_call_in_ten_is_still_reliable(self):
        run = self.live_run([200] * 9 + [401])
        self.assertEqual(run.api_error_count, 1)
        self.assertFalse(run.unreliable)

    def test_two_failed_calls_in_ten_make_the_run_unreliable(self):
        run = self.live_run([200] * 8 + [401, 404])
        self.assertEqual(run.api_error_count, 2)
        self.assertTrue(run.unreliable)

    def test_two_read_timeouts_per_item_count_as_failed_items(self):
        bench = get_benchmark("gpqa", base_dir=REPO)
        truths = [it["answer"] for it in bench.load_items(limit=5)]
        slow = requests.exceptions.ReadTimeout("slow")
        replies = [reply(200, content=f"Answer: {truths[0]}"), slow, slow, reply(200, content=f"Answer: {truths[2]}"),
                   reply(200, content=f"Answer: {truths[3]}"), reply(200, content=f"Answer: {truths[4]}")]
        # item 2 times out, is retried once, times out again -> api_error
        with mock.patch("src.gateway.model_client.requests.post", side_effect=replies) as post:
            run = EvaluationRunner(base_dir=REPO).run_evaluation(
                LIVE_MODEL, ["gpqa"], job_name="real_job", limit_per_benchmark=5, show_progress=False)
        self.assertEqual(post.call_count, 6)  # 1 + 2 (timeout, retry) + 3
        self.assertEqual([i.attempts for i in run.item_results], [1, 2, 1, 1, 1])
        self.assertIn("read timed out after 2 attempt(s)", run.item_results[1].error)
        self.assertEqual(run.api_error_count, 1)


class TestReportAndCommandLine(unittest.TestCase):
    def test_report_header_marks_the_unreliable_side(self):
        ok, bad = simulated_run(10, "real_a"), with_counts(simulated_run(10, "real_b"), api_error=2)
        md = ReportGenerator.generate_markdown(DiffEngine.compare_runs(ok, bad))
        self.assertIn("Reliability: A = ok (0% of items api_error/truncated/no_answer_found), B = UNRELIABLE (20%)", md)
        self.assertIn("UNRELIABLE RUN", md)
        html = ReportGenerator.generate_html(DiffEngine.compare_runs(ok, bad))
        self.assertIn("UNRELIABLE RUN", html)

    def test_report_without_unreliable_runs_has_no_banner(self):
        md = ReportGenerator.generate_markdown(DiffEngine.compare_runs(simulated_run(10, "a"), simulated_run(10, "b")))
        self.assertIn("A = ok", md)
        self.assertIn("B = ok", md)
        self.assertNotIn("UNRELIABLE RUN", md)

    def test_comparison_carries_the_flags(self):
        comp = DiffEngine.compare_runs(simulated_run(10), with_counts(simulated_run(10), truncated=3))
        self.assertEqual((comp["unreliable_a"], comp["unreliable_b"]), (False, True))
        self.assertEqual(comp["unreliable_fraction_b"], 0.3)


class TestCli(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(gc.collect)
        cwd = os.getcwd()
        os.chdir(REPO)
        self.addCleanup(os.chdir, cwd)
        self.buf = io.StringIO()
        self.store = RunStore(os.path.join(self.tmp.name, "runs"))
        real_load_yaml = nel.load_yaml

        def load_yaml(path):
            data = real_load_yaml(path)
            if str(path).replace("\\", "/").endswith("configs/models_config.yaml"):
                data = {**data, "models": {**data.get("models", {}), LIVE_MODEL["id"]: LIVE_MODEL}}
            return data

        patchers = [
            mock.patch.dict(os.environ, {"NVIDIA_API_KEY": KEY}),
            mock.patch.object(UnifiedModelClient, "_sleep"),
            mock.patch.object(nel, "load_yaml", load_yaml),
            mock.patch.object(nel, "RunStore", lambda _p: self.store),
            mock.patch.object(nel, "console", Console(file=self.buf, width=250, force_terminal=False)),
        ]
        for p in patchers:
            p.start()
            self.addCleanup(p.stop)

    def run_cli(self, statuses, job="real_job"):
        path = os.path.join(self.tmp.name, "eval_t.yaml")
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump({"job_name": job, "seed": 42, "limit_per_benchmark": len(statuses),
                            "models": [LIVE_MODEL["id"]], "benchmarks": ["gpqa"]}, f)
        bench = get_benchmark("gpqa", base_dir=REPO)
        truths = [it["answer"] for it in bench.load_items(limit=len(statuses))]
        replies = [reply(s, content=f"Answer: {t}") for s, t in zip(statuses, truths)]
        with mock.patch("src.gateway.model_client.requests.post", side_effect=replies):
            result = CliRunner().invoke(nel.cli, ["eval", "run", "--config", path])
        return result.exit_code, self.buf.getvalue()

    def test_the_run_summary_says_unreliable_only_above_the_limit(self):
        code, out = self.run_cli([200] * 7 + [401] * 3)
        self.assertEqual(code, 0, out)
        self.assertIn("UNRELIABLE RUN", out)
        self.assertIn("3 of 10 items (30%)", out)
        self.buf.truncate(0); self.buf.seek(0)
        code, out = self.run_cli([200] * 9 + [401])
        self.assertEqual(code, 0, out)
        self.assertNotIn("UNRELIABLE RUN", out)

    def test_compare_and_gate_panels_show_the_reliability(self):
        self.run_cli([200] * 7 + [401] * 3)
        self.run_cli([200] * 10)
        bad_id, good_id = [r["run_id"] for r in sorted(self.store.list_runs(), key=lambda r: r["timestamp"])]
        self.buf.truncate(0); self.buf.seek(0)
        result = CliRunner().invoke(nel.cli, ["compare", "--run-a", bad_id, "--run-b", good_id])
        self.assertEqual(result.exit_code, 0, self.buf.getvalue())
        self.assertIn("Reliability: A=UNRELIABLE (30%), B=ok (0%)", self.buf.getvalue())
        self.buf.truncate(0); self.buf.seek(0)
        result = CliRunner().invoke(nel.cli, ["gate", "--baseline", bad_id, "--candidate", good_id])
        self.assertIn("baseline=UNRELIABLE", self.buf.getvalue())
        self.assertIn("not safe", self.buf.getvalue())


if __name__ == "__main__":
    unittest.main()
