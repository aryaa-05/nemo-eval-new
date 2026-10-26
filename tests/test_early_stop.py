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
from src.comparison.diff_engine import DiffEngine
from src.engine.runner import EARLY_STOP_ITEMS, EARLY_STOP_STATUSES, EvaluationRunner, early_stop_reason
from src.gateway.interceptor import InterceptorProxy
from src.gateway.model_client import UnifiedModelClient
from src.reporting.report_generator import ReportGenerator
from src.storage.run_store import RunStore
from src.storage.schema import RunArtifact

KEY = "sentinel-nvidia-key-early-stop"
LIVE = {"id": "vendor/live-model", "name": "Live Model", "provider": "nvidia_nim"}
DEAD = {"id": "vendor/dead-model", "name": "Dead Model", "provider": "nvidia_nim"}


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


class LiveTestCase(unittest.TestCase):
    def setUp(self):
        for p in (mock.patch.dict(os.environ, {"NVIDIA_API_KEY": KEY}), mock.patch.object(UnifiedModelClient, "_sleep")):
            p.start()
            self.addCleanup(p.stop)

    def run_model(self, responses, model=LIVE, benchmarks=("gpqa",), limit=15):
        with mock.patch("src.gateway.model_client.requests.post", side_effect=responses) as post:
            run = EvaluationRunner(base_dir=REPO).run_evaluation(
                model, list(benchmarks), job_name="real_job", limit_per_benchmark=limit, show_progress=False)
        return run, post


class TestRuleConstants(unittest.TestCase):
    def test_the_rule_is_five_items_and_these_four_statuses(self):
        self.assertEqual(EARLY_STOP_ITEMS, 5)
        self.assertEqual(sorted(EARLY_STOP_STATUSES), [401, 403, 404, 410])

    def test_early_stop_reason_needs_exactly_five_identical_listed_statuses(self):
        self.assertIsNone(early_stop_reason(LIVE, [404] * 4))
        self.assertIsNone(early_stop_reason(LIVE, [404] * 6))
        self.assertIsNone(early_stop_reason(LIVE, [404, 404, 404, 404, None]))
        self.assertIsNone(early_stop_reason(LIVE, [404, 404, 404, 404, 401]))
        for status in (400, 408, 413, 422, 429, 500, 502, 503):
            self.assertIsNone(early_stop_reason(LIVE, [status] * 5), status)
        msg = early_stop_reason(LIVE, [404] * 5)
        self.assertIn("HTTP 404", msg)
        self.assertIn("vendor/live-model", msg)
        self.assertIn("first 5 items", msg)


class TestEarlyStop(LiveTestCase):
    def test_five_404s_stop_the_model_and_mark_the_run_failed(self):
        run, post = self.run_model([reply(404)] * 60, benchmarks=("gpqa", "mmlu_pro"))
        self.assertEqual(post.call_count, 5)  # exactly five calls, not 30
        self.assertEqual(run.status, "failed")
        self.assertTrue(run.is_failed)
        self.assertIn("HTTP 404", run.failure_reason)
        self.assertIn("vendor/live-model", run.failure_reason)
        self.assertEqual(len(run.item_results), 5)
        self.assertEqual(run.api_error_count, 5)
        self.assertEqual([i.http_status for i in run.item_results], [404] * 5)
        self.assertEqual(list(run.benchmark_summaries), ["gpqa"])  # mmlu_pro was never started
        self.assertEqual(run.benchmark_summaries["gpqa"].total_samples, 5)

    def test_a_failed_run_is_failed_not_unreliable(self):
        run, _ = self.run_model([reply(404)] * 60)
        self.assertEqual(run.unreliable_fraction, 1.0)  # 5 of 5 items failed...
        self.assertFalse(run.unreliable)                # ...but the run is FAILED, never UNRELIABLE

    def test_each_listed_status_triggers_it_and_names_itself(self):
        for status in EARLY_STOP_STATUSES:
            with self.subTest(status=status):
                run, post = self.run_model([reply(status)] * 60)
                self.assertEqual(post.call_count, 5)
                self.assertEqual(run.status, "failed")
                self.assertIn(f"HTTP {status}", run.failure_reason)

    def test_other_4xx_statuses_do_not_trigger_it(self):
        for status in (400, 408, 413, 422):
            with self.subTest(status=status):
                run, post = self.run_model([reply(status)] * 60, limit=8)
                self.assertEqual(post.call_count, 8)  # every item ran, one request each (4xx is never retried)
                self.assertEqual(run.status, "completed")
                self.assertTrue(run.unreliable)

    def test_429_and_5xx_never_trigger_it(self):
        for status in (429, 500, 502, 503):
            with self.subTest(status=status):
                run, post = self.run_model([reply(status)] * 100, limit=6)
                self.assertEqual(post.call_count, 6 * 5)  # each item used all 1 + 4 attempts
                self.assertEqual(run.status, "completed")
                self.assertEqual([i.http_status for i in run.item_results], [status] * 6)
                self.assertTrue(run.unreliable)

    def test_timeouts_never_trigger_it(self):
        slow = requests.exceptions.ReadTimeout("slow")
        run, post = self.run_model([slow] * 100, limit=6)
        self.assertEqual(post.call_count, 6 * 2)  # a read timeout is retried once, per item
        self.assertEqual(run.status, "completed")
        self.assertEqual([i.http_status for i in run.item_results], [None] * 6)
        self.assertTrue(run.unreliable)

    def test_mixed_statuses_do_not_trigger_it(self):
        run, post = self.run_model([reply(404)] * 4 + [reply(401)] * 4, limit=8)
        self.assertEqual((post.call_count, run.status), (8, "completed"))

    def test_four_failures_then_a_success_do_not_trigger_it(self):
        run, post = self.run_model([reply(404)] * 4 + [reply(200)] + [reply(404)] * 3, limit=8)
        self.assertEqual((post.call_count, run.status), (8, "completed"))

    def test_a_success_among_the_first_five_prevents_it(self):
        run, post = self.run_model([reply(200)] + [reply(404)] * 20, limit=8)
        self.assertEqual((post.call_count, run.status), (8, "completed"))

    def test_fewer_than_five_items_cannot_trigger_it(self):
        run, post = self.run_model([reply(404)] * 10, limit=4)
        self.assertEqual((post.call_count, run.status), (4, "completed"))
        self.assertTrue(run.unreliable)

    def test_the_first_five_items_are_counted_across_benchmarks(self):
        run, post = self.run_model([reply(404)] * 20, benchmarks=("custom_byob", "gpqa"), limit=15)  # byob has 2 rows
        self.assertEqual(post.call_count, 5)
        self.assertEqual(run.status, "failed")
        self.assertEqual({k: v.total_samples for k, v in run.benchmark_summaries.items()}, {"custom_byob": 2, "gpqa": 3})

    def test_the_rule_is_per_model(self):
        runner = EvaluationRunner(base_dir=REPO)
        with mock.patch("src.gateway.model_client.requests.post", side_effect=[reply(404)] * 5 + [reply(200)] * 6):
            dead = runner.run_evaluation(DEAD, ["gpqa"], limit_per_benchmark=6, show_progress=False)
            live = runner.run_evaluation(LIVE, ["gpqa"], limit_per_benchmark=6, show_progress=False)
        self.assertEqual((dead.status, live.status), ("failed", "completed"))
        self.assertEqual(len(live.item_results), 6)

    def test_a_healthy_run_is_completed(self):
        run, _ = self.run_model([reply(200)] * 10, limit=5)
        self.assertEqual((run.status, run.failure_reason), ("completed", None))


class TestStatusIsRecorded(LiveTestCase):
    def test_the_client_reports_the_http_status_of_failures(self):
        client = UnifiedModelClient(LIVE, InterceptorProxy())
        cases = [(404, [reply(404)]), (401, [reply(401)]), (429, [reply(429)] * 5), (503, [reply(503)] * 5)]
        for status, responses in cases:
            with self.subTest(status=status), mock.patch("src.gateway.model_client.requests.post", side_effect=responses):
                self.assertEqual(client.generate(f"p{status}")["http_status"], status)
        with mock.patch("src.gateway.model_client.requests.post", side_effect=[requests.exceptions.ReadTimeout("t")] * 2):
            self.assertIsNone(client.generate("timeout")["http_status"])

    def test_successful_items_have_no_status(self):
        run, _ = self.run_model([reply(200)] * 3, limit=3)
        self.assertEqual([i.http_status for i in run.item_results], [None] * 3)


class TestFailedRunHandling(LiveTestCase):
    def setUp(self):
        super().setUp()
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(gc.collect)

    def test_a_failed_run_is_saved_as_json_but_kept_out_of_the_run_list_and_leaderboard(self):
        store = RunStore(os.path.join(self.tmp.name, "runs"))
        run, _ = self.run_model([reply(404)] * 20)
        path = store.save_run(run)
        self.assertTrue(os.path.isfile(path))
        self.assertEqual(store.list_runs(), [])
        self.assertEqual(store.get_leaderboard(), [])
        loaded = store.get_run(run.run_id)
        self.assertEqual((loaded.status, loaded.failure_reason), ("failed", run.failure_reason))
        self.assertFalse(loaded.unreliable)

    def test_a_completed_run_is_still_registered(self):
        store = RunStore(os.path.join(self.tmp.name, "runs"))
        run, _ = self.run_model([reply(200)] * 5, limit=5)
        store.save_run(run)
        self.assertEqual(len(store.list_runs()), 1)

    def test_old_run_files_without_the_new_fields_load_as_completed(self):
        run, _ = self.run_model([reply(200)] * 3, limit=3)
        data = json.loads(run.model_dump_json())
        for key in ("status", "failure_reason", "unreliable", "unreliable_fraction"):
            data.pop(key, None)
        for item in data["item_results"]:
            item.pop("http_status", None)
        old = RunArtifact.model_validate(data)
        self.assertEqual((old.status, old.failure_reason, old.is_failed), ("completed", None, False))

    def test_a_report_that_includes_a_failed_run_says_so(self):
        failed, _ = self.run_model([reply(404)] * 20)
        ok, _ = self.run_model([reply(200)] * 5, limit=5)
        md = ReportGenerator.generate_markdown(DiffEngine.compare_runs(failed, ok))
        self.assertIn("FAILED RUN", md)
        md2 = ReportGenerator.generate_markdown(DiffEngine.compare_runs(ok, ok))
        self.assertNotIn("FAILED RUN", md2)


class TestCommandLine(unittest.TestCase):
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
                data = {**data, "models": {**data.get("models", {}), LIVE["id"]: LIVE, DEAD["id"]: DEAD}}
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

    def run_cli(self, models, responses):
        path = os.path.join(self.tmp.name, "eval_t.yaml")
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump({"job_name": "real_job", "seed": 42, "limit_per_benchmark": 8,
                            "models": models, "benchmarks": ["gpqa"]}, f)
        with mock.patch("src.gateway.model_client.requests.post", side_effect=responses) as post:
            result = CliRunner().invoke(nel.cli, ["eval", "run", "--config", path])
        return result.exit_code, self.buf.getvalue(), post

    def test_a_dead_model_is_reported_clearly_and_the_other_model_still_runs(self):
        code, out, post = self.run_cli([DEAD["id"], LIVE["id"]], [reply(404)] * 5 + [reply(200)] * 8)
        self.assertEqual(code, 1)
        self.assertIn("RUN FAILED", out)
        self.assertIn("HTTP 404", out)
        self.assertIn("vendor/dead-model", out)
        self.assertNotIn("UNRELIABLE RUN", out)               # failed, not unreliable
        self.assertIn("Completed run", out)                   # the second model ran to the end
        self.assertIn("1 model(s) FAILED", out)
        self.assertEqual(post.call_count, 5 + 8)
        self.assertEqual(len(self.store.list_runs()), 1)      # only the completed run is registered
        self.assertEqual(len([f for f in os.listdir(self.store.runs_dir) if f.endswith(".json")]), 2)  # both JSON records exist

    def test_compare_and_gate_refuse_a_failed_run(self):
        self.run_cli([DEAD["id"], LIVE["id"]], [reply(404)] * 5 + [reply(200)] * 8)
        ok_id = self.store.list_runs()[0]["run_id"]
        failed_file = [f for f in os.listdir(self.store.runs_dir) if f.endswith(".json") and f[:-5] != ok_id][0]
        failed_id = failed_file[:-5]
        for args in (["compare", "--run-a", failed_id, "--run-b", ok_id], ["gate", "--baseline", ok_id, "--candidate", failed_id]):
            self.buf.truncate(0); self.buf.seek(0)
            result = CliRunner().invoke(nel.cli, args)
            self.assertEqual(result.exit_code, 1, args)
            self.assertIn("FAILED and cannot be used", self.buf.getvalue())
            self.assertIn("HTTP 404", self.buf.getvalue())

    def test_a_healthy_run_exits_zero(self):
        code, out, _ = self.run_cli([LIVE["id"]], [reply(200)] * 8)
        self.assertEqual(code, 0, out)
        self.assertNotIn("RUN FAILED", out)


if __name__ == "__main__":
    unittest.main()
