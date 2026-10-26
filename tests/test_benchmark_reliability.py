import gc
import io
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
from src.engine.runner import EvaluationRunner
from src.gateway.model_client import UnifiedModelClient
from src.reporting.report_generator import ReportGenerator
from src.storage.run_store import RunStore
from src.storage.schema import UNRELIABLE_BENCHMARK_THRESHOLD, UNRELIABLE_THRESHOLD, BenchmarkSummary, RunArtifact

KEY = "sentinel-nvidia-key-bench-reliability"
LIVE = {"id": "vendor/live-model", "name": "Live Model", "provider": "nvidia_nim"}
MOCK = {"id": "meta/llama-3.1-8b-instruct", "name": "T", "provider": "mock"}


def summary(total, truncated=0, no_answer=0, api_error=0, bench="mmlu_pro"):
    return BenchmarkSummary(benchmark_id=bench, benchmark_name=bench, domain="General", total_samples=total,
                            correct_samples=0, truncated_count=truncated, no_answer_count=no_answer,
                            api_error_count=api_error, accuracy=0.0)


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


def simulated_run(benchmarks=("mmlu_pro", "gsm8k"), n=10, job="standin_t"):
    return EvaluationRunner(base_dir=REPO).run_evaluation(MOCK, list(benchmarks), job_name=job, limit_per_benchmark=n, show_progress=False)


def with_bench_counts(run, bench_id, **counts):
    """A copy of `run` whose benchmark summary (and run totals) carry the given api_error/truncated/no_answer counts."""
    s = run.benchmark_summaries[bench_id].model_copy(update=counts)
    summaries = {**run.benchmark_summaries, bench_id: s}
    totals = {k: sum(getattr(x, k) for x in summaries.values()) for k in ("truncated_count", "no_answer_count", "api_error_count")}
    return run.model_copy(update={"benchmark_summaries": summaries, **totals})


class TestThreshold(unittest.TestCase):
    def test_the_benchmark_threshold_is_twenty_percent_and_the_run_threshold_stays_ten(self):
        self.assertEqual(UNRELIABLE_BENCHMARK_THRESHOLD, 0.20)
        self.assertEqual(UNRELIABLE_THRESHOLD, 0.10)

    def test_exactly_twenty_percent_is_ok_and_more_is_unreliable(self):
        cases = [(15, 3, False), (15, 4, True), (10, 2, False), (10, 3, True), (5, 1, False), (5, 2, True),
                 (100, 20, False), (100, 21, True), (1, 0, False), (1, 1, True)]
        for total, flagged, expected in cases:
            with self.subTest(total=total, flagged=flagged):
                s = summary(total, api_error=flagged)
                self.assertEqual(s.unreliable, expected)
                self.assertAlmostEqual(s.unreliable_fraction, round(flagged / total, 4))

    def test_all_three_kinds_count_toward_the_same_total(self):
        self.assertFalse(summary(15, truncated=1, no_answer=1, api_error=1).unreliable)   # 3/15 = 20%
        self.assertTrue(summary(15, truncated=2, no_answer=1, api_error=1).unreliable)    # 4/15
        self.assertTrue(summary(10, truncated=3).unreliable)
        self.assertTrue(summary(10, no_answer=3).unreliable)
        self.assertTrue(summary(10, api_error=3).unreliable)

    def test_an_empty_benchmark_is_not_unreliable(self):
        s = summary(0)
        self.assertFalse(s.unreliable)
        self.assertEqual(s.unreliable_fraction, 0.0)

    def test_the_flag_is_written_to_the_run_file(self):
        data = summary(10, api_error=5).model_dump()
        self.assertIs(data["unreliable"], True)
        self.assertEqual(data["unreliable_fraction"], 0.5)


class TestRunVersusBenchmark(unittest.TestCase):
    def test_a_benchmark_can_be_unreliable_while_the_whole_run_is_under_ten_percent(self):
        # 100 clean items in one benchmark, 10 items of which 4 failed in another: 4 of 110 = 3.6% overall, 40% in the second
        run = simulated_run(("mmlu_pro", "gsm8k"), n=10)
        big = run.benchmark_summaries["gsm8k"].model_copy(update={"total_samples": 100, "correct_samples": 50})
        run = run.model_copy(update={"benchmark_summaries": {**run.benchmark_summaries, "gsm8k": big}})
        run = with_bench_counts(run, "mmlu_pro", api_error_count=4)
        padded = run.model_copy(update={"item_results": run.item_results + run.item_results[:10] * 9})  # 110 items in total
        self.assertEqual(len(padded.item_results), 110)
        self.assertLess(padded.unreliable_fraction, UNRELIABLE_THRESHOLD)
        self.assertFalse(padded.unreliable)
        self.assertEqual(padded.unreliable_benchmarks, ["mmlu_pro"])

    def test_a_clean_run_has_no_unreliable_benchmarks(self):
        run = simulated_run()
        self.assertEqual(run.unreliable_benchmarks, [])
        self.assertFalse(any(s.unreliable for s in run.benchmark_summaries.values()))

    def test_several_benchmarks_are_listed_in_run_order(self):
        run = simulated_run(("mmlu_pro", "gsm8k", "gpqa"))
        run = with_bench_counts(run, "mmlu_pro", no_answer_count=5)
        run = with_bench_counts(run, "gpqa", truncated_count=9)
        self.assertEqual(run.unreliable_benchmarks, ["mmlu_pro", "gpqa"])

    def test_a_failed_or_incomplete_run_reports_no_unreliable_benchmarks(self):
        run = with_bench_counts(simulated_run(), "mmlu_pro", api_error_count=9)
        self.assertEqual(run.unreliable_benchmarks, ["mmlu_pro"])
        for status in ("failed", "incomplete"):
            with self.subTest(status=status):
                self.assertEqual(run.model_copy(update={"status": status}).unreliable_benchmarks, [])

    def test_the_run_file_records_the_unreliable_benchmarks(self):
        run = with_bench_counts(simulated_run(), "mmlu_pro", api_error_count=9)
        data = RunArtifact.model_validate_json(run.model_dump_json())
        self.assertEqual(data.unreliable_benchmarks, ["mmlu_pro"])
        self.assertIn('"unreliable_benchmarks"', run.model_dump_json())


class TestRunnerComputesIt(unittest.TestCase):
    def setUp(self):
        for p in (mock.patch.dict(os.environ, {"NVIDIA_API_KEY": KEY}), mock.patch.object(UnifiedModelClient, "_sleep")):
            p.start()
            self.addCleanup(p.stop)

    def run_live(self, responses, benchmarks, limit=15):
        with mock.patch("src.gateway.model_client.requests.post", side_effect=responses):
            return EvaluationRunner(base_dir=REPO).run_evaluation(
                LIVE, list(benchmarks), job_name="real_job", limit_per_benchmark=limit, show_progress=False)

    def test_four_failed_calls_in_fifteen_flag_only_that_benchmark_even_though_the_run_is_under_ten_percent(self):
        # 400 is not retried and not an early-stop status: one request per failed item. 4 of 45 items = 8.9% < 10%.
        run = self.run_live([reply(400)] * 4 + [reply(200)] * 11 + [reply(200)] * 30, ["gpqa", "mmlu_pro", "truthfulqa"])
        self.assertEqual(run.status, "completed")
        self.assertEqual(run.benchmark_summaries["gpqa"].api_error_count, 4)
        self.assertTrue(run.benchmark_summaries["gpqa"].unreliable)
        self.assertFalse(run.benchmark_summaries["mmlu_pro"].unreliable)
        self.assertFalse(run.benchmark_summaries["truthfulqa"].unreliable)
        self.assertEqual(run.unreliable_benchmarks, ["gpqa"])
        self.assertLess(run.unreliable_fraction, UNRELIABLE_THRESHOLD)
        self.assertFalse(run.unreliable)   # the run-level flag alone would have hidden the clustering in gpqa

    def test_three_failed_calls_in_fifteen_are_exactly_twenty_percent_and_not_flagged(self):
        run = self.run_live([reply(400)] * 3 + [reply(200)] * 12, ["gpqa"])
        self.assertEqual(run.benchmark_summaries["gpqa"].api_error_count, 3)
        self.assertEqual(run.unreliable_benchmarks, [])


class Capture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(gc.collect)
        cwd = os.getcwd()
        os.chdir(REPO)
        self.addCleanup(os.chdir, cwd)
        self.buf = io.StringIO()
        self.store = RunStore(os.path.join(self.tmp.name, "runs"))
        for p in (mock.patch.object(nel, "RunStore", lambda _p: self.store),
                  mock.patch.object(nel, "console", Console(file=self.buf, width=250, force_terminal=False))):
            p.start()
            self.addCleanup(p.stop)

    def out(self):
        return self.buf.getvalue()

    def clear(self):
        self.buf.truncate(0)
        self.buf.seek(0)


class TestCommandLine(Capture):
    def flagged_pair(self):
        a = with_bench_counts(simulated_run(), "mmlu_pro", api_error_count=4)
        b = simulated_run()
        self.store.save_run(a)
        self.store.save_run(b)
        return a, b

    def test_the_compare_panel_and_table_name_the_unreliable_benchmark(self):
        a, b = self.flagged_pair()
        result = CliRunner().invoke(nel.cli, ["compare", "--run-a", a.run_id, "--run-b", b.run_id])
        self.assertEqual(result.exit_code, 0, result.output)
        out = self.out()
        self.assertIn("Unreliable benchmarks (over 20% of their items flagged): A=mmlu_pro; B=none", out)
        self.assertIn("A UNRELIABLE (40%)", out)

    def test_a_clean_pair_says_none_and_ok(self):
        a, b = simulated_run(), simulated_run()
        self.store.save_run(a)
        self.store.save_run(b)
        CliRunner().invoke(nel.cli, ["compare", "--run-a", a.run_id, "--run-b", b.run_id])
        self.assertIn("A=none; B=none", self.out())
        self.assertNotIn("A UNRELIABLE", self.out())

    def test_the_markdown_report_flags_the_benchmark_in_banner_and_table(self):
        a, b = simulated_run(), with_bench_counts(simulated_run(), "gsm8k", truncated_count=5)
        md = ReportGenerator.generate_markdown(DiffEngine.compare_runs(a, b))
        self.assertIn("UNRELIABLE BENCHMARK(S): A: none; B: gsm8k", md)
        row = [line for line in md.splitlines() if line.startswith("| **") and "gsm8k" in line.lower()]
        self.assertTrue(row)
        self.assertIn("B UNRELIABLE (50%)", row[0])
        other = [line for line in md.splitlines() if line.startswith("| **") and "mmlu" in line.lower()]
        self.assertTrue(other and other[0].rstrip().endswith("| ok |"), other)

    def test_the_html_report_carries_it_too(self):
        a, b = simulated_run(), with_bench_counts(simulated_run(), "gsm8k", truncated_count=5)
        html = ReportGenerator.generate_html(DiffEngine.compare_runs(a, b))
        self.assertIn("UNRELIABLE BENCHMARK(S)", html)

    def test_the_report_has_no_benchmark_banner_for_a_clean_pair(self):
        md = ReportGenerator.generate_markdown(DiffEngine.compare_runs(simulated_run(), simulated_run()))
        self.assertNotIn("UNRELIABLE BENCHMARK(S)", md)
        self.assertIn("| Reliability |", md)

    def test_the_existing_run_level_report_lines_are_unchanged(self):
        md = ReportGenerator.generate_markdown(DiffEngine.compare_runs(simulated_run(), simulated_run()))
        self.assertIn("Reliability: A = ok (0% of items api_error/truncated/no_answer_found), B = ok (0%)", md)

    def test_the_diff_carries_the_flags_per_benchmark(self):
        a = with_bench_counts(simulated_run(), "mmlu_pro", no_answer_count=6)
        comp = DiffEngine.compare_runs(a, simulated_run())
        self.assertTrue(comp["benchmarks"]["mmlu_pro"]["unreliable_a"])
        self.assertFalse(comp["benchmarks"]["mmlu_pro"]["unreliable_b"])
        self.assertFalse(comp["benchmarks"]["gsm8k"]["unreliable_a"])
        self.assertEqual(comp["unreliable_benchmarks_a"], ["mmlu_pro"])
        self.assertEqual(comp["unreliable_benchmarks_b"], [])


class TestRunCommandOutput(Capture):
    def setUp(self):
        super().setUp()
        for p in (mock.patch.dict(os.environ, {"NVIDIA_API_KEY": KEY}), mock.patch.object(UnifiedModelClient, "_sleep")):
            p.start()
            self.addCleanup(p.stop)
        real_load_yaml = nel.load_yaml

        def load_yaml(path):
            data = real_load_yaml(path)
            if str(path).replace("\\", "/").endswith("configs/models_config.yaml"):
                data = {**data, "models": {**data.get("models", {}), LIVE["id"]: LIVE}}
            return data

        p = mock.patch.object(nel, "load_yaml", load_yaml)
        p.start()
        self.addCleanup(p.stop)

    def run_cli(self, responses, benchmarks=("gpqa", "mmlu_pro")):
        path = os.path.join(self.tmp.name, "eval_t.yaml")
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump({"job_name": "real_job", "seed": 42, "limit_per_benchmark": 15, "models": [LIVE["id"]],
                            "benchmarks": list(benchmarks)}, f)
        with mock.patch("src.gateway.model_client.requests.post", side_effect=responses):
            return CliRunner().invoke(nel.cli, ["eval", "run", "--config", path])

    def test_the_run_table_marks_the_unreliable_benchmark_even_though_the_run_is_under_ten_percent(self):
        # gpqa: 4 of 15 failed (27%); the run: 4 of 45 = 8.9%
        result = self.run_cli([reply(400)] * 4 + [reply(200)] * 11 + [reply(200)] * 30, benchmarks=("gpqa", "mmlu_pro", "truthfulqa"))
        self.assertEqual(result.exit_code, 0, self.out())
        out = self.out()
        self.assertIn("Reliability", out)
        self.assertIn("UNRELIABLE (27%)", out)
        self.assertIn("UNRELIABLE BENCHMARK(S): gpqa", out)
        self.assertNotIn("UNRELIABLE RUN", out)

    def test_a_clean_run_has_no_benchmark_warning(self):
        result = self.run_cli([reply(200)] * 30)
        self.assertEqual(result.exit_code, 0, self.out())
        self.assertNotIn("UNRELIABLE BENCHMARK(S)", self.out())
        self.assertNotIn("UNRELIABLE (", self.out())


if __name__ == "__main__":
    unittest.main()
