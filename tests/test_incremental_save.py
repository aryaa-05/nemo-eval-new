import gc
import io
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

import requests
from click.testing import CliRunner
from rich.console import Console

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)

import src.cli.nel as nel
from src.comparison.diff_engine import DiffEngine
from src.engine.runner import EvaluationRunner, incomplete_reason
from src.gateway.model_client import UnifiedModelClient
from src.reporting.report_generator import ReportGenerator
from src.storage.run_store import RunStore
from src.storage.schema import RunArtifact

MOCK = {"id": "meta/llama-3.1-8b-instruct", "name": "T", "provider": "mock"}
BENCHES = ["gpqa", "mmlu_pro", "gsm8k"]
REAL_GENERATE = UnifiedModelClient.generate


def reply_404():
    r = mock.MagicMock()
    r.status_code = 404
    r.headers = {}
    r.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Client Error: Synthetic")
    return r


class SimulatedCrash(RuntimeError):
    pass


def crash_on_call(n):
    """A replacement for UnifiedModelClient.generate that works for n-1 calls and raises on call number n."""
    calls = {"count": 0}

    def generate(self, *args, **kwargs):
        calls["count"] += 1
        if calls["count"] == n:
            raise SimulatedCrash(f"simulated crash on call {n}")
        return REAL_GENERATE(self, *args, **kwargs)

    return generate


def run(benchmarks=BENCHES, limit=4, checkpoint=None, job="standin_t"):
    return EvaluationRunner(base_dir=REPO).run_evaluation(
        MOCK, list(benchmarks), job_name=job, limit_per_benchmark=limit, show_progress=False, checkpoint=checkpoint)


class TestRunnerCheckpoints(unittest.TestCase):
    def test_a_checkpoint_follows_every_benchmark_except_the_last(self):
        partials = []
        final = run(checkpoint=partials.append)
        self.assertEqual([list(p.benchmark_summaries) for p in partials], [["gpqa"], ["gpqa", "mmlu_pro"]])
        self.assertEqual(final.status, "completed")
        self.assertEqual(list(final.benchmark_summaries), BENCHES)

    def test_a_checkpoint_is_an_incomplete_snapshot_of_the_same_run(self):
        partials = []
        final = run(checkpoint=partials.append)
        for p in partials:
            with self.subTest(done=len(p.benchmark_summaries)):
                self.assertEqual(p.run_id, final.run_id)
                self.assertEqual(p.status, "incomplete")
                self.assertTrue(p.is_incomplete)
                self.assertFalse(p.is_usable)
                self.assertTrue(p.failure_reason.startswith("INCOMPLETE"))
                self.assertEqual(p.execution_params["benchmarks"], BENCHES)   # what was planned, so the gap is visible
                self.assertEqual(len(p.item_results), sum(s.total_samples for s in p.benchmark_summaries.values()))
                self.assertEqual(p.job_name, final.job_name)
                self.assertEqual(p.run_mode, final.run_mode)
                self.assertEqual(p.scorer_version, final.scorer_version)
                self.assertFalse(p.unreliable)             # INCOMPLETE is reported, never UNRELIABLE
                self.assertEqual(p.unreliable_benchmarks, [])

    def test_a_checkpoint_is_not_changed_by_later_benchmarks(self):
        partials = []
        run(checkpoint=partials.append)
        self.assertEqual([len(p.item_results) for p in partials], [4, 8])

    def test_the_reason_names_what_is_done_and_what_is_missing(self):
        msg = incomplete_reason(["gpqa"], BENCHES)
        self.assertIn("1 of 3 benchmarks (gpqa)", msg)
        self.assertIn("mmlu_pro, gsm8k", msg)
        self.assertIn("cannot be compared or gated", msg)

    def test_no_checkpoint_for_a_single_benchmark_run(self):
        partials = []
        final = run(benchmarks=["gpqa"], checkpoint=partials.append)
        self.assertEqual((partials, final.status), ([], "completed"))

    def test_running_without_a_callback_still_works(self):
        self.assertEqual(run().status, "completed")

    def test_an_early_stopped_run_is_failed_and_makes_no_checkpoint(self):
        partials = []
        with mock.patch.dict(os.environ, {"NVIDIA_API_KEY": "sentinel-key-incremental"}), \
                mock.patch.object(UnifiedModelClient, "_sleep"), mock.patch("src.gateway.model_client.requests.post") as post:
            post.side_effect = [reply_404() for _ in range(30)]
            failed = EvaluationRunner(base_dir=REPO).run_evaluation(
                {"id": "vendor/dead", "name": "Dead", "provider": "nvidia_nim"}, BENCHES, job_name="real_job",
                limit_per_benchmark=8, show_progress=False, checkpoint=partials.append)
        self.assertEqual((failed.status, partials), ("failed", []))


class TestSimulatedCrashInTheRunner(unittest.TestCase):
    def crash(self, call_number, limit=4):
        partials = []
        with mock.patch.object(UnifiedModelClient, "generate", crash_on_call(call_number)):
            with self.assertRaises(SimulatedCrash):
                run(limit=limit, checkpoint=partials.append)
        return partials

    def test_a_crash_in_the_second_benchmark_keeps_the_first(self):
        partials = self.crash(call_number=6)   # 4 items in gpqa, then the 2nd call of mmlu_pro
        self.assertEqual(len(partials), 1)
        kept = partials[0]
        self.assertEqual(list(kept.benchmark_summaries), ["gpqa"])
        self.assertEqual(len(kept.item_results), 4)
        self.assertEqual(kept.status, "incomplete")
        self.assertEqual({i.benchmark for i in kept.item_results}, {"gpqa"})   # nothing from the benchmark that crashed

    def test_a_crash_in_the_third_benchmark_keeps_the_first_two(self):
        partials = self.crash(call_number=10)
        self.assertEqual(list(partials[-1].benchmark_summaries), ["gpqa", "mmlu_pro"])
        self.assertEqual(len(partials[-1].item_results), 8)

    def test_a_crash_in_the_first_benchmark_leaves_nothing_to_keep(self):
        self.assertEqual(self.crash(call_number=2), [])

    def test_a_crash_exactly_at_a_benchmark_boundary_keeps_the_finished_benchmark(self):
        partials = self.crash(call_number=5)   # the very first call of mmlu_pro
        self.assertEqual(list(partials[-1].benchmark_summaries), ["gpqa"])

    def test_the_kept_part_has_the_same_scores_a_normal_run_gives(self):
        normal = run(benchmarks=["gpqa"])
        partials = self.crash(call_number=6)
        self.assertEqual(partials[0].benchmark_summaries["gpqa"].accuracy, normal.benchmark_summaries["gpqa"].accuracy)
        self.assertEqual(partials[0].benchmark_summaries["gpqa"].correct_samples, normal.benchmark_summaries["gpqa"].correct_samples)


class TestStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(gc.collect)
        self.store = RunStore(os.path.join(self.tmp.name, "runs"))

    def checkpoint(self):
        partials = []
        final = run(checkpoint=partials.append)
        return partials[-1], final

    def test_a_checkpoint_is_saved_as_json_but_kept_out_of_the_run_list_and_leaderboard(self):
        partial, _ = self.checkpoint()
        path = self.store.save_checkpoint(partial)
        self.assertTrue(os.path.isfile(path))
        self.assertEqual((self.store.list_runs(), self.store.get_leaderboard()), ([], []))

    def test_an_incomplete_record_loads_back_as_incomplete(self):
        partial, _ = self.checkpoint()
        self.store.save_checkpoint(partial)
        loaded = self.store.get_run(partial.run_id)
        self.assertTrue(loaded.is_incomplete)
        self.assertEqual(json.loads(loaded.model_dump_json())["status"], "incomplete")

    def test_list_unusable_describes_incomplete_and_failed_records(self):
        partial, _ = self.checkpoint()
        self.store.save_checkpoint(partial)
        failed = run(benchmarks=["gpqa"]).model_copy(update={"run_id": "run_failed_x", "status": "failed", "failure_reason": "HTTP 404 ..."})
        self.store.save_run(failed)
        rows = {r["run_id"]: r for r in self.store.list_unusable()}
        self.assertEqual(set(rows), {partial.run_id, "run_failed_x"})
        inc = rows[partial.run_id]
        self.assertEqual((inc["status"], inc["benchmarks_done"], inc["planned"]), ("incomplete", ["gpqa", "mmlu_pro"], BENCHES))
        self.assertTrue(inc["reason"].startswith("INCOMPLETE"))
        self.assertEqual(rows["run_failed_x"]["status"], "failed")

    def test_list_unusable_ignores_completed_runs_and_junk_files(self):
        done = run(benchmarks=["gpqa"])
        self.store.save_run(done)
        with open(os.path.join(self.store.runs_dir, "broken.json"), "w", encoding="utf-8") as f:
            f.write("{ not json")
        with open(os.path.join(self.store.runs_dir, "list.json"), "w", encoding="utf-8") as f:
            f.write("[1, 2, 3]")
        with open(os.path.join(self.store.runs_dir, "notes.txt"), "w", encoding="utf-8") as f:
            f.write("hello")
        self.assertEqual(self.store.list_unusable(), [])

    def test_the_final_save_replaces_the_checkpoint_and_registers_the_run(self):
        partial, final = self.checkpoint()
        self.store.save_checkpoint(partial)
        self.assertEqual(len(self.store.list_unusable()), 1)
        self.store.save_run(final)
        self.assertEqual(self.store.list_unusable(), [])
        self.assertEqual([r["run_id"] for r in self.store.list_runs()], [final.run_id])
        self.assertEqual(self.store.get_run(final.run_id).status, "completed")
        self.assertEqual(len([f for f in os.listdir(self.store.runs_dir) if f.endswith(".json")]), 1)

    def test_register_false_keeps_a_completed_run_out_of_the_list(self):
        done = run(benchmarks=["gpqa"])
        self.store.save_run(done, register=False)
        self.assertEqual(self.store.list_runs(), [])
        self.assertEqual(self.store.get_run(done.run_id).status, "completed")


class TestIncompleteRunsInComparisons(unittest.TestCase):
    def test_the_report_says_incomplete(self):
        partials = []
        final = run(checkpoint=partials.append)
        md = ReportGenerator.generate_markdown(DiffEngine.compare_runs(partials[-1], final))
        self.assertIn("INCOMPLETE RUN", md)
        self.assertNotIn("INCOMPLETE RUN", ReportGenerator.generate_markdown(DiffEngine.compare_runs(final, final)))

    def test_the_diff_exposes_the_flags(self):
        partials = []
        final = run(checkpoint=partials.append)
        comp = DiffEngine.compare_runs(partials[0], final)
        self.assertEqual((comp["incomplete_a"], comp["incomplete_b"]), (True, False))


class CliCase(unittest.TestCase):
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

    def eval_run(self, model="sim/crash-model", benches=BENCHES, limit=4):
        args = ["eval", "run", "--config", "configs/eval_config_default.yaml", "--model", model, "--limit", str(limit),
                "--job-name", "crash_test"]
        for b in benches:
            args += ["--bench", b]
        return CliRunner().invoke(nel.cli, args)

    def json_files(self):
        return sorted(f[:-5] for f in os.listdir(self.store.runs_dir) if f.endswith(".json"))


class TestCliCrash(CliCase):
    def test_a_crash_leaves_an_incomplete_record_and_says_so(self):
        with mock.patch.object(UnifiedModelClient, "generate", crash_on_call(6)):
            result = self.eval_run()
        self.assertEqual(result.exit_code, 1)
        self.assertIsInstance(result.exception, SimulatedCrash)
        self.assertIn("RUN INTERRUPTED (SimulatedCrash)", self.out())
        self.assertIn("1 benchmark(s) finished before it are saved as an INCOMPLETE record", self.out())
        self.assertNotIn("Completed run", self.out())
        ids = self.json_files()
        self.assertEqual(len(ids), 1)
        kept = self.store.get_run(ids[0])
        self.assertTrue(kept.is_incomplete)
        self.assertEqual(list(kept.benchmark_summaries), ["gpqa"])
        self.assertEqual(self.store.list_runs(), [])   # never registered

    def test_a_keyboard_interrupt_is_handled_the_same_way(self):
        def interrupt(self, *a, **k):
            interrupt.calls += 1
            if interrupt.calls == 9:
                raise KeyboardInterrupt()
            return REAL_GENERATE(self, *a, **k)

        interrupt.calls = 0
        with mock.patch.object(UnifiedModelClient, "generate", interrupt):
            result = self.eval_run()
        self.assertEqual(result.exit_code, 1)
        self.assertIn("RUN INTERRUPTED (KeyboardInterrupt)", self.out())
        self.assertIn("2 benchmark(s) finished", self.out())
        self.assertEqual(list(self.store.get_run(self.json_files()[0]).benchmark_summaries), ["gpqa", "mmlu_pro"])

    def test_a_crash_before_any_benchmark_finishes_saves_nothing_and_claims_nothing(self):
        with mock.patch.object(UnifiedModelClient, "generate", crash_on_call(2)):
            result = self.eval_run()
        self.assertEqual(result.exit_code, 1)
        self.assertNotIn("INCOMPLETE", self.out())
        self.assertEqual(self.json_files(), [])

    def test_list_shows_the_incomplete_record_apart_from_the_run_list(self):
        with mock.patch.object(UnifiedModelClient, "generate", crash_on_call(10)):
            self.eval_run()
        incomplete_id = self.json_files()[0]
        self.clear()
        result = CliRunner().invoke(nel.cli, ["list"])
        self.assertEqual(result.exit_code, 0, self.out())
        out = self.out()
        self.assertIn("INCOMPLETE or FAILED run records (1)", out)
        self.assertIn(incomplete_id, out)
        self.assertIn("INCOMPLETE", out)
        self.assertIn("2 of 3", out)
        self.assertIn("Saved Evaluation Runs (0 runs)", out)

    def test_compare_and_gate_refuse_an_incomplete_run(self):
        good = run(benchmarks=BENCHES, job="crash_test")
        self.store.save_run(good)
        with mock.patch.object(UnifiedModelClient, "generate", crash_on_call(6)):
            self.eval_run()
        bad_id = [i for i in self.json_files() if i != good.run_id][0]
        for args in (["compare", "--run-a", bad_id, "--run-b", good.run_id],
                     ["compare", "--run-a", good.run_id, "--run-b", bad_id],
                     ["gate", "--baseline", good.run_id, "--candidate", bad_id],
                     ["gate", "--baseline", bad_id, "--candidate", good.run_id]):
            with self.subTest(args=args[0] + ":" + args[2][:8]):
                self.clear()
                result = CliRunner().invoke(nel.cli, args)
                self.assertEqual(result.exit_code, 1)
                self.assertIn(f"run {bad_id} is INCOMPLETE and cannot be used here", self.out())
                self.assertNotIn("McNemar", self.out())
                self.assertNotIn("GO", self.out().split("cannot be used here")[-1].split())  # no verdict was printed

    def test_a_normal_run_leaves_one_completed_record_and_no_incomplete_one(self):
        result = self.eval_run()
        self.assertEqual(result.exit_code, 0, self.out())
        self.assertEqual(len(self.json_files()), 1)    # the checkpoints were overwritten by the final save
        self.assertEqual(len(self.store.list_runs()), 1)
        self.assertEqual(self.store.list_unusable(), [])
        self.assertEqual(self.store.get_run(self.json_files()[0]).status, "completed")
        self.assertNotIn("INCOMPLETE", self.out())

    def test_a_rerun_after_a_crash_gives_a_new_complete_run_and_the_old_record_stays_incomplete(self):
        with mock.patch.object(UnifiedModelClient, "generate", crash_on_call(6)):
            self.eval_run()
        crashed_id = self.json_files()[0]
        result = self.eval_run()
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(len(self.json_files()), 2)
        self.assertEqual([r["run_id"] for r in self.store.list_unusable()], [crashed_id])
        self.assertEqual(len(self.store.list_runs()), 1)


if __name__ == "__main__":
    unittest.main()
