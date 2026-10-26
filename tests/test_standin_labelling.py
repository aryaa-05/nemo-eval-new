import gc
import io
import os
import re
import sys
import tempfile
import unittest
from unittest import mock

import yaml
from click.testing import CliRunner
from rich.console import Console

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)

import src.cli.nel as nel
from src.comparison.diff_engine import DiffEngine
from src.engine.runner import EvaluationRunner
from src.reporting.report_generator import ReportGenerator
from src.storage.run_store import RunStore
from src.storage.schema import STANDIN_JOB_PREFIX, RunArtifact

MOCK_MODEL = {"id": "meta/llama-3.1-70b-instruct", "name": "Test Model", "provider": "mock"}


def _run(job_name):
    return EvaluationRunner(base_dir=REPO).run_evaluation(
        MOCK_MODEL, ["mmlu_pro"], job_name=job_name, limit_per_benchmark=2, show_progress=False
    )


class TestArtifactAndReport(unittest.TestCase):
    def test_job_name_is_recorded_in_the_artifact_and_survives_a_round_trip(self):
        run = _run("standin_demo")
        self.assertEqual(run.job_name, "standin_demo")
        self.assertEqual(RunArtifact.model_validate_json(run.model_dump_json()).job_name, "standin_demo")

    def test_is_standin_follows_the_job_name_prefix(self):
        self.assertEqual(STANDIN_JOB_PREFIX, "standin_")
        self.assertTrue(_run("standin_x").is_standin)
        self.assertTrue(_run("STANDIN_x").is_standin)
        self.assertFalse(_run("pair1_real_models").is_standin)
        self.assertFalse(_run("not_standin_at_start").is_standin)

    def test_comparison_carries_job_names_and_flags(self):
        comp = DiffEngine.compare_runs(_run("standin_a"), _run("real_b"))
        self.assertEqual((comp["job_name_a"], comp["job_name_b"]), ("standin_a", "real_b"))
        self.assertEqual((comp["standin_a"], comp["standin_b"]), (True, False))

    def test_report_header_shows_job_names_and_the_standin_banner(self):
        comp = DiffEngine.compare_runs(_run("standin_a"), _run("standin_b"))
        for text in (ReportGenerator.generate_markdown(comp), ReportGenerator.generate_html(comp)):
            self.assertIn("Job: A = standin_a, B = standin_b", text)
            self.assertIn("STAND-IN RUN", text)
            self.assertIn("NOT a result for the real models", text)

    def test_banner_appears_when_only_one_side_is_a_standin(self):
        md = ReportGenerator.generate_markdown(DiffEngine.compare_runs(_run("real_a"), _run("standin_b")))
        self.assertIn("STAND-IN RUN", md)

    def test_no_banner_for_ordinary_jobs(self):
        md = ReportGenerator.generate_markdown(DiffEngine.compare_runs(_run("real_a"), _run("real_b")))
        self.assertIn("Job: A = real_a, B = real_b", md)
        self.assertNotIn("STAND-IN RUN", md)

    def test_banner_and_run_mode_banner_are_independent(self):
        md = ReportGenerator.generate_markdown(DiffEngine.compare_runs(_run("standin_a"), _run("standin_b")))
        self.assertIn("Run mode: A = SIMULATED", md)
        self.assertIn("SIMULATED RESULTS", md)


class TestCommandLine(unittest.TestCase):
    def setUp(self):
        # RunStore leaves its sqlite connection to the garbage collector, which on Windows keeps the file locked.
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(gc.collect)
        cwd = os.getcwd()
        os.chdir(REPO)  # nel.py reads configs/ relative to the current directory
        self.addCleanup(os.chdir, cwd)
        self.buf = io.StringIO()
        self.store = RunStore(os.path.join(self.tmp.name, "runs"))
        for patcher in (
            mock.patch.object(nel, "RunStore", lambda _path: self.store),
            mock.patch.object(nel, "console", Console(file=self.buf, width=250, force_terminal=False)),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def config(self, filename, job_name):
        path = os.path.join(self.tmp.name, filename)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump({"job_name": job_name, "seed": 42, "limit_per_benchmark": 2,
                            "models": [MOCK_MODEL["id"]], "benchmarks": ["mmlu_pro"]}, f)
        return path

    def invoke(self, *args):
        result = CliRunner().invoke(nel.cli, list(args))
        return result.exit_code, self.buf.getvalue()

    def test_standin_config_runs_and_is_visibly_labelled(self):
        code, out = self.invoke("eval", "run", "--config", self.config("standin_t.yaml", "standin_t"))
        self.assertEqual(code, 0, out)
        self.assertIn("job standin_t", out)
        self.assertIn("STAND-IN RUN", out)
        self.assertEqual(self.store.list_runs()[0]["job_name"], "standin_t")

    def test_job_name_override_cannot_strip_the_standin_label(self):
        code, out = self.invoke("eval", "run", "--config", self.config("standin_t.yaml", "standin_t"), "--job-name", "plain_name")
        self.assertEqual(code, 1)
        self.assertIn("stand-in config", out)
        self.assertEqual(self.store.list_runs(), [])  # nothing ran

    def test_standin_config_with_an_unlabelled_job_name_is_refused(self):
        code, out = self.invoke("eval", "run", "--config", self.config("standin_t.yaml", "pair1"))
        self.assertEqual(code, 1)
        self.assertEqual(self.store.list_runs(), [])

    def test_ordinary_config_is_unaffected(self):
        code, out = self.invoke("eval", "run", "--config", self.config("eval_t.yaml", "real_job"))
        self.assertEqual(code, 0, out)
        self.assertIn("job real_job", out)
        self.assertNotIn("STAND-IN", out)

    def test_compare_panel_and_list_show_the_job(self):
        self.invoke("eval", "run", "--config", self.config("standin_t.yaml", "standin_t"))
        self.invoke("eval", "run", "--config", self.config("eval_t.yaml", "real_job"))
        run_ids = [r["run_id"] for r in self.store.list_runs()]
        self.assertEqual(len(run_ids), 2)
        self.buf.truncate(0); self.buf.seek(0)
        code, out = self.invoke("compare", "--run-a", run_ids[0], "--run-b", run_ids[1])
        self.assertEqual(code, 0, out)
        self.assertTrue(re.search(r"Job: A=\w+, B=\w+", out), out)
        self.assertIn("STAND-IN", out)
        self.buf.truncate(0); self.buf.seek(0)
        code, out = self.invoke("list")
        self.assertEqual(code, 0, out)
        self.assertIn("standin_t", out)
        self.assertIn("real_job", out)


if __name__ == "__main__":
    unittest.main()
