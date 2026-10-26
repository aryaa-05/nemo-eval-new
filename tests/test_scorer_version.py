import gc
import io
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

from click.testing import CliRunner
from rich.console import Console

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)

import src.cli.nel as nel
from src.comparison.diff_engine import DiffEngine
from src.engine.runner import EvaluationRunner
from src.reporting.report_generator import ReportGenerator
from src.scoring.version import SCORER_VERSION, UNVERSIONED
from src.storage.run_store import RunStore
from src.storage.schema import RunArtifact

MOCK = {"id": "meta/llama-3.1-8b-instruct", "name": "T", "provider": "mock"}


def simulated_run(**update):
    run = EvaluationRunner(base_dir=REPO).run_evaluation(MOCK, ["mmlu_pro", "gsm8k"], job_name="standin_t",
                                                         limit_per_benchmark=5, show_progress=False)
    return run.model_copy(update=update) if update else run


def legacy(run):
    """The same run as an artifact saved before the version stamp existed."""
    data = json.loads(run.model_dump_json())
    for key in ("scorer_version", "rescored_from", "rescored_at"):
        data.pop(key, None)
    return RunArtifact.model_validate(data)


class TestVersionConstant(unittest.TestCase):
    def test_constants(self):
        self.assertIsInstance(SCORER_VERSION, str)
        self.assertTrue(SCORER_VERSION)
        self.assertEqual(UNVERSIONED, "unversioned")
        self.assertNotEqual(SCORER_VERSION, UNVERSIONED)


class TestArtifactStamp(unittest.TestCase):
    def test_a_new_run_carries_the_current_version(self):
        run = simulated_run()
        self.assertEqual(run.scorer_version, SCORER_VERSION)
        self.assertEqual(json.loads(run.model_dump_json())["scorer_version"], SCORER_VERSION)

    def test_a_run_file_without_the_field_loads_as_unversioned(self):
        old = legacy(simulated_run())
        self.assertEqual(old.scorer_version, UNVERSIONED)
        self.assertIsNone(old.rescored_from)
        self.assertFalse(old.is_rescored)

    def test_the_stamp_survives_saving_and_loading(self):
        run = simulated_run()
        self.assertEqual(RunArtifact.model_validate_json(run.model_dump_json()).scorer_version, SCORER_VERSION)

    def test_a_rescored_artifact_says_where_it_came_from(self):
        run = simulated_run(rescored_from="run_original", rescored_at="2026-10-20T00:00:00Z")
        self.assertTrue(run.is_rescored)
        again = RunArtifact.model_validate_json(run.model_dump_json())
        self.assertEqual((again.rescored_from, again.rescored_at), ("run_original", "2026-10-20T00:00:00Z"))


class TestReportHeader(unittest.TestCase):
    def md(self, a, b):
        return ReportGenerator.generate_markdown(DiffEngine.compare_runs(a, b))

    def test_the_header_shows_the_version_of_both_runs(self):
        md = self.md(simulated_run(), simulated_run())
        self.assertIn(f"Scorer version: A = {SCORER_VERSION}, B = {SCORER_VERSION}", md)
        self.assertNotIn("different scorer versions", md)
        self.assertNotIn("RESCORED", md)

    def test_an_unversioned_run_is_shown_and_warned_about(self):
        md = self.md(legacy(simulated_run()), simulated_run())
        self.assertIn(f"Scorer version: A = {UNVERSIONED}, B = {SCORER_VERSION}", md)
        self.assertIn("different scorer versions", md)

    def test_two_unversioned_runs_are_not_a_mismatch(self):
        md = self.md(legacy(simulated_run()), legacy(simulated_run()))
        self.assertIn(f"Scorer version: A = {UNVERSIONED}, B = {UNVERSIONED}", md)
        self.assertNotIn("different scorer versions", md)

    def test_a_rescored_run_is_labelled_in_the_header_and_a_banner(self):
        md = self.md(simulated_run(rescored_from="run_orig_a"), simulated_run())
        self.assertIn(f"A = {SCORER_VERSION} (RESCORED)", md)
        self.assertIn("RESCORED: at least one run was re-scored offline", md)
        self.assertIn("run_orig_a", md)
        self.assertIn("No model was called again", md)

    def test_the_existing_header_lines_are_unchanged(self):
        md = self.md(simulated_run(), simulated_run())
        self.assertIn("Run mode: A = SIMULATED, B = SIMULATED", md)
        self.assertIn("Job: A = standin_t, B = standin_t", md)
        self.assertIn("STAND-IN RUN", md)

    def test_the_html_report_has_it_too(self):
        html = ReportGenerator.generate_html(DiffEngine.compare_runs(simulated_run(), simulated_run()))
        self.assertIn("Scorer version", html)

    def test_the_diff_carries_the_version_fields(self):
        comp = DiffEngine.compare_runs(simulated_run(rescored_from="x"), legacy(simulated_run()))
        self.assertEqual((comp["scorer_version_a"], comp["scorer_version_b"]), (SCORER_VERSION, UNVERSIONED))
        self.assertEqual((comp["rescored_a"], comp["rescored_b"]), (True, False))
        self.assertEqual((comp["rescored_from_a"], comp["rescored_from_b"]), ("x", None))


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
        for p in (mock.patch.object(nel, "RunStore", lambda _p: self.store),
                  mock.patch.object(nel, "console", Console(file=self.buf, width=250, force_terminal=False))):
            p.start()
            self.addCleanup(p.stop)

    def test_eval_run_prints_the_version(self):
        result = CliRunner().invoke(nel.cli, ["eval", "run", "--config", "configs/eval_config_default.yaml", "--model", "sim/x",
                                              "--bench", "gpqa", "--limit", "3", "--job-name", "t"])
        self.assertEqual(result.exit_code, 0, self.buf.getvalue())
        self.assertIn(f"scorer v{SCORER_VERSION}", self.buf.getvalue())

    def test_compare_and_gate_show_the_version_and_the_mismatch(self):
        a, b = simulated_run(), simulated_run()
        old = legacy(simulated_run())
        for r in (a, b, old):
            self.store.save_run(r)
        CliRunner().invoke(nel.cli, ["compare", "--run-a", a.run_id, "--run-b", b.run_id])
        self.assertIn(f"Scorer version: A={SCORER_VERSION}, B={SCORER_VERSION}", self.buf.getvalue())
        self.assertNotIn("not directly comparable", self.buf.getvalue())
        self.buf.truncate(0); self.buf.seek(0)
        CliRunner().invoke(nel.cli, ["compare", "--run-a", old.run_id, "--run-b", b.run_id])
        self.assertIn(f"Scorer version: A={UNVERSIONED}, B={SCORER_VERSION}", self.buf.getvalue())
        self.assertIn("not directly comparable", self.buf.getvalue())
        self.buf.truncate(0); self.buf.seek(0)
        CliRunner().invoke(nel.cli, ["gate", "--baseline", old.run_id, "--candidate", b.run_id])
        self.assertIn(f"Scorer version: baseline={UNVERSIONED}, candidate={SCORER_VERSION}", self.buf.getvalue())
        self.assertIn("not directly comparable", self.buf.getvalue())

    def test_compare_labels_a_rescored_run(self):
        a, b = simulated_run(rescored_from="run_orig"), simulated_run()
        self.store.save_run(a)
        self.store.save_run(b)
        CliRunner().invoke(nel.cli, ["compare", "--run-a", a.run_id, "--run-b", b.run_id])
        self.assertIn(f"A={SCORER_VERSION} (RESCORED)", self.buf.getvalue())


if __name__ == "__main__":
    unittest.main()
