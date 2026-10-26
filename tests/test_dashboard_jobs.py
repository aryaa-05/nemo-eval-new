import os
import sys
import tempfile
import time
import unittest
from unittest import mock

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)

from src.dashboard.jobs import clear_job, get_job, is_running, start_job
from src.storage.run_store import RunStore

MOCK_MODEL = {
    "id": "meta/llama-3.1-70b-instruct",
    "name": "Test simulator",
    "provider": "mock",
    "params": {"temperature": 0.1, "max_tokens": 1024, "top_p": 0.95},
}


def _wait_for_finish(key, timeout=20.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = get_job(key)
        if job and job["status"] not in ("starting", "running"):
            return job
        time.sleep(0.02)
    raise AssertionError(f"job {key} did not finish within {timeout}s")


class TestBackgroundJobs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.store = RunStore(os.path.join(self.tmp.name, "runs"))

    def _start(self, model_config=MOCK_MODEL, **extra):
        key = start_job(
            store=self.store,
            model_config=model_config,
            benchmark_ids=["mmlu_pro"],
            job_name="test_dashboard_job",
            description="Unit test job.",
            limit_per_benchmark=1,
            solver_type="simple",
            seed=1,
            base_dir=REPO,
            **extra,
        )
        self.addCleanup(clear_job, key)
        return key

    def test_a_simulated_run_completes_and_is_saved(self):
        key = self._start()
        job = _wait_for_finish(key)
        self.assertEqual(job["status"], "completed", job.get("error"))
        self.assertFalse(is_running(job))
        self.assertIsNotNone(job["artifact"])
        self.assertEqual(job["artifact"].run_mode, "simulated")
        self.assertEqual(job["artifact"].provider, "mock")
        self.assertTrue(os.path.isfile(job["saved_path"]))
        loaded = self.store.get_run(job["run_id"])
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.run_id, job["run_id"])
        self.assertEqual(job["finished_benchmarks"], ["mmlu_pro"])

    def test_the_job_record_exposes_checkpoints(self):
        key = self._start()
        job = _wait_for_finish(key)
        self.assertEqual(job["status"], "completed")
        self.assertIsInstance(job["checkpoints"], list)

    def test_a_config_error_is_reported_without_saving_a_run(self):
        bad = {"id": "vendor/auto-without-routes", "name": "Bad", "provider": "auto"}
        key = self._start(bad)
        job = _wait_for_finish(key)
        self.assertEqual(job["status"], "error")
        self.assertIn("ProviderConfigError", job["error"])
        self.assertIn("routes", job["error"])
        self.assertIsNone(job["saved_path"])
        self.assertEqual(self.store.list_runs(), [])

    def test_an_auto_model_without_a_route_key_reports_the_missing_key(self):
        auto = {
            "id": "vendor/requested",
            "name": "Auto",
            "provider": "auto",
            "routes": {"openrouter": {"id": "served/model"}},
        }
        with mock.patch.dict(os.environ, {}, clear=True):
            key = self._start(auto)
            job = _wait_for_finish(key)
        self.assertEqual(job["status"], "error")
        self.assertIn("OPENROUTER_API_KEY", job["error"])
        self.assertIsNone(job["saved_path"])

    def test_a_cleared_job_is_gone(self):
        key = self._start()
        _wait_for_finish(key)
        clear_job(key)
        self.assertIsNone(get_job(key))
        self.assertFalse(is_running(None))


if __name__ == "__main__":
    unittest.main()
