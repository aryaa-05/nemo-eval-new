import json
import os
import subprocess
import sys
import unittest
from unittest import mock

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)

from src.benchmarks.base import get_benchmark
from src.comparison.diff_engine import DiffEngine
from src.engine.runner import EvaluationRunner
from src.gateway.mock_models import MockModelEngine
from src.gateway.model_client import ProviderConfigError, UnifiedModelClient
from src.reporting.report_generator import ReportGenerator
from src.storage.schema import RunArtifact

LLAMA_8B = "meta/llama-3.1-8b-instruct"

PROBE = """
import json, sys
sys.path.insert(0, {repo!r})
from src.gateway.mock_models import MockModelEngine as M
mc = {{"question_id": "mmlu_01", "options": ["a", "b", "c", "d"], "answer": "B"}}
he = {{"task_id": "HumanEval/3", "canonical_solution": "    return a < b\\n"}}
out = [
    M.generate_response({model!r}, "prompt one", "mmlu_pro", "B", mc, seed=7),
    M.generate_response({model!r}, "prompt two", "humaneval", "", he, seed=7),
]
print(json.dumps(out, sort_keys=True))
"""


def _run_probe(hash_seed: str) -> str:
    env = {**os.environ, "PYTHONHASHSEED": hash_seed, "PYTHONDONTWRITEBYTECODE": "1"}
    proc = subprocess.run(
        [sys.executable, "-c", PROBE.format(repo=REPO, model=LLAMA_8B)],
        capture_output=True, text=True, env=env, check=True,
    )
    return proc.stdout.strip()


def _mock_config(provider="mock", **extra):
    return {"id": LLAMA_8B, "name": "Test", "provider": provider, **extra}


def _fresh_run(model_id=LLAMA_8B, benchmarks=("mmlu_pro",), limit=2, seed=42):
    runner = EvaluationRunner(base_dir=REPO)
    return runner.run_evaluation(
        _mock_config() | {"id": model_id}, list(benchmarks), limit_per_benchmark=limit, seed=seed
    )


class TestSimulatorDeterminism(unittest.TestCase):
    def test_same_output_across_separate_processes(self):
        # Each process gets a different PYTHONHASHSEED; the old hash()-based roll would differ here.
        first = _run_probe("1")
        second = _run_probe("2")
        self.assertEqual(first, second)
        json.loads(first)  # sanity: it really is the JSON payload

    def test_subprocess_output_matches_in_process_output(self):
        mc = {"question_id": "mmlu_01", "options": ["a", "b", "c", "d"], "answer": "B"}
        he = {"task_id": "HumanEval/3", "canonical_solution": "    return a < b\n"}
        here = [
            MockModelEngine.generate_response(LLAMA_8B, "prompt one", "mmlu_pro", "B", mc, seed=7),
            MockModelEngine.generate_response(LLAMA_8B, "prompt two", "humaneval", "", he, seed=7),
        ]
        self.assertEqual(json.dumps(here, sort_keys=True), _run_probe("3"))

    def test_seed_is_honoured(self):
        keys = [f"mmlu_{i:02d}" for i in range(1, 16)]
        same_a = [MockModelEngine.roll(LLAMA_8B, "mmlu_pro", k, seed=1) for k in keys]
        same_b = [MockModelEngine.roll(LLAMA_8B, "mmlu_pro", k, seed=1) for k in keys]
        other = [MockModelEngine.roll(LLAMA_8B, "mmlu_pro", k, seed=2) for k in keys]
        self.assertEqual(same_a, same_b)
        self.assertNotEqual(same_a, other)

    def test_seed_reaches_the_simulator_through_the_runner(self):
        a = _fresh_run(benchmarks=("gpqa",), limit=15, seed=1)
        b = _fresh_run(benchmarks=("gpqa",), limit=15, seed=1)
        c = _fresh_run(benchmarks=("gpqa",), limit=15, seed=2)
        answers = lambda run: [it.raw_response for it in run.item_results]
        self.assertEqual(answers(a), answers(b))
        self.assertNotEqual(answers(a), answers(c))


class TestHumanEvalTaskId(unittest.TestCase):
    def test_item_key_uses_task_id_when_question_id_missing(self):
        self.assertEqual(MockModelEngine.item_key({"task_id": "HumanEval/9"}, "p"), "HumanEval/9")
        self.assertEqual(MockModelEngine.item_key({"question_id": "q1", "task_id": "t1"}, "p"), "q1")

    def test_item_key_without_any_id_depends_on_prompt(self):
        self.assertNotEqual(MockModelEngine.item_key({}, "prompt A"), MockModelEngine.item_key({}, "prompt B"))
        self.assertEqual(MockModelEngine.item_key(None, "prompt A"), MockModelEngine.item_key({}, "prompt A"))

    def test_humaneval_items_no_longer_share_one_roll(self):
        items = get_benchmark("humaneval", base_dir=REPO).load_items(limit=15)
        self.assertEqual(len(items), 15)
        self.assertTrue(all("question_id" not in it for it in items), "dataset shape changed; revisit this test")
        rolls = {
            MockModelEngine.roll(LLAMA_8B, "humaneval", MockModelEngine.item_key(it, it["prompt"]))
            for it in items
        }
        self.assertGreater(len(rolls), 1)

    def test_humaneval_scores_are_not_all_or_nothing(self):
        # With the old bug every HumanEval item shared one roll, so a model scored 0/N or N/N.
        # At least one of these profiles must now land strictly between.
        partial = []
        for model_id in (LLAMA_8B, "mistralai/mixtral-8x7b-instruct", "openai/gpt-4o-mini"):
            run = _fresh_run(model_id=model_id, benchmarks=("humaneval",), limit=15)
            summary = run.benchmark_summaries["humaneval"]
            partial.append(0 < summary.correct_samples < summary.total_samples)
        self.assertTrue(any(partial))


class TestLiveProviderFailsLoudly(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict(os.environ, {}, clear=False)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop("NVIDIA_API_KEY", None)
        os.environ.pop("CUSTOM_KEY_VAR", None)

    def test_nim_without_key_raises(self):
        with self.assertRaises(ProviderConfigError) as ctx:
            UnifiedModelClient(_mock_config("nvidia_nim"))
        self.assertIn("NVIDIA_API_KEY", str(ctx.exception))

    def test_openai_without_key_raises_and_names_custom_env_var(self):
        with self.assertRaises(ProviderConfigError) as ctx:
            UnifiedModelClient(_mock_config("openai", api_key_env="CUSTOM_KEY_VAR"))
        self.assertIn("CUSTOM_KEY_VAR", str(ctx.exception))

    def test_unsupported_provider_raises_instead_of_using_mock(self):
        with self.assertRaises(ProviderConfigError):
            UnifiedModelClient(_mock_config("huggingface"))

    def test_runner_raises_before_any_item_is_scored(self):
        runner = EvaluationRunner(base_dir=REPO)
        with self.assertRaises(ProviderConfigError):
            runner.run_evaluation(_mock_config("nvidia_nim"), ["mmlu_pro"], limit_per_benchmark=1)

    def test_mock_needs_no_key_and_is_simulated(self):
        self.assertEqual(UnifiedModelClient(_mock_config("mock")).run_mode, "simulated")

    def test_ollama_needs_no_key_and_is_live(self):
        self.assertEqual(UnifiedModelClient(_mock_config("ollama")).run_mode, "live")

    def test_keyed_provider_with_key_is_live(self):
        with mock.patch.dict(os.environ, {"NVIDIA_API_KEY": "dummy-not-a-real-key"}):
            self.assertEqual(UnifiedModelClient(_mock_config("nvidia_nim")).run_mode, "live")


class TestRunModeRecorded(unittest.TestCase):
    def test_simulated_run_is_recorded_as_simulated(self):
        self.assertEqual(_fresh_run().run_mode, "simulated")

    def test_run_mode_survives_json_round_trip(self):
        run = _fresh_run()
        again = RunArtifact.model_validate_json(run.model_dump_json())
        self.assertEqual(again.run_mode, "simulated")

    def _legacy_dict(self):
        run = _fresh_run()
        data = json.loads(run.model_dump_json())
        del data["run_mode"]
        return data

    def test_legacy_mock_artifact_is_inferred_simulated(self):
        self.assertEqual(RunArtifact.model_validate(self._legacy_dict()).run_mode, "simulated")

    def test_legacy_non_mock_artifact_stays_unknown(self):
        data = self._legacy_dict()
        data["provider"] = "nvidia_nim"
        self.assertEqual(RunArtifact.model_validate(data).run_mode, "unknown")


class TestReportShowsRunMode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.comparison = DiffEngine.compare_runs(_fresh_run(model_id=LLAMA_8B), _fresh_run(model_id="openai/gpt-4o-mini"))

    def test_simulated_report_header_and_banner(self):
        md = ReportGenerator.generate_markdown(self.comparison)
        self.assertIn("Run mode: A = SIMULATED, B = SIMULATED", md)
        self.assertIn("SIMULATED RESULTS", md)
        self.assertNotIn("WARNING: these runs have different modes", md)

    def test_html_report_carries_the_same_header(self):
        self.assertIn("Run mode: A = SIMULATED, B = SIMULATED", ReportGenerator.generate_html(self.comparison))

    def test_mixed_mode_report_warns(self):
        md = ReportGenerator.generate_markdown({**self.comparison, "run_mode_b": "live"})
        self.assertIn("Run mode: A = SIMULATED, B = LIVE", md)
        self.assertIn("WARNING: these runs have different modes", md)

    def test_live_only_report_has_no_simulated_banner(self):
        md = ReportGenerator.generate_markdown({**self.comparison, "run_mode_a": "live", "run_mode_b": "live"})
        self.assertIn("Run mode: A = LIVE, B = LIVE", md)
        self.assertNotIn("SIMULATED RESULTS", md)


if __name__ == "__main__":
    unittest.main()
