import os
import sys
import unittest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.engine.nemo_adapter import NemoEvaluatorAdapter
from src.storage.run_store import RunStore


class TestNemoEvaluatorAdapter(unittest.TestCase):
    def setUp(self):
        self.adapter = NemoEvaluatorAdapter(output_dir="runs")

    def test_nemo_evaluator_package_installed(self):
        import nemo_evaluator as ne
        self.assertTrue(hasattr(ne, "EvaluationConfig"))
        self.assertTrue(hasattr(ne, "EvaluationTarget"))
        self.assertTrue(hasattr(ne, "ApiEndpoint"))

    def test_build_nemo_target_mock(self):
        cfg = {"id": "meta/llama-3.1-70b-instruct", "provider": "mock"}
        target = self.adapter.build_nemo_target(cfg)
        self.assertEqual(target.api_endpoint.model_id, "meta/llama-3.1-70b-instruct")
        self.assertEqual(target.api_endpoint.url, "http://localhost:8000/v1")

    def test_build_nemo_target_nim(self):
        cfg = {
            "id": "meta/llama-3.2-11b-vision-instruct",
            "provider": "nvidia_nim",
            "api_base": "https://integrate.api.nvidia.com/v1",
        }
        target = self.adapter.build_nemo_target(cfg)
        self.assertEqual(target.api_endpoint.url, "https://integrate.api.nvidia.com/v1")

    def test_build_nemo_config_all_benchmarks(self):
        for b_id in self.adapter.SUPPORTED_BENCHMARKS:
            config = self.adapter.build_nemo_config(b_id, limit=5, temperature=0.2, max_tokens=512)
            self.assertEqual(config.type, b_id)
            self.assertEqual(config.params.limit_samples, 5)
            self.assertEqual(config.params.temperature, 0.2)
            self.assertEqual(config.params.max_new_tokens, 512)

    def test_unsupported_benchmark_raises_error(self):
        with self.assertRaises(ValueError):
            self.adapter.run_benchmark_with_nemo(
                model_config={"id": "meta/llama-3.1-70b-instruct", "provider": "mock"},
                benchmark_ids=["non_existent_benchmark"],
            )

    def test_run_benchmark_with_nemo_adapter_mock(self):
        model_cfg = {
            "id": "meta/llama-3.1-70b-instruct",
            "name": "Llama-3.1-70B-Instruct",
            "provider": "mock",
            "params": {"temperature": 0.1, "max_tokens": 512, "top_p": 0.95},
        }
        benchmarks = ["mmlu_pro", "gsm8k", "kyc_india", "image_processing"]
        artifact = self.adapter.run_benchmark_with_nemo(
            model_config=model_cfg,
            benchmark_ids=benchmarks,
            job_name="unit_test_nemo_run",
            limit_per_benchmark=2,
            solver_type="simple",
        )

        self.assertEqual(artifact.status, "completed")
        self.assertEqual(artifact.execution_params.get("evaluation_engine"), "nvidia_nemo_evaluator")
        self.assertIn("nemo_evaluator_version", artifact.execution_params)
        self.assertEqual(len(artifact.benchmark_summaries), 4)
        for b in benchmarks:
            self.assertIn(b, artifact.benchmark_summaries)
            self.assertEqual(artifact.benchmark_summaries[b].total_samples, 2)

        # Ensure RunStore can save and retrieve it
        store = RunStore("runs")
        saved_path = store.save_run(artifact)
        self.assertTrue(os.path.exists(saved_path))
        loaded = store.get_run(artifact.run_id)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.run_id, artifact.run_id)


if __name__ == "__main__":
    unittest.main()
