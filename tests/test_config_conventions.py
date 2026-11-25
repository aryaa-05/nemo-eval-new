import glob
import os
import unittest

import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIGS = os.path.join(REPO, "configs")


def _load(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class TestFreeTierConfigConvention(unittest.TestCase):
    """Free-tier role comparisons live in configs/free_tier_pair*.yaml and never claim to be the required pairs."""

    def test_free_tier_files_are_labelled_by_mode(self):
        for path in glob.glob(os.path.join(CONFIGS, "free_tier_pair*.yaml")):
            with self.subTest(file=os.path.basename(path)):
                cfg = _load(path)
                self.assertTrue(str(cfg.get("job_name", "")).startswith("free_tier_"))
                description = str(cfg.get("description", ""))
                self.assertTrue("LIVE" in description or "SIMULATED" in description)
                self.assertIn("not results for", description.lower())

    def test_the_three_free_tier_pair_configs_exist(self):
        for n in (1, 2, 3):
            self.assertTrue(os.path.isfile(os.path.join(CONFIGS, f"free_tier_pair{n}.yaml")))

    def test_free_tier_configs_only_use_labelled_models_and_real_benchmarks(self):
        import sys
        from unittest import mock
        sys.path.insert(0, REPO)
        from src.benchmarks.base import get_benchmark
        from src.gateway.model_client import PROVIDER_SPECS, UnifiedModelClient

        models = _load(os.path.join(CONFIGS, "models_config.yaml"))["models"]
        keys = {spec["api_key_env"]: "dummy-key-for-config-check" for spec in PROVIDER_SPECS.values()}
        for path in glob.glob(os.path.join(CONFIGS, "free_tier_pair*.yaml")):
            cfg = _load(path)
            for model_id in cfg["models"]:
                with self.subTest(file=os.path.basename(path), model=model_id):
                    self.assertIn(model_id, models, "model has no block in models_config.yaml")
                    block = models[model_id]
                    self.assertIn(block["provider"], ("nvidia_nim", "groq", "mock"))
                    self.assertTrue(block["name"].startswith("Live free-tier") or block["name"].startswith("Simulated"),
                                    "display name must say LIVE free-tier or SIMULATED")
                    self.assertEqual(block["cost_per_1m_input"], 0.0)
                    if block["provider"] != "mock":
                        with mock.patch.dict(os.environ, keys):
                            UnifiedModelClient(block)
            for bench_id in cfg["benchmarks"]:
                with self.subTest(file=os.path.basename(path), benchmark=bench_id):
                    get_benchmark(bench_id, base_dir=REPO)

    def test_free_tier_pair_configs_do_not_touch_the_real_pair_model_ids(self):
        for path in glob.glob(os.path.join(CONFIGS, "free_tier_pair*.yaml")):
            for model_id in _load(path)["models"]:
                self.assertNotIn("ministral", model_id.lower())
                self.assertNotIn("llama-4", model_id.lower())
                self.assertNotIn("qwen3.5", model_id.lower())

    def test_unavailable_models_are_simulated(self):
        models = _load(os.path.join(CONFIGS, "models_config.yaml"))["models"]
        for model_id in ("google/gemma-3-12b-it", "meta/llama-3.2-11b-vision-instruct"):
            with self.subTest(model=model_id):
                self.assertEqual(models[model_id]["provider"], "mock")
                self.assertTrue(models[model_id]["name"].startswith("Simulated"))

    def test_real_configs_have_their_own_job_names(self):
        for path in glob.glob(os.path.join(CONFIGS, "eval_*.yaml")):
            with self.subTest(file=os.path.basename(path)):
                cfg = _load(path)
                self.assertFalse(str(cfg.get("job_name", "")).startswith("free_tier_"))


if __name__ == "__main__":
    unittest.main()
