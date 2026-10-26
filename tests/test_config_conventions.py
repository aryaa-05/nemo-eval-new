import glob
import os
import unittest

import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIGS = os.path.join(REPO, "configs")


def _load(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class TestStandinConfigConvention(unittest.TestCase):
    """Stand-in runs (cheaper look-alike models used only to test the pipeline) live in configs/standin_*.yaml
    and are always labelled as pipeline validation. These checks pass trivially while no such file exists."""

    def test_standin_files_are_labelled_as_pipeline_validation(self):
        for path in glob.glob(os.path.join(CONFIGS, "standin_*.yaml")):
            with self.subTest(file=os.path.basename(path)):
                cfg = _load(path)
                self.assertTrue(str(cfg.get("job_name", "")).startswith("standin_"), "job_name must start with 'standin_'")
                self.assertIn("pipeline validation", str(cfg.get("description", "")).lower())

    def test_the_three_standin_pair_configs_exist(self):
        for n in (1, 2, 3):
            self.assertTrue(os.path.isfile(os.path.join(CONFIGS, f"standin_pair{n}.yaml")))

    def test_standin_configs_only_use_labelled_live_standin_models_and_real_benchmarks(self):
        import sys
        from unittest import mock
        sys.path.insert(0, REPO)
        from src.benchmarks.base import get_benchmark
        from src.gateway.model_client import PROVIDER_SPECS, UnifiedModelClient

        models = _load(os.path.join(CONFIGS, "models_config.yaml"))["models"]
        keys = {spec["api_key_env"]: "dummy-key-for-config-check" for spec in PROVIDER_SPECS.values()}
        for path in glob.glob(os.path.join(CONFIGS, "standin_*.yaml")):
            cfg = _load(path)
            for model_id in cfg["models"]:
                with self.subTest(file=os.path.basename(path), model=model_id):
                    self.assertIn(model_id, models, "stand-in model has no block in models_config.yaml")
                    block = models[model_id]
                    self.assertIn(block["provider"], ("nvidia_nim", "groq"), "stand-ins are free live providers, not mock/paid")
                    self.assertTrue(block["name"].startswith("Stand-in"), "display name must say it is a stand-in")
                    self.assertEqual(block["cost_per_1m_input"], 0.0)
                    with mock.patch.dict(os.environ, keys):
                        UnifiedModelClient(block)  # host, key variable, timeout and rpm are all valid
            for bench_id in cfg["benchmarks"]:
                with self.subTest(file=os.path.basename(path), benchmark=bench_id):
                    get_benchmark(bench_id, base_dir=REPO)

    def test_standin_pair_configs_do_not_touch_the_real_pair_model_ids(self):
        for path in glob.glob(os.path.join(CONFIGS, "standin_*.yaml")):
            for model_id in _load(path)["models"]:
                self.assertNotIn("ministral", model_id.lower())
                self.assertNotIn("llama-4", model_id.lower())
                self.assertNotIn("qwen3.5", model_id.lower())

    def test_real_configs_do_not_hide_a_standin_run(self):
        for path in glob.glob(os.path.join(CONFIGS, "eval_*.yaml")):
            with self.subTest(file=os.path.basename(path)):
                cfg = _load(path)
                self.assertFalse(str(cfg.get("job_name", "")).startswith("standin_"),
                                 "stand-in runs must be in configs/standin_*.yaml, not in a real-looking config")


if __name__ == "__main__":
    unittest.main()
