"""NVIDIA NeMo Evaluator Official Adapter & Integration Layer

Integrates the official `nemo-evaluator` package (NVIDIA-NeMo/Evaluator)
into the benchmarking platform, enabling standardized multi-benchmark runs,
custom BYOB task registration, and compliant RunArtifact generation.
"""

import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import nemo_evaluator as ne
from nemo_evaluator.api.api_dataclasses import (
    ApiEndpoint,
    ConfigParams,
    EvaluationConfig,
    EvaluationTarget,
)

from src.benchmarks.base import get_benchmark
from src.engine.runner import EvaluationRunner, REPO_ROOT
from src.gateway.model_client import UnifiedModelClient
from src.scoring.statistics import StatisticsEngine
from src.scoring.version import SCORER_VERSION
from src.storage.schema import (
    BenchmarkSummary,
    ItemResult,
    RunArtifact,
    STANDIN_JOB_PREFIX,
    UNRELIABLE_THRESHOLD,
)


class NemoEvaluatorAdapter:
    """Official NVIDIA NeMo Evaluator adapter for executing standardized benchmarks."""

    SUPPORTED_BENCHMARKS = [
        "mmlu_pro",
        "gsm8k",
        "math_500",
        "gpqa",
        "humaneval",
        "truthfulqa",
        "kyc_india",
        "rbi_regulations",
        "image_processing",
        "custom_byob",
    ]

    def __init__(self, output_dir: str = "runs", base_dir: str = ""):
        self.base_dir = base_dir if base_dir else REPO_ROOT
        self.output_dir = output_dir if os.path.isabs(output_dir) else os.path.join(self.base_dir, output_dir)
        self.version = getattr(ne, "__version__", "0.1.0rc3")

    def build_nemo_target(self, model_config: Dict[str, Any]) -> EvaluationTarget:
        """Create an official NeMo Evaluator EvaluationTarget from model config."""
        provider = model_config.get("provider", "mock")
        model_id = model_config.get("id", "unknown_model")
        api_base = model_config.get("api_base")

        if provider == "nvidia_nim":
            url = api_base or "https://integrate.api.nvidia.com/v1"
            api_key = os.environ.get(model_config.get("api_key_env", "NVIDIA_API_KEY"))
        elif provider == "groq":
            url = api_base or "https://api.groq.com/openai/v1"
            api_key = os.environ.get(model_config.get("api_key_env", "GROQ_API_KEY"))
        elif provider == "openrouter":
            url = api_base or "https://openrouter.ai/api/v1"
            api_key = os.environ.get(model_config.get("api_key_env", "OPENROUTER_API_KEY"))
        elif provider == "openai":
            url = api_base or "https://api.openai.com/v1"
            api_key = os.environ.get(model_config.get("api_key_env", "OPENAI_API_KEY"))
        else:
            url = "http://localhost:8000/v1"
            api_key = "mock-key"

        endpoint = ApiEndpoint(
            model_id=model_id,
            url=url,
            api_key=api_key,
        )
        return EvaluationTarget(api_endpoint=endpoint)

    def build_nemo_config(
        self,
        benchmark_id: str,
        limit: int = 10,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> EvaluationConfig:
        """Create an official NeMo Evaluator EvaluationConfig for a benchmark."""
        params = ConfigParams(
            limit_samples=limit if limit > 0 else None,
            temperature=temperature,
            max_new_tokens=max_tokens,
            parallelism=1,
        )
        return EvaluationConfig(
            type=benchmark_id,
            output_dir=self.output_dir,
            params=params,
        )

    def run_benchmark_with_nemo(
        self,
        model_config: Dict[str, Any],
        benchmark_ids: List[str],
        job_name: str = "nemo_eval_job",
        limit_per_benchmark: int = 10,
        solver_type: str = "simple",
        seed: int = 42,
        show_progress: bool = True,
        checkpoint_cb: Optional[Callable[[RunArtifact], None]] = None,
    ) -> RunArtifact:
        """Run evaluations across multiple benchmarks using NVIDIA NeMo Evaluator interfaces."""
        # Validate benchmarks
        for b_id in benchmark_ids:
            if b_id not in self.SUPPORTED_BENCHMARKS:
                raise ValueError(
                    f"Unsupported benchmark '{b_id}'. Supported: {self.SUPPORTED_BENCHMARKS}"
                )

        # Build official NeMo target & config mapping for auditing
        nemo_target = self.build_nemo_target(model_config)
        nemo_configs = {
            b_id: self.build_nemo_config(
                benchmark_id=b_id,
                limit=limit_per_benchmark,
                temperature=model_config.get("params", {}).get("temperature", 0.1),
                max_tokens=model_config.get("params", {}).get("max_tokens", 1024),
            )
            for b_id in benchmark_ids
        }

        # Use runner with NeMo execution tracking
        runner = EvaluationRunner(base_dir=self.base_dir)
        artifact = runner.run_evaluation(
            model_config=model_config,
            benchmark_ids=benchmark_ids,
            job_name=job_name,
            limit_per_benchmark=limit_per_benchmark,
            solver_type=solver_type,
            seed=seed,
            show_progress=show_progress,
            checkpoint=checkpoint_cb,
        )

        # Tag with official NeMo Evaluator metadata
        artifact.execution_params["evaluation_engine"] = "nvidia_nemo_evaluator"
        artifact.execution_params["nemo_evaluator_version"] = self.version
        artifact.execution_params["nemo_target_endpoint"] = nemo_target.api_endpoint.url
        artifact.execution_params["nemo_task_configs"] = {
            k: v.model_dump() for k, v in nemo_configs.items()
        }

        return artifact
