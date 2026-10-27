from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, computed_field, model_validator

STANDIN_JOB_PREFIX = "standin_"  # a job name starting with this marks a stand-in (pipeline validation) run
UNRELIABLE_THRESHOLD = 0.10      # a run is UNRELIABLE when MORE than this share of its items are api_error/truncated/no_answer
UNRELIABLE_BENCHMARK_THRESHOLD = 0.20  # a benchmark is UNRELIABLE when MORE than this share of ITS items are flagged
RUN_COMPLETED, RUN_FAILED, RUN_INCOMPLETE = "completed", "failed", "incomplete"

class EvalItem(BaseModel):
    question_id: str
    benchmark: str
    question: str
    prompt: Optional[str] = None
    options: Optional[List[str]] = None
    answer: Optional[str] = None
    reasoning: Optional[str] = None
    solution: Optional[str] = None
    subject: Optional[str] = None
    domain: Optional[str] = None
    canonical_solution: Optional[str] = None
    test: Optional[str] = None
    entry_point: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ModelConfig(BaseModel):
    id: str
    name: str
    provider: str = "mock"  # "nvidia_nim", "openai", "huggingface", "ollama", "mock"
    api_base: Optional[str] = None
    api_key_env: Optional[str] = None  # defaults to the provider's own variable (NVIDIA_API_KEY / OPENAI_API_KEY)
    timeout_seconds: float = 120.0
    context_window: int = 131072
    params: Dict[str, Any] = Field(default_factory=lambda: {"temperature": 0.1, "max_tokens": 1024, "top_p": 0.95})
    cost_per_1m_input: float = 0.0
    cost_per_1m_output: float = 0.0
    strengths: List[str] = Field(default_factory=list)

class BenchmarkConfig(BaseModel):
    id: str
    name: str
    domain: str
    description: str
    scorer: str
    sample_file: str
    num_options: Optional[int] = None

class ItemResult(BaseModel):
    question_id: str
    benchmark: str
    model_id: str
    prompt: str
    raw_response: str
    scored_text: Optional[str] = None  # text actually scored after reasoning was stripped; None if same as raw_response
    reasoning_text: Optional[str] = None  # separate reasoning field returned by the provider, kept for audit only
    finish_reason: Optional[str] = None
    truncated: bool = False
    attempts: int = 1  # HTTP requests made for this item (more than 1 means 429/5xx responses were retried)
    # What each request returned, in order: an HTTP status ("503", "200"), "read_timeout", or an error class name.
    # Everything before the last entry is an intermediate failure, so retries can be audited.
    attempt_log: List[str] = Field(default_factory=list)
    http_status: Optional[int] = None  # HTTP status of a failed call (None for success, timeouts and network errors)
    extracted_answer: Optional[str] = None
    ground_truth: Optional[str] = None
    is_correct: bool = False
    score: float = 0.0
    latency_sec: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0
    error: Optional[str] = None
    solver_type: str = "simple"
    domain: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class BenchmarkSummary(BaseModel):
    benchmark_id: str
    benchmark_name: str
    domain: str
    total_samples: int
    correct_samples: int
    truncated_count: int = 0  # responses cut off by max_tokens; counted incorrect
    no_answer_count: int = 0  # responses with no extractable answer; counted incorrect
    api_error_count: int = 0  # calls that failed (after any retries); counted incorrect
    accuracy: float
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    mean_latency: float = 0.0
    median_latency: float = 0.0
    total_tokens: int = 0
    total_cost: float = 0.0

    @computed_field  # written to the run file as well
    @property
    def unreliable_fraction(self) -> float:
        flagged = self.truncated_count + self.no_answer_count + self.api_error_count
        return round(flagged / self.total_samples, 4) if self.total_samples else 0.0

    @computed_field
    @property
    def unreliable(self) -> bool:
        """More than 20% of this benchmark's items are api_error / truncated / no_answer_found."""
        flagged = self.truncated_count + self.no_answer_count + self.api_error_count
        return bool(self.total_samples) and flagged / self.total_samples > UNRELIABLE_BENCHMARK_THRESHOLD

class RunArtifact(BaseModel):
    run_id: str
    job_name: str
    description: Optional[str] = ""
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    git_hash: Optional[str] = "HEAD"
    seed: int = 42
    model_id: str                          # the model the run asked for (may be routed to a different served id)
    served_model_id: Optional[str] = None  # the id the provider actually served; None on older run files
    model_substituted: bool = False        # True when the auto route is a look-alike/substitute, not the exact model
    route_note: Optional[str] = None       # why/how the auto route was chosen (never contains a key)
    model_name: str
    provider: str                          # the provider that actually served the run (never "auto")
    run_mode: str = "unknown"  # "simulated" (mock engine) or "live" (real provider)
    # "completed"; "failed" (stopped early because the model is unavailable); or "incomplete" (saved after some
    # benchmarks and never finished). Only "completed" runs can be compared or gated.
    status: str = "completed"
    failure_reason: Optional[str] = None  # why a run is failed / incomplete
    scorer_version: str = "unversioned"   # version of the scoring rules that produced the verdicts (src/scoring/version.py)
    rescored_from: Optional[str] = None   # run_id of the original run, when this file was re-scored offline from its stored replies
    rescored_at: Optional[str] = None
    truncated_count: int = 0
    no_answer_count: int = 0
    api_error_count: int = 0
    overall_accuracy: float = 0.0
    overall_ci_lower: float = 0.0
    overall_ci_upper: float = 0.0
    mean_latency: float = 0.0
    total_tokens: int = 0
    total_cost: float = 0.0
    benchmark_summaries: Dict[str, BenchmarkSummary] = Field(default_factory=dict)
    item_results: List[ItemResult] = Field(default_factory=list)
    execution_params: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _infer_legacy_run_mode(cls, data: Any) -> Any:
        # Artifacts saved before run_mode existed: provider "mock" was always served by the simulator.
        # Any other provider could have silently fallen back to the simulator, so it stays "unknown".
        if isinstance(data, dict) and "run_mode" not in data:
            data = {**data, "run_mode": "simulated" if data.get("provider") == "mock" else "unknown"}
        return data

    @property
    def is_standin(self) -> bool:
        return self.job_name.lower().startswith(STANDIN_JOB_PREFIX)

    def _unscorable_items(self) -> int:
        # every item has at most one of these three flags, so the counts can simply be added
        return self.api_error_count + self.truncated_count + self.no_answer_count

    @computed_field  # also written to the run file, so the artifact itself says UNRELIABLE
    @property
    def unreliable_fraction(self) -> float:
        n = len(self.item_results)
        return round(self._unscorable_items() / n, 4) if n else 0.0

    @property
    def is_failed(self) -> bool:
        return self.status == RUN_FAILED

    @property
    def is_incomplete(self) -> bool:
        return self.status == RUN_INCOMPLETE

    @property
    def is_usable(self) -> bool:
        """Only a completed run may be compared or gated."""
        return self.status == RUN_COMPLETED

    @property
    def is_rescored(self) -> bool:
        return self.rescored_from is not None

    @computed_field
    @property
    def unreliable(self) -> bool:
        # a failed or incomplete run is reported as FAILED / INCOMPLETE, never as UNRELIABLE
        n = len(self.item_results)
        return self.is_usable and bool(n) and self._unscorable_items() / n > UNRELIABLE_THRESHOLD

    @computed_field
    @property
    def unreliable_benchmarks(self) -> List[str]:
        """Benchmarks where more than 20% of the items are flagged, even if the whole run is under 10%."""
        return [b for b, s in self.benchmark_summaries.items() if s.unreliable] if self.is_usable else []
