import os
import time
import uuid
import statistics
from datetime import datetime, timezone
from typing import Callable, Dict, Any, List, Optional
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from src.storage.schema import RunArtifact, ItemResult, BenchmarkSummary, RUN_COMPLETED, RUN_FAILED, RUN_INCOMPLETE
from src.gateway.model_client import UnifiedModelClient
from src.gateway.interceptor import InterceptorProxy
from src.benchmarks.base import get_benchmark
from src.engine.solvers import get_solver
from src.engine.answer_scoring import score_generation
from src.scoring.statistics import StatisticsEngine
from src.scoring.version import SCORER_VERSION

# Early stop: if the first EARLY_STOP_ITEMS items of a model ALL fail with the same one of these non-retryable HTTP
# statuses, the model is (almost certainly) unavailable for this account, so the run is aborted and marked failed.
# 429, 5xx and timeouts never trigger it, and neither do other 4xx statuses.
EARLY_STOP_ITEMS = 5
EARLY_STOP_STATUSES = (401, 403, 404, 410)
_STATUS_HINTS = {
    401: "the API key was rejected",
    403: "access is denied for this account",
    404: "the model was not found for this account (a model can be listed by the provider and still return 404)",
    410: "the model has been retired",
}


def early_stop_reason(model_config: Dict[str, Any], statuses: List[Optional[int]]) -> Optional[str]:
    """The failure message if `statuses` (HTTP status of each of the first items, None for an item that did not fail
    with an HTTP status) meets the early-stop rule, else None."""
    if len(statuses) != EARLY_STOP_ITEMS or any(s is None for s in statuses) or len(set(statuses)) != 1:
        return None
    status = statuses[0]
    if status not in EARLY_STOP_STATUSES:
        return None
    model_id = model_config["id"]
    provider = model_config.get("provider", "?")
    return (f"Model '{model_id}' (provider {provider}) returned HTTP {status} on each of its first {EARLY_STOP_ITEMS} items: "
            f"{_STATUS_HINTS[status]}. The run was stopped after {EARLY_STOP_ITEMS} calls instead of running every item.")


def incomplete_reason(done: List[str], planned: List[str]) -> str:
    return (f"INCOMPLETE: saved after {len(done)} of {len(planned)} benchmarks ({', '.join(done)}); the run did not finish. "
            f"Results for {', '.join(b for b in planned if b not in done)} are missing. It cannot be compared or gated.")


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

class EvaluationRunner:
    def __init__(self, base_dir: str = ""):
        self.base_dir = base_dir if base_dir else REPO_ROOT
        self.interceptor = InterceptorProxy()

    def run_evaluation(
        self,
        model_config: Dict[str, Any],
        benchmark_ids: List[str],
        job_name: str = "eval_run",
        description: str = "",
        limit_per_benchmark: int = 0,
        solver_type: str = "simple",
        seed: int = 42,
        show_progress: bool = True,
        checkpoint: Optional[Callable[[RunArtifact], None]] = None,
    ) -> RunArtifact:
        """Run one model over the benchmarks.

        `checkpoint`, if given, is called with an INCOMPLETE artifact after every benchmark except the last, so a crashed
        run keeps its finished benchmarks. The artifact returned at the end has the same run_id and replaces it.
        """
        run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        client = UnifiedModelClient(model_config, self.interceptor, seed=seed)
        resolved_model_config = {**model_config, "provider": client.provider}
        solver = get_solver(solver_type)

        all_item_results: List[ItemResult] = []
        bench_summaries: Dict[str, BenchmarkSummary] = {}

        total_correct = 0
        total_evaluated = 0
        total_truncated = 0
        total_no_answer = 0
        total_api_error = 0
        first_statuses: List[Optional[int]] = []   # HTTP status of each of the first items (None = no HTTP failure)
        failure_reason: Optional[str] = None

        def build(status: str, note: Optional[str]) -> RunArtifact:
            overall_acc = total_correct / total_evaluated if total_evaluated > 0 else 0.0
            ci_low, ci_high = StatisticsEngine.wilson_confidence_interval(total_correct, total_evaluated)
            lats = [it.latency_sec for it in all_item_results]
            return RunArtifact(
                run_id=run_id,
                job_name=job_name,
                description=description,
                timestamp=datetime.now(timezone.utc).isoformat(),
                seed=seed,
                model_id=model_config["id"],
                served_model_id=client.served_model_id,
                model_substituted=client.route_substitute,
                route_note=client.route_note,
                model_name=model_config.get("name", model_config["id"]),
                provider=client.provider,
                run_mode=client.run_mode,
                status=status,
                failure_reason=note,
                scorer_version=SCORER_VERSION,
                truncated_count=total_truncated,
                no_answer_count=total_no_answer,
                api_error_count=total_api_error,
                overall_accuracy=round(overall_acc, 4),
                overall_ci_lower=ci_low,
                overall_ci_upper=ci_high,
                mean_latency=round(statistics.mean(lats), 3) if lats else 0.0,
                total_tokens=sum(it.total_tokens for it in all_item_results),
                total_cost=round(sum(it.cost for it in all_item_results), 6),
                benchmark_summaries=dict(bench_summaries),
                item_results=list(all_item_results),
                execution_params={
                    "solver_type": solver_type,
                    "limit_per_benchmark": limit_per_benchmark,
                    "benchmarks": benchmark_ids
                }
            )

        for b_idx, b_id in enumerate(benchmark_ids):
            if failure_reason:
                break
            bench = get_benchmark(b_id, base_dir=self.base_dir)
            items = bench.load_items(limit=limit_per_benchmark)

            bench_done = 0
            bench_correct = 0
            bench_truncated = 0
            bench_no_answer = 0
            bench_api_error = 0
            bench_latencies = []
            bench_tokens = 0
            bench_cost = 0.0

            for item in items:
                formatted_prompt = solver.format_prompt(item, b_id)
                ground_truth = item.get("answer", "")

                # Query Model
                gen_resp = client.generate(
                    prompt=formatted_prompt,
                    benchmark_id=b_id,
                    ground_truth=ground_truth,
                    item_meta=item
                )

                raw_text = gen_resp.get("text", "")
                latency = gen_resp.get("latency_sec", 0.0)
                p_tokens = gen_resp.get("prompt_tokens", 0)
                c_tokens = gen_resp.get("completion_tokens", 0)
                t_tokens = gen_resp.get("total_tokens", p_tokens + c_tokens)
                cost = gen_resp.get("cost", 0.0)

                # Evaluate Scorer (strips reasoning text, flags truncated / no-answer responses)
                scored = score_generation(bench, item, gen_resp)
                is_correct, score, ext_ans, err = scored.is_correct, scored.score, scored.extracted, scored.error
                if is_correct:
                    bench_correct += 1
                if scored.truncated:
                    bench_truncated += 1
                if scored.no_answer:
                    bench_no_answer += 1
                if scored.api_error:
                    bench_api_error += 1

                bench_latencies.append(latency)
                bench_tokens += t_tokens
                bench_cost += cost

                q_id = item.get("question_id") or item.get("task_id", f"{b_id}_{len(all_item_results)}")

                item_res = ItemResult(
                    question_id=q_id,
                    benchmark=b_id,
                    model_id=model_config["id"],
                    prompt=formatted_prompt,
                    raw_response=raw_text,
                    scored_text=scored.scored_text,
                    reasoning_text=gen_resp.get("reasoning_text"),
                    finish_reason=gen_resp.get("finish_reason"),
                    truncated=scored.truncated,
                    attempts=gen_resp.get("attempts", 1),
                    attempt_log=[str(x) for x in (gen_resp.get("attempt_log") or [])],
                    http_status=gen_resp.get("http_status") if scored.api_error else None,
                    extracted_answer=ext_ans,
                    ground_truth=str(ground_truth),
                    is_correct=is_correct,
                    score=score,
                    latency_sec=latency,
                    prompt_tokens=p_tokens,
                    completion_tokens=c_tokens,
                    total_tokens=t_tokens,
                    cost=cost,
                    error=err,
                    solver_type=solver_type,
                    domain=bench.domain
                )
                all_item_results.append(item_res)
                bench_done += 1

                if len(first_statuses) < EARLY_STOP_ITEMS:
                    first_statuses.append(item_res.http_status)
                    failure_reason = early_stop_reason(resolved_model_config, first_statuses)
                    if failure_reason:
                        break

            n_samples = bench_done
            b_acc = bench_correct / n_samples if n_samples > 0 else 0.0
            ci_low, ci_high = StatisticsEngine.wilson_confidence_interval(bench_correct, n_samples)
            m_lat = statistics.mean(bench_latencies) if bench_latencies else 0.0
            med_lat = statistics.median(bench_latencies) if bench_latencies else 0.0

            summary = BenchmarkSummary(
                benchmark_id=b_id,
                benchmark_name=bench.name,
                domain=bench.domain,
                total_samples=n_samples,
                correct_samples=bench_correct,
                truncated_count=bench_truncated,
                no_answer_count=bench_no_answer,
                api_error_count=bench_api_error,
                accuracy=round(b_acc, 4),
                ci_lower=ci_low,
                ci_upper=ci_high,
                mean_latency=round(m_lat, 3),
                median_latency=round(med_lat, 3),
                total_tokens=bench_tokens,
                total_cost=round(bench_cost, 6)
            )
            bench_summaries[b_id] = summary

            total_correct += bench_correct
            total_evaluated += n_samples
            total_truncated += bench_truncated
            total_no_answer += bench_no_answer
            total_api_error += bench_api_error

            # Save what is finished so far, so a crash in a later benchmark does not lose it.
            if checkpoint and not failure_reason and b_idx < len(benchmark_ids) - 1:
                done = list(bench_summaries)
                checkpoint(build(RUN_INCOMPLETE, incomplete_reason(done, benchmark_ids)))

        if failure_reason:
            return build(RUN_FAILED, failure_reason)
        return build(RUN_COMPLETED, None)
