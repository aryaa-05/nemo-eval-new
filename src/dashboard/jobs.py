"""Background evaluation jobs for the Streamlit dashboard.

Streamlit reruns the script on every interaction, so a long run must not execute inline or the page freezes.
This module starts one daemon thread per run and keeps an in-memory status record that the page can poll.
The thread saves each finished benchmark as an INCOMPLETE checkpoint, then saves the final artifact through
RunStore exactly like the CLI does.
"""
from __future__ import annotations

import threading
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.engine.runner import EvaluationRunner
from src.storage.run_store import RunStore

_LOCK = threading.Lock()
_JOBS: Dict[str, Dict[str, Any]] = {}


def _utc_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def start_job(
    *,
    store: RunStore,
    model_config: Dict[str, Any],
    benchmark_ids: List[str],
    job_name: str,
    description: str,
    limit_per_benchmark: int,
    solver_type: str,
    seed: int = 42,
    standin: bool = False,
    base_dir: str = "",
) -> str:
    """Start a run in a daemon thread and return its job key."""
    key = uuid.uuid4().hex
    with _LOCK:
        _JOBS[key] = {
            "key": key,
            "status": "starting",
            "run_id": None,
            "saved_path": None,
            "model_id": model_config.get("id", ""),
            "model_name": model_config.get("name", model_config.get("id", "")),
            "provider": model_config.get("provider", "mock"),
            "job_name": job_name,
            "benchmark_ids": list(benchmark_ids),
            "limit_per_benchmark": limit_per_benchmark,
            "solver_type": solver_type,
            "standin": bool(standin),
            "started_at": time.time(),
            "started_at_iso": _utc_iso(),
            "finished_at": None,
            "finished_at_iso": None,
            "checkpoints": [],
            "finished_benchmarks": [],
            "artifact": None,
            "error": None,
            "error_type": None,
        }

    thread = threading.Thread(
        target=_execute,
        kwargs={
            "key": key,
            "store": store,
            "model_config": model_config,
            "benchmark_ids": list(benchmark_ids),
            "job_name": job_name,
            "description": description,
            "limit_per_benchmark": limit_per_benchmark,
            "solver_type": solver_type,
            "seed": seed,
            "base_dir": base_dir,
        },
        daemon=True,
        name=f"nemo-eval-{key[:8]}",
    )
    thread.start()
    return key


def _update(key: str, **fields: Any) -> None:
    with _LOCK:
        job = _JOBS.get(key)
        if job is not None:
            job.update(fields)


def _execute(
    *,
    key: str,
    store: RunStore,
    model_config: Dict[str, Any],
    benchmark_ids: List[str],
    job_name: str,
    description: str,
    limit_per_benchmark: int,
    solver_type: str,
    seed: int,
    base_dir: str,
) -> None:
    _update(key, status="running")

    def save_checkpoint(partial: Any) -> None:
        path = store.save_checkpoint(partial)
        with _LOCK:
            job = _JOBS.get(key)
            if job is None:
                return
            job["status"] = "running"
            job["run_id"] = partial.run_id
            job["finished_benchmarks"] = list(partial.benchmark_summaries)
            job["checkpoints"].append(
                {
                    "run_id": partial.run_id,
                    "path": path,
                    "benchmarks": list(partial.benchmark_summaries),
                    "saved_at": time.time(),
                }
            )

    try:
        runner = EvaluationRunner(base_dir=base_dir)
        artifact = runner.run_evaluation(
            model_config=model_config,
            benchmark_ids=benchmark_ids,
            job_name=job_name,
            description=description,
            limit_per_benchmark=limit_per_benchmark,
            solver_type=solver_type,
            seed=seed,
            show_progress=False,
            checkpoint=save_checkpoint,
        )
        saved_path = store.save_run(artifact)
        status = "completed" if artifact.is_usable else ("failed" if artifact.is_failed else "incomplete")
        _update(
            key,
            status=status,
            artifact=artifact,
            saved_path=saved_path,
            run_id=artifact.run_id,
            finished_at=time.time(),
            finished_at_iso=_utc_iso(),
            finished_benchmarks=list(artifact.benchmark_summaries),
        )
    except BaseException as exc:  # a bad config, a crash or an interruption: keep the status, do not kill the page
        _update(
            key,
            status="error",
            error=f"{type(exc).__name__}: {exc}",
            error_type=type(exc).__name__,
            finished_at=time.time(),
            finished_at_iso=_utc_iso(),
        )


def get_job(key: Optional[str]) -> Optional[Dict[str, Any]]:
    """A copy of the job record, safe to render while the thread keeps updating the original."""
    if not key:
        return None
    with _LOCK:
        job = _JOBS.get(key)
        if job is None:
            return None
        copy = dict(job)
        copy["checkpoints"] = list(job.get("checkpoints") or [])
        copy["finished_benchmarks"] = list(job.get("finished_benchmarks") or [])
        return copy


def clear_job(key: Optional[str]) -> None:
    if not key:
        return
    with _LOCK:
        _JOBS.pop(key, None)


def is_running(job: Optional[Dict[str, Any]]) -> bool:
    return bool(job) and job.get("status") in ("starting", "running")
