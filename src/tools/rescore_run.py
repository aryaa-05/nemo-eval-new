"""Re-score the stored replies of a finished run with the CURRENT scorer, offline.

No model is called and the original run file is never modified. The result is a NEW run file, labelled as re-scored,
that carries the scorer version and the id of the run it came from. It is saved as JSON only (it is not added to the run
list or the leaderboard), so it can be used with `compare` and `gate` by run id.

    python src/tools/rescore_run.py RUN_ID [RUN_ID ...] [--runs-dir runs] [--report-md FILE]

Items of failed-call (api_error) and truncated replies go through the same function as in a live run, so their verdicts
cannot change. HumanEval replies are kept as they are: re-running them needs the code harness and the original test
cases, which a run file does not store.
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from src.benchmarks.base import get_benchmark
from src.engine.answer_scoring import ERR_NO_ANSWER, ERR_TRUNCATED, score_generation
from src.scoring.statistics import StatisticsEngine
from src.scoring.version import SCORER_VERSION
from src.storage.run_store import RunStore
from src.storage.schema import RUN_COMPLETED, BenchmarkSummary, ItemResult, RunArtifact

KEPT_BENCHMARKS = ("humaneval",)  # not re-scored: needs the code harness and the original tests
API_ERROR_PREFIX = "api_error"


class RescoreError(Exception):
    pass


def rescored_run_id(run_id: str) -> str:
    return f"{run_id}_rescored_v{SCORER_VERSION}"


def _flags(item: ItemResult) -> Tuple[bool, bool, bool]:
    """(truncated, no_answer, api_error) of an item as stored."""
    err = item.error or ""
    return bool(item.truncated), err == ERR_NO_ANSWER, err.startswith(API_ERROR_PREFIX)


def rescore_item(bench: Any, item: ItemResult) -> ItemResult:
    gen_resp: Dict[str, Any] = {"text": item.raw_response, "finish_reason": item.finish_reason}
    if (item.error or "").startswith(API_ERROR_PREFIX):
        gen_resp["error"] = item.error[len(API_ERROR_PREFIX):].lstrip(": ").strip() or "unknown"
    scored = score_generation(bench, {"answer": item.ground_truth or ""}, gen_resp)
    return item.model_copy(update={
        "extracted_answer": scored.extracted, "is_correct": scored.is_correct, "score": scored.score,
        "error": scored.error, "truncated": scored.truncated, "scored_text": scored.scored_text,
    })


def _tail(text: str, n: int = 240) -> str:
    text = (text or "").strip().replace("\r", "")
    return text if len(text) <= n else "..." + text[-n:]


def rescore_artifact(run: RunArtifact, base_dir: str = REPO) -> Tuple[RunArtifact, Dict[str, Any]]:
    """The re-scored copy of `run` and a report of what changed. `run` itself is not modified."""
    if run.status != RUN_COMPLETED:
        raise RescoreError(f"run {run.run_id} is {run.status.upper()}; only COMPLETED runs can be re-scored")

    new_items: List[ItemResult] = []
    changed: List[Dict[str, Any]] = []
    kept = 0
    benches = {b: get_benchmark(b, base_dir=base_dir) for b in {i.benchmark for i in run.item_results}}
    for old in run.item_results:
        if old.benchmark in KEPT_BENCHMARKS:
            new_items.append(old)
            kept += 1
            continue
        new = rescore_item(benches[old.benchmark], old)
        new_items.append(new)
        if (new.is_correct, new.extracted_answer, new.error) != (old.is_correct, old.extracted_answer, old.error):
            changed.append({
                "benchmark": old.benchmark, "question_id": old.question_id, "ground_truth": old.ground_truth,
                "old_correct": old.is_correct, "new_correct": new.is_correct,
                "old_extracted": old.extracted_answer, "new_extracted": new.extracted_answer,
                "old_error": old.error, "new_error": new.error, "response_tail": _tail(new.scored_text or new.raw_response),
            })

    summaries: Dict[str, BenchmarkSummary] = {}
    bench_report: Dict[str, Dict[str, Any]] = {}
    for b_id, old_sum in run.benchmark_summaries.items():
        items = [i for i in new_items if i.benchmark == b_id]
        if len(items) != old_sum.total_samples:
            raise RescoreError(f"{run.run_id}: {b_id} has {len(items)} stored items but its summary says {old_sum.total_samples}")
        correct = sum(1 for i in items if i.is_correct)
        flags = [_flags(i) for i in items]
        lo, hi = StatisticsEngine.wilson_confidence_interval(correct, len(items))
        summaries[b_id] = old_sum.model_copy(update={
            "correct_samples": correct, "accuracy": round(correct / len(items), 4) if items else 0.0, "ci_lower": lo, "ci_upper": hi,
            "truncated_count": sum(f[0] for f in flags), "no_answer_count": sum(f[1] for f in flags), "api_error_count": sum(f[2] for f in flags),
        })
        bench_report[b_id] = {"name": old_sum.benchmark_name, "total": len(items), "kept_as_stored": b_id in KEPT_BENCHMARKS,
                              "correct_before": old_sum.correct_samples, "correct_after": correct,
                              "accuracy_before": old_sum.accuracy, "accuracy_after": summaries[b_id].accuracy}

    total_correct = sum(s.correct_samples for s in summaries.values())
    total = sum(s.total_samples for s in summaries.values())
    lo, hi = StatisticsEngine.wilson_confidence_interval(total_correct, total)
    now = datetime.now(timezone.utc).isoformat()
    note = f"RESCORED offline with scorer v{SCORER_VERSION} from run {run.rescored_from or run.run_id}; no model was called."
    new_run = run.model_copy(update={
        "run_id": rescored_run_id(run.run_id), "scorer_version": SCORER_VERSION, "rescored_from": run.rescored_from or run.run_id,
        "rescored_at": now, "description": f"{run.description} {note}".strip(),
        "benchmark_summaries": summaries, "item_results": new_items,
        "overall_accuracy": round(total_correct / total, 4) if total else 0.0, "overall_ci_lower": lo, "overall_ci_upper": hi,
        "truncated_count": sum(s.truncated_count for s in summaries.values()),
        "no_answer_count": sum(s.no_answer_count for s in summaries.values()),
        "api_error_count": sum(s.api_error_count for s in summaries.values()),
    })
    report = {
        "run_id": run.run_id, "new_run_id": new_run.run_id, "model_name": run.model_name, "job_name": run.job_name,
        "run_mode": run.run_mode, "old_scorer_version": run.scorer_version, "new_scorer_version": SCORER_VERSION,
        "overall_before": run.overall_accuracy, "overall_after": new_run.overall_accuracy,
        "benchmarks": bench_report, "changed": changed, "items_kept_as_stored": kept,
        "wrong_to_right": [c for c in changed if c["new_correct"] and not c["old_correct"]],
        "right_to_wrong": [c for c in changed if c["old_correct"] and not c["new_correct"]],
        "other_changes": [c for c in changed if c["old_correct"] == c["new_correct"]],
    }
    return new_run, report


def load_run(runs_dir: str, run_id: str) -> RunArtifact:
    path = os.path.join(runs_dir, f"{run_id}.json")
    if not os.path.isfile(path):
        raise RescoreError(f"no run file {path}")
    with open(path, "r", encoding="utf-8") as f:
        return RunArtifact.model_validate(json.load(f))


def save_rescored(runs_dir: str, new_run: RunArtifact) -> str:
    """Write the new run file (JSON only). Never overwrites a file that is not itself a re-scored run."""
    path = os.path.join(runs_dir, f"{new_run.run_id}.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        if not (isinstance(existing, dict) and existing.get("rescored_from")):
            raise RescoreError(f"{path} exists and is not a re-scored run; refusing to overwrite it")
    return RunStore(runs_dir).save_run(new_run, register=False)


def render_markdown(reports: List[Dict[str, Any]]) -> str:
    out = ["# Offline re-score", "",
           f"Scorer version applied: {SCORER_VERSION}. No model was called. Original run files are untouched.", ""]
    for r in reports:
        out += [f"## {r['model_name']} - {r['run_mode'].upper()} - job {r['job_name']}", "",
                f"Original run `{r['run_id']}` (scorer {r['old_scorer_version']}) -> new run `{r['new_run_id']}` (scorer {r['new_scorer_version']}, RESCORED)", "",
                "| Benchmark | Before | After | Correct before | Correct after | Items |", "| :--- | :---: | :---: | :---: | :---: | :---: |"]
        for b_id, b in r["benchmarks"].items():
            kept = " (kept as stored)" if b["kept_as_stored"] else ""
            out.append(f"| {b['name']}{kept} | {b['accuracy_before']*100:.1f}% | {b['accuracy_after']*100:.1f}% | {b['correct_before']} | {b['correct_after']} | {b['total']} |")
        out += ["", f"Overall: {r['overall_before']*100:.1f}% -> {r['overall_after']*100:.1f}%. "
                    f"Wrong to right: {len(r['wrong_to_right'])}. Right to wrong: {len(r['right_to_wrong'])}. "
                    f"Other changes: {len(r['other_changes'])}.", ""]
        for title, key in (("Wrong -> right", "wrong_to_right"), ("Right -> wrong", "right_to_wrong"), ("Other changes", "other_changes")):
            if r[key]:
                out += [f"### {title}", ""]
                for c in r[key]:
                    out += [f"- `{c['benchmark']}` `{c['question_id']}`: key `{c['ground_truth']}`; extracted `{c['old_extracted']}` -> `{c['new_extracted']}` "
                            f"(error {c['old_error']} -> {c['new_error']})", f"  - end of reply: {c['response_tail']!r}"]
                out.append("")
    return "\n".join(out)


def main(argv: List[str] = None) -> int:
    parser = argparse.ArgumentParser(description="Re-score stored run replies offline with the current scorer.")
    parser.add_argument("run_ids", nargs="+", help="run id(s) of COMPLETED runs")
    parser.add_argument("--runs-dir", default=os.path.join(REPO, "runs"))
    parser.add_argument("--report-md", default=None, help="also write the before/after report to this file")
    parser.add_argument("--report-json", default=None, help="also write the report as JSON to this file")
    args = parser.parse_args(argv)

    reports = []
    try:
        for run_id in args.run_ids:
            run = load_run(args.runs_dir, run_id)
            new_run, report = rescore_artifact(run)
            path = save_rescored(args.runs_dir, new_run)
            reports.append(report)
            print(f"{run_id} -> {new_run.run_id}  ({report['overall_before']*100:.1f}% -> {report['overall_after']*100:.1f}%, "
                  f"{len(report['wrong_to_right'])} wrong->right, {len(report['right_to_wrong'])} right->wrong)  saved to {path}")
    except RescoreError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    if args.report_md:
        with open(args.report_md, "w", encoding="utf-8") as f:
            f.write(render_markdown(reports))
    if args.report_json:
        with open(args.report_json, "w", encoding="utf-8") as f:
            json.dump(reports, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
