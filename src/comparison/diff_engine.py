from typing import Dict, Any, List
from src.storage.schema import RunArtifact, ItemResult
from src.scoring.statistics import StatisticsEngine

class DiffEngine:
    @staticmethod
    def compare_runs(run_a: RunArtifact, run_b: RunArtifact) -> Dict[str, Any]:
        # Aggregate comparison
        delta_acc = run_b.overall_accuracy - run_a.overall_accuracy
        delta_lat = run_b.mean_latency - run_a.mean_latency
        delta_cost = run_b.total_cost - run_a.total_cost

        # Per benchmark breakdown
        bench_comparison = {}
        all_benches = set(list(run_a.benchmark_summaries.keys()) + list(run_b.benchmark_summaries.keys()))

        for b_id in all_benches:
            sum_a = run_a.benchmark_summaries.get(b_id)
            sum_b = run_b.benchmark_summaries.get(b_id)

            acc_a = sum_a.accuracy if sum_a else 0.0
            acc_b = sum_b.accuracy if sum_b else 0.0
            bench_comparison[b_id] = {
                "unreliable_a": bool(sum_a and sum_a.unreliable),
                "unreliable_b": bool(sum_b and sum_b.unreliable),
                "unreliable_fraction_a": sum_a.unreliable_fraction if sum_a else 0.0,
                "unreliable_fraction_b": sum_b.unreliable_fraction if sum_b else 0.0,
                "benchmark_name": sum_a.benchmark_name if sum_a else (sum_b.benchmark_name if sum_b else b_id),
                "domain": sum_a.domain if sum_a else (sum_b.domain if sum_b else "General"),
                "accuracy_a": acc_a,
                "accuracy_b": acc_b,
                "delta": round(acc_b - acc_a, 4),
                "ci_a": (sum_a.ci_lower, sum_a.ci_upper) if sum_a else (0.0, 0.0),
                "ci_b": (sum_b.ci_lower, sum_b.ci_upper) if sum_b else (0.0, 0.0),
                "mean_latency_a": sum_a.mean_latency if sum_a else 0.0,
                "mean_latency_b": sum_b.mean_latency if sum_b else 0.0
            }

        # Item-by-item alignment & flip detection
        items_a_map = {it.question_id: it for it in run_a.item_results}
        items_b_map = {it.question_id: it for it in run_b.item_results}

        common_keys = [k for k in items_a_map if k in items_b_map]

        a_results = [items_a_map[k].is_correct for k in common_keys]
        b_results = [items_b_map[k].is_correct for k in common_keys]

        # McNemar test
        mcnemar = StatisticsEngine.mcnemar_test(a_results, b_results)

        # Flips
        a_wins_flips = []
        b_wins_flips = []
        both_correct = []
        both_failed = []

        for k in common_keys:
            ia = items_a_map[k]
            ib = items_b_map[k]

            flip_item = {
                "question_id": k,
                "benchmark": ia.benchmark,
                "domain": ia.domain,
                "prompt": ia.prompt,
                "ground_truth": ia.ground_truth,
                "response_a": ia.raw_response,
                "extracted_a": ia.extracted_answer,
                "response_b": ib.raw_response,
                "extracted_b": ib.extracted_answer,
                "latency_a": ia.latency_sec,
                "latency_b": ib.latency_sec
            }

            if ia.is_correct and not ib.is_correct:
                a_wins_flips.append(flip_item)
            elif not ia.is_correct and ib.is_correct:
                b_wins_flips.append(flip_item)
            elif ia.is_correct and ib.is_correct:
                both_correct.append(k)
            else:
                both_failed.append(k)

        return {
            "run_a_id": run_a.run_id,
            "model_a_id": run_a.model_id,
            "model_a_name": run_a.model_name,
            "served_model_id_a": run_a.served_model_id or run_a.model_id,
            "model_substituted_a": run_a.model_substituted,
            "route_note_a": run_a.route_note or "",
            "run_mode_a": run_a.run_mode,
            "run_mode_b": run_b.run_mode,
            "job_name_a": run_a.job_name,
            "job_name_b": run_b.job_name,
            "standin_a": run_a.is_standin,
            "standin_b": run_b.is_standin,
            "truncated_a": run_a.truncated_count,
            "no_answer_a": run_a.no_answer_count,
            "api_error_a": run_a.api_error_count,
            "failed_a": run_a.is_failed,
            "failed_b": run_b.is_failed,
            "incomplete_a": run_a.is_incomplete,
            "incomplete_b": run_b.is_incomplete,
            "scorer_version_a": run_a.scorer_version,
            "scorer_version_b": run_b.scorer_version,
            "rescored_a": run_a.is_rescored,
            "rescored_b": run_b.is_rescored,
            "rescored_from_a": run_a.rescored_from,
            "rescored_from_b": run_b.rescored_from,
            "unreliable_benchmarks_a": run_a.unreliable_benchmarks,
            "unreliable_benchmarks_b": run_b.unreliable_benchmarks,
            "unreliable_a": run_a.unreliable,
            "unreliable_fraction_a": run_a.unreliable_fraction,
            "unreliable_b": run_b.unreliable,
            "unreliable_fraction_b": run_b.unreliable_fraction,
            "truncated_b": run_b.truncated_count,
            "no_answer_b": run_b.no_answer_count,
            "api_error_b": run_b.api_error_count,
            "run_b_id": run_b.run_id,
            "model_b_id": run_b.model_id,
            "model_b_name": run_b.model_name,
            "served_model_id_b": run_b.served_model_id or run_b.model_id,
            "model_substituted_b": run_b.model_substituted,
            "route_note_b": run_b.route_note or "",
            "accuracy_a": run_a.overall_accuracy,
            "accuracy_b": run_b.overall_accuracy,
            "delta_accuracy": round(delta_acc, 4),
            "delta_latency": round(delta_lat, 3),
            "delta_cost": round(delta_cost, 6),
            "benchmarks": bench_comparison,
            "mcnemar": mcnemar,
            "total_aligned_items": len(common_keys),
            "a_only_correct_count": len(a_wins_flips),
            "b_only_correct_count": len(b_wins_flips),
            "both_correct_count": len(both_correct),
            "both_failed_count": len(both_failed),
            "a_wins_flips": a_wins_flips,
            "b_wins_flips": b_wins_flips
        }
