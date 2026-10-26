import math
from typing import Dict, List, Tuple, Any
from scipy import stats

class StatisticsEngine:
    @staticmethod
    def wilson_confidence_interval(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
        if n == 0:
            return (0.0, 0.0)
        z = stats.norm.ppf((1 + confidence) / 2)
        p = k / n
        denominator = 1 + (z ** 2) / n
        centre = (p + (z ** 2) / (2 * n)) / denominator
        margin = (z * math.sqrt((p * (1 - p) / n) + (z ** 2) / (4 * (n ** 2)))) / denominator
        lower = max(0.0, centre - margin)
        upper = min(1.0, centre + margin)
        return (round(lower, 4), round(upper, 4))

    @staticmethod
    def mcnemar_test(results_a: List[bool], results_b: List[bool]) -> Dict[str, Any]:
        if len(results_a) != len(results_b):
            raise ValueError("Both result sets must have identical sample length for paired McNemar test")
        
        n = len(results_a)
        if n == 0:
            return {"statistic": 0.0, "p_value": 1.0, "significant": False, "table": [[0,0],[0,0]]}

        b = sum(1 for a, b in zip(results_a, results_b) if a and not b)
        c = sum(1 for a, b in zip(results_a, results_b) if not a and b)

        if (b + c) == 0:
            chi2 = 0.0
            p_val = 1.0
        else:
            chi2 = ((abs(b - c) - 1) ** 2) / (b + c)
            p_val = 1.0 - stats.chi2.cdf(chi2, df=1)

        return {
            "statistic": round(float(chi2), 4),
            "p_value": round(float(p_val), 5),
            "significant": bool(p_val < 0.05),
            "a_better_than_b_count": b,
            "b_better_than_a_count": c,
            "total_flips": b + c
        }

    @staticmethod
    def evaluate_quality_gate(baseline_summary: Dict[str, Any], candidate_summary: Dict[str, Any], min_accuracy: float = 0.65, max_regression: float = 0.03) -> Dict[str, Any]:
        base_acc = baseline_summary.get("overall_accuracy", 0.0)
        cand_acc = candidate_summary.get("overall_accuracy", 0.0)
        delta = cand_acc - base_acc

        reasons = []
        status = "PASS"

        if cand_acc < min_accuracy:
            status = "FAIL"
            reasons.append(f"Candidate accuracy ({cand_acc:.1%}) below required minimum ({min_accuracy:.1%})")

        if delta < -max_regression:
            status = "FAIL"
            reasons.append(f"Candidate regression delta ({delta:+.1%}) exceeds maximum allowable regression (-{max_regression:.1%})")

        if status == "PASS":
            reasons.append(f"Candidate achieves {cand_acc:.1%} accuracy (Delta: {delta:+.1%}) meeting all deployment criteria")

        return {
            "status": status,
            "baseline_accuracy": round(base_acc, 4),
            "candidate_accuracy": round(cand_acc, 4),
            "accuracy_delta": round(delta, 4),
            "rationale": "; ".join(reasons)
        }
