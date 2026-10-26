import hashlib
import time
from typing import Dict, Any, Optional

class MockModelEngine:
    MODEL_PROFILES = {
        "meta/llama-3.1-70b-instruct": {
            "accuracy_bias": {"mmlu_pro": 0.82, "gsm8k": 0.88, "math_500": 0.62, "gpqa": 0.60, "humaneval": 0.80, "truthfulqa": 0.85, "custom_byob": 0.90},
            "mean_latency": 0.75,
            "cost_in": 0.88,
            "cost_out": 0.88,
        },
        "meta/llama-3.1-8b-instruct": {
            "accuracy_bias": {"mmlu_pro": 0.64, "gsm8k": 0.74, "math_500": 0.44, "gpqa": 0.42, "humaneval": 0.62, "truthfulqa": 0.72, "custom_byob": 0.70},
            "mean_latency": 0.28,
            "cost_in": 0.18,
            "cost_out": 0.18,
        },
        "qwen/qwen-2.5-72b-instruct": {
            "accuracy_bias": {"mmlu_pro": 0.85, "gsm8k": 0.92, "math_500": 0.75, "gpqa": 0.65, "humaneval": 0.88, "truthfulqa": 0.82, "custom_byob": 0.88},
            "mean_latency": 0.82,
            "cost_in": 0.90,
            "cost_out": 0.90,
        },
        "mistralai/mixtral-8x7b-instruct": {
            "accuracy_bias": {"mmlu_pro": 0.72, "gsm8k": 0.78, "math_500": 0.50, "gpqa": 0.48, "humaneval": 0.68, "truthfulqa": 0.76, "custom_byob": 0.75},
            "mean_latency": 0.48,
            "cost_in": 0.60,
            "cost_out": 0.60,
        },
        "deepseek/deepseek-r1-distill-llama-70b": {
            "accuracy_bias": {"mmlu_pro": 0.86, "gsm8k": 0.94, "math_500": 0.84, "gpqa": 0.68, "humaneval": 0.84, "truthfulqa": 0.80, "custom_byob": 0.85},
            "mean_latency": 1.45,
            "cost_in": 0.80,
            "cost_out": 0.80,
        },
        "openai/gpt-4o-mini": {
            "accuracy_bias": {"mmlu_pro": 0.78, "gsm8k": 0.84, "math_500": 0.58, "gpqa": 0.52, "humaneval": 0.74, "truthfulqa": 0.82, "custom_byob": 0.80},
            "mean_latency": 0.35,
            "cost_in": 0.15,
            "cost_out": 0.60,
        }
    }

    @staticmethod
    def item_key(item_meta: Optional[Dict[str, Any]], prompt: str) -> str:
        """Stable identifier for an item. HumanEval uses task_id; if neither id exists, hash the prompt."""
        if item_meta:
            key = item_meta.get("question_id") or item_meta.get("task_id")
            if key:
                return str(key)
        return "prompt-" + hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def roll(model_id: str, benchmark_id: str, item_key: str, seed: int = 42) -> int:
        """Deterministic integer in [0, 10000) per (seed, model, benchmark, item); stable across processes."""
        seed_str = f"{seed}|{model_id}|{benchmark_id}|{item_key}"
        return int(hashlib.sha256(seed_str.encode("utf-8")).hexdigest(), 16) % 10000

    @classmethod
    def generate_response(cls, model_id: str, prompt: str, benchmark_id: str, ground_truth: str = "", item_meta: Dict[str, Any] = None, seed: int = 42) -> Dict[str, Any]:
        profile = cls.MODEL_PROFILES.get(model_id, cls.MODEL_PROFILES["meta/llama-3.1-70b-instruct"])
        bench_acc = profile["accuracy_bias"].get(benchmark_id, 0.75)

        # Hash-based deterministic roll per question so runs are reproducible
        roll_int = cls.roll(model_id, benchmark_id, cls.item_key(item_meta, prompt), seed)
        rand_val = roll_int / 10000.0
        should_succeed = (rand_val < bench_acc)

        latency = max(0.05, profile["mean_latency"] * (0.85 + (rand_val % 0.3)))
        time.sleep(min(0.05, latency * 0.05))  # simulate small delay

        prompt_tokens = len(prompt.split()) + 25
        
        # Build plausible model outputs
        if benchmark_id in ["mmlu_pro", "gpqa", "truthfulqa", "custom_byob", "kyc_india", "rbi_regulations", "image_processing"]:
            valid_opts = item_meta.get("options", ["A", "B", "C", "D"]) if item_meta else ["A", "B", "C", "D"]
            num_opts = len(valid_opts)
            gt_letter = ground_truth.strip().upper() if ground_truth else "A"
            if should_succeed:
                chosen = gt_letter
                explanation = f"Analyzing the problem systematically. Given the conditions, the only logically sound choice is {chosen}."
            else:
                other_opts = [chr(65+i) for i in range(num_opts) if chr(65+i) != gt_letter]
                chosen = other_opts[roll_int % len(other_opts)] if other_opts else "B"
                explanation = f"After considering the premises, option {chosen} appears to be consistent with common approximations."
            
            if "deepseek" in model_id:
                raw_text = f"<think>\nLet's evaluate each option carefully.\n...\n</think>\n\nBased on comprehensive analysis, the correct answer is {chosen}."
            else:
                raw_text = f"{explanation}\n\nAnswer: {chosen}"
            completion_tokens = len(raw_text.split()) + 15

        elif benchmark_id in ["gsm8k", "math_500"]:
            if should_succeed:
                ans_str = ground_truth
                steps = f"Step 1: Parse the equation and parameters.\nStep 2: Carry out intermediate arithmetic reduction.\nFinal value = {ans_str}"
            else:
                try:
                    num = float(ground_truth)
                    ans_str = str(int(num + 3) if num.is_integer() else round(num + 2.5, 2))
                except Exception:
                    ans_str = ground_truth + "_err"
                steps = f"Let's calculate the value.\nStep 1: Simplify terms.\nWe obtain {ans_str}."
            
            if "deepseek" in model_id:
                raw_text = f"<think>\nStep by step mathematical reasoning:\n{steps}\n</think>\n\nThe final answer is \\boxed{{{ans_str}}}."
            else:
                raw_text = f"{steps}\n\n#### {ans_str}"
            completion_tokens = len(raw_text.split()) + 20

        elif benchmark_id == "humaneval":
            canonical = item_meta.get("canonical_solution", "    return True\n") if item_meta else "    return True\n"
            entry_point = item_meta.get("entry_point", "solve") if item_meta else "solve"
            stub = item_meta.get("prompt", prompt) if item_meta else prompt  # the function stub without the instruction line
            if should_succeed:
                solution = canonical
            else:
                # Deliberate minor bug
                solution = canonical.replace("==", "!=").replace("<", ">") if "==" in canonical or "<" in canonical else "    return False\n"
            # Reply the way a chat model does: a short lead-in and a fenced code block
            raw_text = f"Here is the completed function:\n\n```python\n{stub}{solution}```\n"
            completion_tokens = len(raw_text.split()) + 25
        else:
            raw_text = f"Evaluated response for {model_id} on {benchmark_id}. Output: {ground_truth if should_succeed else 'Incorrect'}"
            completion_tokens = 50

        cost = (prompt_tokens / 1_000_000.0 * profile["cost_in"]) + (completion_tokens / 1_000_000.0 * profile["cost_out"])

        return {
            "text": raw_text,
            "finish_reason": "stop",
            "latency_sec": round(latency, 3),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "cost": round(cost, 6)
        }
