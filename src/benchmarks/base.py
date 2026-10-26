import os
from typing import List, Dict, Any, Tuple, Optional
from src.benchmarks.loader import BenchmarkLoader
from src.scoring.rule_scorers import RuleScorers
from src.scoring.code_extraction import extract_python_code
from src.engine.sandbox import CodeSandbox

class BaseBenchmark:
    def __init__(self, bench_id: str, name: str, domain: str, sample_file: str, base_dir: str = ""):
        self.bench_id = bench_id
        self.name = name
        self.domain = domain
        self.sample_file = os.path.join(base_dir, sample_file) if base_dir else sample_file

    def load_items(self, limit: int = 0) -> List[Dict[str, Any]]:
        return BenchmarkLoader.load_jsonl(self.sample_file, limit=limit)

    def evaluate(self, item: Dict[str, Any], raw_response: str) -> Tuple[bool, float, Optional[str], Optional[str]]:
        raise NotImplementedError

class MMLUProBenchmark(BaseBenchmark):
    def __init__(self, base_dir: str = ""):
        super().__init__("mmlu_pro", "MMLU-Pro", "General Knowledge", "datasets/mmlu_pro/sample.jsonl", base_dir)

    def evaluate(self, item: Dict[str, Any], raw_response: str) -> Tuple[bool, float, Optional[str], Optional[str]]:
        gt = item.get("answer", "").strip()
        correct, ext = RuleScorers.score_mcq(raw_response, gt, valid_options="ABCDEFGHIJ")
        return correct, 1.0 if correct else 0.0, ext, None

class GSM8KBenchmark(BaseBenchmark):
    def __init__(self, base_dir: str = ""):
        super().__init__("gsm8k", "GSM8K", "Mathematical Reasoning", "datasets/gsm8k/sample.jsonl", base_dir)

    def evaluate(self, item: Dict[str, Any], raw_response: str) -> Tuple[bool, float, Optional[str], Optional[str]]:
        gt = str(item.get("answer", "")).strip()
        correct, ext = RuleScorers.score_math(raw_response, gt)
        return correct, 1.0 if correct else 0.0, ext, None

class MATH500Benchmark(BaseBenchmark):
    def __init__(self, base_dir: str = ""):
        super().__init__("math_500", "MATH-500", "Mathematical Reasoning", "datasets/math_500/sample.jsonl", base_dir)

    def evaluate(self, item: Dict[str, Any], raw_response: str) -> Tuple[bool, float, Optional[str], Optional[str]]:
        gt = str(item.get("answer", "")).strip()
        correct, ext = RuleScorers.score_math(raw_response, gt)
        return correct, 1.0 if correct else 0.0, ext, None

class GPQABenchmark(BaseBenchmark):
    def __init__(self, base_dir: str = ""):
        super().__init__("gpqa", "GPQA", "Logical Reasoning", "datasets/gpqa/sample.jsonl", base_dir)

    def evaluate(self, item: Dict[str, Any], raw_response: str) -> Tuple[bool, float, Optional[str], Optional[str]]:
        gt = item.get("answer", "").strip()
        correct, ext = RuleScorers.score_mcq(raw_response, gt, valid_options="ABCD")
        return correct, 1.0 if correct else 0.0, ext, None

class TruthfulQABenchmark(BaseBenchmark):
    def __init__(self, base_dir: str = ""):
        super().__init__("truthfulqa", "TruthfulQA", "Question Answering", "datasets/truthfulqa/sample.jsonl", base_dir)

    def evaluate(self, item: Dict[str, Any], raw_response: str) -> Tuple[bool, float, Optional[str], Optional[str]]:
        gt = item.get("answer", "").strip()
        correct, ext = RuleScorers.score_mcq(raw_response, gt, valid_options="ABCD")
        return correct, 1.0 if correct else 0.0, ext, None

class CustomBYOBBenchmark(BaseBenchmark):
    def __init__(self, base_dir: str = ""):
        super().__init__("custom_byob", "Custom BYOB", "Custom Evaluation", "datasets/custom_byob/sample.jsonl", base_dir)

    def evaluate(self, item: Dict[str, Any], raw_response: str) -> Tuple[bool, float, Optional[str], Optional[str]]:
        gt = item.get("answer", "").strip()
        correct, ext = RuleScorers.score_mcq(raw_response, gt, valid_options="ABCD")
        return correct, 1.0 if correct else 0.0, ext, None

class HumanEvalBenchmark(BaseBenchmark):
    def __init__(self, base_dir: str = ""):
        super().__init__("humaneval", "HumanEval", "Code Generation", "datasets/humaneval/sample.jsonl", base_dir)

    def evaluate(self, item: Dict[str, Any], raw_response: str) -> Tuple[bool, float, Optional[str], Optional[str]]:
        test_code = item.get("test", "")
        entry_point = item.get("entry_point", "")
        code = extract_python_code(raw_response, item.get("prompt", ""))
        if code is None:
            return False, 0.0, None, None  # no code in the reply; the runner flags it as no_answer_found
        passed, err = CodeSandbox.execute_code(code, test_code, entry_point, timeout_seconds=3.0)
        return passed, 1.0 if passed else 0.0, "Executed" if passed else "Failed", err

class KYCIndiaBenchmark(BaseBenchmark):
    def __init__(self, base_dir: str = ""):
        super().__init__("kyc_india", "KYC India", "Compliance & Regulation", "datasets/kyc_india/sample.jsonl", base_dir)

    def evaluate(self, item: Dict[str, Any], raw_response: str) -> Tuple[bool, float, Optional[str], Optional[str]]:
        gt = item.get("answer", "").strip()
        correct, ext = RuleScorers.score_mcq(raw_response, gt, valid_options="ABCD")
        return correct, 1.0 if correct else 0.0, ext, None

class RBIRegulationsBenchmark(BaseBenchmark):
    def __init__(self, base_dir: str = ""):
        super().__init__("rbi_regulations", "Bank Regulation (RBI)", "Banking & Regulation", "datasets/rbi_regulations/sample.jsonl", base_dir)

    def evaluate(self, item: Dict[str, Any], raw_response: str) -> Tuple[bool, float, Optional[str], Optional[str]]:
        gt = item.get("answer", "").strip()
        correct, ext = RuleScorers.score_mcq(raw_response, gt, valid_options="ABCD")
        return correct, 1.0 if correct else 0.0, ext, None

class ImageProcessingBenchmark(BaseBenchmark):
    def __init__(self, base_dir: str = ""):
        super().__init__("image_processing", "Image Processing", "Computer Vision", "datasets/image_processing/sample.jsonl", base_dir)

    def evaluate(self, item: Dict[str, Any], raw_response: str) -> Tuple[bool, float, Optional[str], Optional[str]]:
        gt = item.get("answer", "").strip()
        correct, ext = RuleScorers.score_mcq(raw_response, gt, valid_options="ABCD")
        return correct, 1.0 if correct else 0.0, ext, None

def get_benchmark(bench_id: str, base_dir: str = "") -> BaseBenchmark:
    mapping = {
        "mmlu_pro": MMLUProBenchmark,
        "gsm8k": GSM8KBenchmark,
        "math_500": MATH500Benchmark,
        "gpqa": GPQABenchmark,
        "humaneval": HumanEvalBenchmark,
        "truthfulqa": TruthfulQABenchmark,
        "custom_byob": CustomBYOBBenchmark,
        "kyc_india": KYCIndiaBenchmark,
        "rbi_regulations": RBIRegulationsBenchmark,
        "image_processing": ImageProcessingBenchmark
    }
    cls = mapping.get(bench_id)
    if not cls:
        raise ValueError(f"Unknown benchmark: {bench_id}. Available: {list(mapping.keys())}")
    return cls(base_dir=base_dir)
