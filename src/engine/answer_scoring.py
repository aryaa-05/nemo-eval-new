from dataclasses import dataclass
from typing import Any, Dict, Optional

from src.scoring.reasoning import UNCLOSED, strip_reasoning

TRUNCATION_FINISH_REASONS = ("length", "max_tokens", "max_output_tokens")

ERR_TRUNCATED = "truncated"
ERR_NO_ANSWER = "no_answer_found"


@dataclass
class ScoredGeneration:
    is_correct: bool
    score: float
    extracted: Optional[str]
    error: Optional[str]
    scored_text: Optional[str]  # the text that was scored, only set when it differs from the raw response
    truncated: bool = False
    no_answer: bool = False
    api_error: bool = False


def score_generation(bench: Any, item: Dict[str, Any], gen_resp: Dict[str, Any]) -> ScoredGeneration:
    """Score one model response.

    Order matters: API failures and truncated outputs are never sent to the extractor, reasoning text is
    stripped before extraction, and anything with no usable answer is flagged and counted incorrect.
    """
    raw = gen_resp.get("text") or ""

    if gen_resp.get("error"):
        return ScoredGeneration(False, 0.0, None, f"api_error: {gen_resp['error']}", None, api_error=True)

    if gen_resp.get("finish_reason") in TRUNCATION_FINISH_REASONS:
        return ScoredGeneration(False, 0.0, None, ERR_TRUNCATED, None, truncated=True)

    answer_text, status = strip_reasoning(raw)
    scored_text = answer_text if answer_text != raw else None

    if status == UNCLOSED or not answer_text:
        return ScoredGeneration(False, 0.0, None, ERR_NO_ANSWER, scored_text, no_answer=True)

    is_correct, score, extracted, error = bench.evaluate(item, answer_text)
    if extracted is None and error is None:
        return ScoredGeneration(False, 0.0, None, ERR_NO_ANSWER, scored_text, no_answer=True)
    return ScoredGeneration(is_correct, score, extracted, error, scored_text)
