import re
from typing import List, Optional, Tuple

from src.scoring import math_answers

# Words that introduce a final answer. The words are case-insensitive; the letter that follows is NOT,
# so ordinary words such as "a" or "I" are never read as options.
_ANSWER_WORD = r"(?i:final\s+answer|correct\s+answer|correct\s+option|correct\s+choice|answer|choice|option)"
_LEAD = r"[\*\[\(\{\"'`\s]*(?:(?i:option|choice)\s+)?"

# Explicit markers, most to least trustworthy. Within a pattern the LAST valid match wins, so a draft
# earlier in the text never beats the final statement.
_MARKER_PATTERNS = [
    re.compile(rf"{_ANSWER_WORD}\s*(?:is\s*[:=]?|[:=]){_LEAD}([A-J])(?![A-Za-z0-9])"),
    re.compile(r"\\boxed\{\s*(?:\\text(?:bf)?\{\s*)?\(?([A-J])\)?\s*\}"),
    re.compile(r"\*\*\s*\(?([A-J])\)?\s*\*\*"),
    re.compile(r"\*\*\s*\(?([A-J])[\.\):]\s+[^*\n]*\*\*"),
    # "Option A is clearly the choice" / "Option B is correct" (but not "Option B is wrong")
    re.compile(
        r"(?i:option|choice)\s+([A-J])\s+(?i:is|seems|appears\s+to\s+be)\s+"
        r"(?i:(?:clearly|definitely|obviously)\s+)?(?i:the\s+)?(?i:correct|right|best|true|answer|choice)\b"
    ),
]

# A line that is nothing but an answer: "C", "(C)", "**C**", "C.", "C. Newton's law", "Option C".
_LINE_PATTERNS = [
    re.compile(r"^[\W_]*\(?([A-J])\)?[\W_]*$"),
    re.compile(r"^[\W_]*\(?([A-J])[\.\):]\s+\S"),
    re.compile(r"^[\W_]*(?i:option|choice)\s+([A-J])[\W_]*$"),
]


def _last_valid(pattern: "re.Pattern", text: str, valid_set: set) -> Optional[str]:
    for match in reversed(list(pattern.finditer(text))):
        if match.group(1) in valid_set:
            return match.group(1)
    return None


def _letter_from_line(line: str, valid_set: set) -> Optional[str]:
    for pattern in _LINE_PATTERNS:
        match = pattern.match(line)
        if match and match.group(1) in valid_set:
            return match.group(1)
    return None


class RuleScorers:
    @staticmethod
    def extract_mcq_answer(response: str, valid_options: str = "ABCDEFGHIJ") -> Optional[str]:
        """Pull the final option letter out of an already-stripped answer text.

        Callers must remove reasoning text first (see src/scoring/reasoning.py). Only an explicit
        answer marker or an answer-only line is accepted; a stray letter inside prose is never used.
        """
        if not response:
            return None
        text = response.strip()
        valid_set = set(valid_options.upper())

        for pattern in _MARKER_PATTERNS:
            found = _last_valid(pattern, text, valid_set)
            if found:
                return found

        lines: List[str] = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if lines:
            for line in (lines[-1], lines[0]):
                found = _letter_from_line(line, valid_set)
                if found:
                    return found
        return None

    @staticmethod
    def score_mcq(response: str, ground_truth: str, valid_options: str = "ABCD") -> Tuple[bool, Optional[str]]:
        extracted = RuleScorers.extract_mcq_answer(response, valid_options)
        if not extracted or not ground_truth:
            return False, extracted
        return extracted.strip().upper() == ground_truth.strip().upper(), extracted

    @staticmethod
    def extract_math_answer(response: str) -> Optional[str]:
        """Pull the final answer out of an already-stripped reply (rules in src/scoring/math_answers.py)."""
        return math_answers.extract_math_answer(response)

    @staticmethod
    def normalize_math_str(val: str) -> str:
        return math_answers.normalize_math_str(val)

    @staticmethod
    def score_math(response: str, ground_truth: str) -> Tuple[bool, Optional[str]]:
        return math_answers.score_math(response, ground_truth)
