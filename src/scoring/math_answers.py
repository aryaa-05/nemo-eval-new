"""Extract and compare the final answer of a math reply (GSM8K, MATH-500).

How the final answer is found, in order (the first rule that finds a value wins):
  1. the last \\boxed{...}
  2. the last "#### value"
  3. the text after the LAST explicit label ("Answer:", "The answer is", "Final answer", "Conclusion:", or a
     "### Conclusion" heading): a value written right after the label, else a bold value, else the value after the last
     "=" or after an "is/are/has ..." cue (a restating cue such as "simplifies to" makes the later value win), else the
     first value (the last one if the sentence contains a negation). "X or Y" with different values is a hedge: no answer
  4. otherwise the last sentences of the reply, with the same bold / "=" / cue rules, then the last value
  5. otherwise nothing: the reply is flagged no_answer_found. There is deliberately NO "last number anywhere in the
     reply" fallback, because that would let a number from the working be mistaken for the answer.

Two answers are equal when their normalised text is equal or when both evaluate to the same number (plain numbers,
fractions a/b, and radicals such as 9*sqrt(3)). Units are never converted: 0.5 hours is not 30 minutes, 25% is not 0.25.
"""
import re
from typing import List, Optional, Tuple

_NUMBER = r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
_VALUE = re.compile(
    rf"(?P<rad>{_NUMBER}\*sqrt\(\d+\)|-?sqrt\(\d+\))|(?P<frac>-?\d+(?:\.\d+)?/\d+(?:\.\d+)?)|(?P<num>{_NUMBER})"
)
_SKIP = " \t\r\n*_`([:\\"
# Verbs that introduce the stated result: "the revenue is 1,680", "Weng earns $10", "Kendra has 45 pencils left".
_CUE = re.compile(
    r"(?i)\b(?:is|are|was|were|has|have|had|equals?|gives?|gets?|makes?|totals?|earns?|costs?|pays?|spends?|sells?|"
    r"sold|needs?|runs?|ran|travels?|walks?|drives?|saves?|saved|becomes?|amounts?\s+to|comes?\s+to|results?\s+in|"
    r"simplif(?:y|ies|ied)|reduc(?:e|es|ed)|equivalent\s+to|(?:written|rewritten|expressed)\s+as)\b"
)
# "6 or 7": a hedge between two different values is not an answer
_OR = re.compile(r"(?i)\s*,?\s*(?:and/or|or)\s+(?:maybe\s+|perhaps\s+)?")
# a clause that starts after one of these is a side remark ("..., which occurs at x = 2")
_CLAUSE_BREAK = re.compile(r"(?i)[,;]|\bwhich\b|\bwhere\b|\bwhen\b|\bwhile\b|\bbecause\b|\bsince\b")
_NEGATION = re.compile(r"(?i)\b(?:not|isn't|aren't|wasn't|instead\s+of|rather\s+than|incorrect|wrong|mistake)\b")
_LABEL = re.compile(
    r"(?i)(?:final\s+answer|answer|final\s+conclusion|conclusion)\s*(?:\*\*|__)?\s*(?::|\bis\b)\s*(?:\*\*|__)?"
)
_HEADING = re.compile(r"(?im)^[ \t]*#{1,6}[ \t]*(?:final[ \t]+)?(?:answer|conclusion)s?[ \t]*:?[ \t]*$")
_BOLD = re.compile(r"\*\*(.+?)\*\*", re.S)
# closing chatter that is never a statement of the answer ("Hope this helps with the 10 other problems!")
_CHATTER = re.compile(
    r"(?i)\b(?:hope|let\s+me\s+know|feel\s+free|happy\s+to|glad\s+to|if\s+you\s+(?:have|need|want|would)|"
    r"any\s+other|anything\s+else|further\s+(?:help|questions?)|questions?)\b"
)


def last_boxed(text: str) -> Optional[str]:
    """Contents of the last \\boxed{...}, honouring nested braces such as \\boxed{\\frac{1}{2}}."""
    starts = [m.end() for m in re.finditer(r"\\boxed\{", text)]
    for start in reversed(starts):
        depth = 1
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    inner = text[start:i].strip()
                    if inner:
                        return inner
                    break
    return None


def _prep(s: str) -> str:
    """Flatten LaTeX and currency so numbers can be found: \\frac{1}{6} -> 1/6, 9\\sqrt{3} -> 9*sqrt(3), \\$1{,}680 -> 1,680."""
    s = s.replace("\u2212", "-").replace("\u2013", "-")
    s = s.replace("\\$", "$").replace("{,}", ",")
    for token in ("\\displaystyle", "\\left", "\\right", "\\qquad", "\\quad"):
        s = s.replace(token, " ")
    s = re.sub(r"\\[,;:! ]", " ", s)
    for _ in range(2):
        s = re.sub(r"\\(?:text|mathrm|mathbf|textbf)\{([^{}]*)\}", r" \1 ", s)
    s = re.sub(r"\\[dt]?frac\{\s*(-?\d+(?:\.\d+)?)\s*\}\{\s*(\d+(?:\.\d+)?)\s*\}", r"\1/\2", s)
    s = re.sub(r"(\d)\s*(?:\\sqrt\{\s*(\d+)\s*\}|\u221a\s*\(?\s*(\d+)\s*\)?)",
               lambda m: f"{m.group(1)}*sqrt({m.group(2) or m.group(3)})", s)
    s = re.sub(r"\\sqrt\{\s*(\d+)\s*\}|\u221a\s*\(?\s*(\d+)\s*\)?",
               lambda m: f"sqrt({m.group(1) or m.group(2)})", s)
    s = re.sub(r"(\d)\s*\*?\s*sqrt\((\d+)\)", r"\1*sqrt(\2)", s)
    for token in ("\\(", "\\)", "\\[", "\\]", "$$"):
        s = s.replace(token, " ")
    return s.replace("$", "")


def _values(s: str) -> List[Tuple[int, str]]:
    return [(m.start(), m.group(0)) for m in _VALUE.finditer(s)]


def _value_at(s: str, pos: int) -> Optional[Tuple[int, str]]:
    while pos < len(s) and s[pos] in _SKIP:
        pos += 1
    m = _VALUE.match(s, pos)
    return (m.start(), m.group(0)) if m else None


def _value_after_cue(s: str, pos: int) -> Optional[Tuple[int, str]]:
    """The value that follows a cue verb, allowing up to three plain words in between ("has saved about 45")."""
    for _ in range(4):
        hit = _value_at(s, pos)
        if hit:
            return hit
        m = re.match(r"[\s*_`(\[:\\]*[A-Za-z']+", s[pos:])
        if not m:
            return None
        pos += m.end()
    return None


def _fragment_value(fragment: str) -> Optional[str]:
    """The value a bold fragment states: the right-hand side of its last '=', else its first value."""
    p = _prep(fragment)
    if "=" in p:
        rhs = _values(p.rsplit("=", 1)[1])
        if rhs:
            return rhs[0][1]
    vals = _values(p)
    return vals[0][1] if vals else None


def _hedged(s: str, hit: Tuple[int, str]) -> bool:
    """True if the value is immediately followed by "or <a different value>"."""
    rest = s[hit[0] + len(hit[1]):]
    m = _OR.match(rest)
    if not m:
        return False
    other = _value_at(rest, m.end())
    return bool(other) and not _same_value(hit[1], other[1])


def _from_sentence(sentence: str, last_value_fallback: bool) -> Optional[str]:
    s = _prep(sentence)
    # 1. a bold value is the author's emphasis on the answer
    bold = [v for v in (_fragment_value(m.group(1)) for m in _BOLD.finditer(s)) if v]
    if bold:
        return bold[-1]
    # 2. the value after the last '=' or after a cue verb; whichever comes later in the sentence, except that an
    #    '=' inside a trailing side clause ("is 3, which occurs at x = 2") never overrides the main statement
    equals_candidate: Optional[Tuple[int, str]] = None
    cue_candidates: List[Tuple[int, str]] = []
    eq = s.rfind("=")
    if eq != -1:
        rhs = _values(s[eq + 1:])
        if rhs:
            equals_candidate = (eq + 1 + rhs[0][0], rhs[0][1])
    for m in _CUE.finditer(s):
        hit = _value_after_cue(s, m.end())
        if hit:
            cue_candidates.append(hit)
    if equals_candidate or cue_candidates:
        if equals_candidate and cue_candidates:
            breaks = [b.end() for b in _CLAUSE_BREAK.finditer(s[:eq])]
            clause_start = max(breaks) if breaks else -1
            before = [c for c in cue_candidates if c[0] < clause_start]
            after = [c for c in cue_candidates if c[0] >= clause_start]
            if before and not after:
                chosen = before[-1]           # '=' is in a side clause: the main statement wins
            else:
                chosen = max([equals_candidate] + cue_candidates)
        else:
            chosen = equals_candidate or max(cue_candidates)
        return None if _hedged(s, chosen) else chosen[1]
    # 3. any value in the sentence
    vals = _values(s)
    if not vals:
        return None
    use_last = last_value_fallback or bool(_NEGATION.search(s))
    return vals[-1][1] if use_last else vals[0][1]


def _same_value(a: str, b: str) -> bool:
    na, nb = _number(normalize_math_str(a)), _number(normalize_math_str(b))
    return normalize_math_str(a) == normalize_math_str(b) or (na is not None and nb is not None and abs(na - nb) <= 1e-9)


def _contradicted_later(prepared: str, hit: Tuple[int, str]) -> bool:
    """True if, after the value written right after the label, the same sentence goes on to state a DIFFERENT value
    with '=' or an is/are/has cue ("Answer: 40 minus 1, which is 39"). Such a reply is ambiguous."""
    sentences = _sentences(prepared)
    first = sentences[0] if sentences else prepared
    rest = first[hit[0] + len(hit[1]):]
    later: List[str] = []
    if "=" in rest:
        rhs = _values(rest.rsplit("=", 1)[1])
        if rhs:
            later.append(rhs[0][1])
    for m in _CUE.finditer(rest):
        found = _value_after_cue(rest, m.end())
        if found:
            later.append(found[1])
    return any(not _same_value(hit[1], v) for v in later)


def _label_segment(text: str) -> Optional[str]:
    last_end = None
    for m in _LABEL.finditer(text):
        last_end = m.end()
    for m in _HEADING.finditer(text):
        if last_end is None or m.end() > last_end:
            last_end = m.end()
    if last_end is None:
        return None
    rest = text[last_end:].lstrip()
    return re.split(r"\n\s*\n", rest, maxsplit=1)[0][:400]


def _sentences(paragraph: str) -> List[str]:
    return [p for p in re.split(r"(?<=[.!?])\s+", paragraph.strip()) if p.strip()]


def extract_math_answer(response: Optional[str]) -> Optional[str]:
    if not response or not response.strip():
        return None
    text = response.strip()

    boxed = last_boxed(text)
    if boxed:
        return clean_candidate(boxed)

    hashes = re.findall(r"####\s*([^\n]+)", text)
    if hashes and clean_candidate(hashes[-1]):
        return clean_candidate(hashes[-1])

    segment = _label_segment(text)
    if segment:
        prepared = _prep(segment)
        hit = _value_at(prepared, 0)
        if hit:
            return None if _contradicted_later(prepared, hit) or _hedged(prepared, hit) else hit[1]
        sentences = _sentences(segment)
        if sentences:
            found = _from_sentence(sentences[0], last_value_fallback=False)
            if found:
                return found

    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    for paragraph in reversed(paragraphs[-3:]):
        for sentence in reversed(_sentences(paragraph)[-3:]):
            if _CHATTER.search(sentence):
                continue
            found = _from_sentence(sentence, last_value_fallback=True)
            if found:
                return found
    return None


def clean_candidate(value: str) -> str:
    """Drop markdown emphasis, code ticks, math delimiters and spaces around an answer (inner characters are kept)."""
    return value.strip().strip("*_`$ \t").strip()


def normalize_math_str(value: str) -> str:
    s = _prep(value).lower()
    s = s.replace("\\times", "*").replace("\\cdot", "*").replace("\u00d7", "*")
    s = s.replace("^\\circ", "").replace("\u00b0", "").replace("\\%", "").replace("%", "")
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", s)
    return s.strip(".").strip("*_`")


def _number(s: str) -> Optional[float]:
    m = re.fullmatch(r"-?\d+(?:\.\d+)?", s)
    if m:
        return float(s)
    m = re.fullmatch(r"(-?\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)", s)
    if m and float(m.group(2)) != 0:
        return float(m.group(1)) / float(m.group(2))
    m = re.fullmatch(r"(-?)(\d+(?:\.\d+)?)?\*?sqrt\((\d+)\)", s)
    if m:
        coef = float(m.group(2)) if m.group(2) else 1.0
        return (-1.0 if m.group(1) else 1.0) * coef * float(m.group(3)) ** 0.5
    return None


def answers_equal(extracted: str, truth: str) -> bool:
    a, b = normalize_math_str(extracted), normalize_math_str(truth)
    if a == b:
        return True
    na, nb = _number(a), _number(b)
    return na is not None and nb is not None and abs(na - nb) <= 1e-5 * max(1.0, abs(nb))


def score_math(response: str, ground_truth: str) -> Tuple[bool, Optional[str]]:
    extracted = extract_math_answer(response)
    if not extracted or not ground_truth:
        return False, extracted
    return answers_equal(extracted, ground_truth), extracted
