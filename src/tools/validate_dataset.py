"""Check a benchmark dataset file (JSONL) before it is used in a run.

Usage (from the repo root):
    python src/tools/validate_dataset.py datasets/kyc_india/sample.jsonl
    python src/tools/validate_dataset.py datasets/kyc_india/sample.jsonl --final

It checks format and consistency only. It cannot tell whether a question is factually right or whether a
`source` really says what the question claims; a person has to compare a sample against the source text.
"""
import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.benchmarks.loader import BenchmarkLoader

FINAL_COUNT = 100
FINAL_MIN_CHECKED = 0.20
MAX_EXCERPT_CHARS = 600
MIN_EXCERPT_CHARS = 20


@dataclass(frozen=True)
class Profile:
    kind: str  # "mcq" | "math" | "code"
    n_options: int = 0
    id_field: str = "question_id"
    grounded: bool = False  # `source` and `source_excerpt` are required by default


_MCQ4 = Profile("mcq", 4)
_GROUNDED4 = Profile("mcq", 4, grounded=True)
PROFILES: Dict[str, Profile] = {
    "mmlu_pro": Profile("mcq", 10),
    "gpqa": _MCQ4,
    "truthfulqa": _MCQ4,
    "custom_byob": _MCQ4,
    "kyc_india": _GROUNDED4,
    "rbi_regulations": _GROUNDED4,
    "image_processing": _GROUNDED4,
    "gsm8k": Profile("math"),
    "math_500": Profile("math"),
    "humaneval": Profile("code", id_field="task_id"),
}

REQUIRED = {
    "mcq": ["question", "options", "answer"],
    "math": ["question", "answer"],
    "code": ["prompt", "canonical_solution", "test", "entry_point"],
}
OPTIONAL_KNOWN = {"topic", "domain", "subject", "source", "source_excerpt", "review", "human_checked", "notes"}

# ---- source checking ---------------------------------------------------------------------------------------------
_GENERIC_SOURCES = {
    "rbi", "kyc", "aml", "pmla", "rbi kyc", "rbi aml", "kyc norms", "rbi norms", "rbi guidelines", "guidelines",
    "rbi circular", "circular", "rbi master direction", "master direction", "reserve bank of india",
    "rbi notification", "notification", "na", "n a", "none", "tbd", "todo", "unknown", "textbook", "internet",
    "web", "google", "wikipedia", "source", "see source", "as above",
}
_DOC_STOPWORDS = {
    "rbi", "reserve", "bank", "of", "india", "the", "a", "an", "and", "on", "for", "to", "in", "as", "amended",
    "updated", "dated", "no", "master", "direction", "directions", "circular", "circulars", "notification",
    "guidelines", "kyc", "aml", "cft", "norms", "rules", "rule", "act", "regulations",
}
_LOCATOR_RE = re.compile(
    r"(?:(?i:\b(?:sections?|sec|paras?|paragraphs?|clauses?|chapters?|annex(?:ure)?s?|appendix|articles?|rule|"
    r"regulation|part|schedule|table|figure|pages?|pp)\b\.?)|§)"
    r"\s*[:\-]?\s*\(?(?:[0-9]+|[IVXLC]+\b)[A-Za-z0-9().\-/]*"
)
_PLACEHOLDER_RE = re.compile(r"(?i)\b(todo|tbd|xxx|fixme|placeholder|lorem)\b|\?\?")


def validate_source(value: Any) -> Optional[str]:
    """Return an error message, or None if the source names a specific document AND a section/paragraph."""
    if not isinstance(value, str) or not value.strip():
        return "`source` is missing or empty"
    text = value.strip()
    norm = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    if norm in _GENERIC_SOURCES:
        return f"`source` {text!r} is too generic: name the document and the section or paragraph"
    if _PLACEHOLDER_RE.search(text):
        return f"`source` {text!r} looks like a placeholder"
    locators = list(_LOCATOR_RE.finditer(text))
    if not locators:
        return f"`source` {text!r} has no section, paragraph, clause, chapter, annex or page number"
    doc_text = _LOCATOR_RE.sub(" ", text)
    tokens = [t.lower() for t in re.findall(r"[A-Za-z0-9][A-Za-z0-9/.\-]*", doc_text)]
    distinctive = [t for t in tokens if t.strip(".-/") not in _DOC_STOPWORDS]
    if len(tokens) < 2 or not distinctive:
        return (f"`source` {text!r} does not identify a specific document: give the full document title and "
                f"year or circular number as well as the section")
    return None


# ---- report ------------------------------------------------------------------------------------------------------
@dataclass
class Report:
    path: str
    bench_type: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    n_items: int = 0
    answer_counts: Dict[str, int] = field(default_factory=dict)
    review_true: int = 0
    human_checked: int = 0
    source_text_used: bool = False
    excerpts_checked: int = 0
    excerpts_found: int = 0

    @property
    def ok(self) -> bool:
        return not self.errors

    def error(self, msg: str, line: Optional[int] = None, item_id: Any = None) -> None:
        self.errors.append(_where(line, item_id) + msg)

    def warn(self, msg: str, line: Optional[int] = None, item_id: Any = None) -> None:
        self.warnings.append(_where(line, item_id) + msg)


def _where(line: Optional[int], item_id: Any) -> str:
    if line is None:
        return ""
    return f"line {line} ({item_id}): " if item_id else f"line {line}: "


def _norm_text(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def _squash(s: str) -> str:
    """Letters and digits only, case-folded: lets an excerpt match source text despite line breaks, hyphenation,
    curly quotes and other punctuation differences introduced by copying out of a PDF."""
    return re.sub(r"[\W_]+", "", s.casefold())


def _load_source_keys(paths: List[str], rep: Report) -> List[str]:
    keys = []
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8", errors="replace") as f:
                keys.append(_squash(f.read()))
        except OSError as e:
            rep.error(f"cannot read --source-text file {p!r}: {e.strerror or e}")
    return keys


def _read_items(path: str, rep: Report) -> List[Tuple[int, Dict[str, Any]]]:
    with open(path, "rb") as f:
        raw = f.read()
    if raw.startswith(b"\xef\xbb\xbf"):
        rep.error("file starts with a UTF-8 byte-order mark (BOM); the loader cannot read it. Re-save as UTF-8 without BOM.")
        raw = raw[3:]
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        rep.error(f"file is not valid UTF-8 ({e})")
        return []
    items: List[Tuple[int, Dict[str, Any]]] = []
    for n, line in enumerate(text.split("\n"), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            rep.error(f"invalid JSON ({e.msg} at column {e.colno})", n)
            continue
        if not isinstance(obj, dict):
            rep.error("each line must be a JSON object", n)
            continue
        items.append((n, obj))
    return items


def _sibling_ids(path: str) -> Dict[str, str]:
    """question_id / task_id of every other datasets/<name>/sample.jsonl, so cross-benchmark collisions are caught."""
    me = os.path.abspath(path)
    root = os.path.dirname(os.path.dirname(me))
    if os.path.basename(root) != "datasets":
        return {}
    found: Dict[str, str] = {}
    for name in sorted(os.listdir(root)):
        other = os.path.join(root, name, "sample.jsonl")
        if os.path.abspath(other) == me or not os.path.isfile(other):
            continue
        try:
            with open(other, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    key = obj.get("question_id") or obj.get("task_id")
                    if key:
                        found[str(key)] = name
        except (OSError, ValueError):
            continue
    return found


def _check_mcq(rep: Report, line: int, item: Dict[str, Any], prof: Profile, qid: Any, letters: str) -> None:
    question = item.get("question")
    if not isinstance(question, str) or not question.strip():
        rep.error("`question` must be a non-empty string", line, qid)
    elif len(question.strip()) < 20:
        rep.warn("question is very short", line, qid)
    elif re.search(r"(?m)^\s*\(?[A-J][\.\)]\s", question):
        rep.warn("question text seems to contain its own lettered options; options belong in `options`", line, qid)

    options = item.get("options")
    if not isinstance(options, list) or len(options) != prof.n_options:
        got = len(options) if isinstance(options, list) else type(options).__name__
        rep.error(f"`options` must be a list of exactly {prof.n_options} strings (found {got}); the scorer only accepts {letters[0]}-{letters[-1]}", line, qid)
        options = None
    else:
        seen = set()
        for i, opt in enumerate(options):
            tag = f"option {letters[i]}"
            if not isinstance(opt, str) or not opt.strip():
                rep.error(f"{tag} must be a non-empty string", line, qid)
                continue
            if opt != opt.strip():
                rep.warn(f"{tag} has leading or trailing whitespace", line, qid)
            if re.match(r"^\s*\(?[A-Ja-j][\.\)]\s", opt):
                rep.error(f"{tag} starts with its own letter label; the prompt adds the letters itself", line, qid)
            key = " ".join(opt.lower().split())  # punctuation matters in options: "-3/5" and "3/5" are different answers
            if key in seen:
                rep.error(f"{tag} duplicates another option", line, qid)
            seen.add(key)
            if re.search(r"(?i)\b(all|none) of the above\b|\bboth [a-j] and [a-j]\b", opt):
                rep.warn(f"{tag} refers to other options ('all/none of the above', 'both A and B')", line, qid)

    answer = item.get("answer")
    if not isinstance(answer, str) or answer not in list(letters):
        rep.error(f"`answer` must be exactly one uppercase letter from {letters[0]}-{letters[-1]} (found {answer!r})", line, qid)


def _check_math(rep: Report, line: int, item: Dict[str, Any], qid: Any) -> None:
    if not isinstance(item.get("question"), str) or not item["question"].strip():
        rep.error("`question` must be a non-empty string", line, qid)
    answer = item.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        rep.error("`answer` must be a non-empty string", line, qid)


def _check_code(rep: Report, line: int, item: Dict[str, Any], qid: Any) -> None:
    for name in ("prompt", "canonical_solution", "test", "entry_point"):
        if not isinstance(item.get(name), str) or not item[name].strip():
            rep.error(f"`{name}` must be a non-empty string", line, qid)
            return
    if "def check" not in item["test"]:
        rep.error("`test` must define `def check(candidate)`", line, qid)
    if not re.search(rf"def\s+{re.escape(item['entry_point'])}\s*\(", item["prompt"]):
        rep.error("`entry_point` is not defined by a `def` in `prompt`", line, qid)
    try:
        compile(item["prompt"] + item["canonical_solution"], "<canonical>", "exec")
    except SyntaxError as e:
        rep.error(f"prompt + canonical_solution is not valid Python ({e.msg})", line, qid)


def validate_file(
    path: str,
    bench_type: Optional[str] = None,
    *,
    require_source: Optional[bool] = None,
    require_excerpt: Optional[bool] = None,
    expect_count: Optional[int] = None,
    min_checked: Optional[float] = None,
    balance_tolerance: float = 0.10,
    check_collisions: bool = True,
    source_texts: Optional[List[str]] = None,
) -> Report:
    rep = Report(path=path)
    bench_type = bench_type or os.path.basename(os.path.dirname(os.path.abspath(path)))
    rep.bench_type = bench_type
    prof = PROFILES.get(bench_type)
    if prof is None:
        rep.error(f"unknown benchmark type {bench_type!r}; known types: {', '.join(sorted(PROFILES))} (use --type)")
        return rep
    if not os.path.isfile(path):
        rep.error("file not found")
        return rep
    need_source = prof.grounded if require_source is None else require_source
    need_excerpt = need_source if require_excerpt is None else require_excerpt

    items = _read_items(path, rep)
    rep.n_items = len(items)
    letters = "ABCDEFGHIJ"[: prof.n_options] if prof.kind == "mcq" else ""
    known = set(REQUIRED[prof.kind]) | OPTIONAL_KNOWN | {prof.id_field}

    ids: Dict[str, int] = {}
    questions: Dict[str, int] = {}
    token_sets: List[Tuple[int, Any, set]] = []
    answers: List[str] = []
    longest_hits = 0
    sibling = _sibling_ids(path) if check_collisions else {}
    source_keys: List[str] = []
    if source_texts:
        rep.source_text_used = True
        source_keys = _load_source_keys(source_texts, rep)

    for line, item in items:
        qid = item.get(prof.id_field)
        if not isinstance(qid, str) or not qid.strip():
            rep.error(f"`{prof.id_field}` must be a non-empty string", line)
            qid = None
        else:
            if qid != qid.strip() or not re.fullmatch(r"[A-Za-z0-9_./\-]+", qid):
                rep.error(f"`{prof.id_field}` {qid!r} may only contain letters, digits and _ . / -", line, qid)
            if qid in ids:
                rep.error(f"duplicate {prof.id_field} (first used on line {ids[qid]})", line, qid)
            ids.setdefault(qid, line)
            if qid in sibling:
                rep.error(f"{prof.id_field} also exists in datasets/{sibling[qid]}; ids must be unique across benchmarks "
                          f"because run comparison matches questions by id", line, qid)

        for name in REQUIRED[prof.kind]:
            if name not in item:
                rep.error(f"missing required field `{name}`", line, qid)
        if all(n in item for n in REQUIRED[prof.kind]):
            if prof.kind == "mcq":
                _check_mcq(rep, line, item, prof, qid, letters)
            elif prof.kind == "math":
                _check_math(rep, line, item, qid)
            else:
                _check_code(rep, line, item, qid)

        for name in item:
            if name not in known:
                rep.warn(f"unknown field `{name}` (typo? it is stored but ignored by the scorer)", line, qid)

        # source / excerpt
        if need_source or "source" in item:
            problem = validate_source(item.get("source"))
            if problem:
                rep.error(problem, line, qid)
        if need_excerpt or "source_excerpt" in item:
            ex = item.get("source_excerpt")
            if not isinstance(ex, str) or not ex.strip():
                rep.error("`source_excerpt` is missing or empty: paste the sentence from the source that supports the answer", line, qid)
            elif len(ex.strip()) < MIN_EXCERPT_CHARS:
                rep.error(f"`source_excerpt` is shorter than {MIN_EXCERPT_CHARS} characters", line, qid)
            elif len(ex) > MAX_EXCERPT_CHARS:
                rep.error(f"`source_excerpt` is longer than {MAX_EXCERPT_CHARS} characters; keep only the supporting sentence(s)", line, qid)
            elif source_keys:
                rep.excerpts_checked += 1
                key = _squash(ex)
                if len(key) >= 15 and any(key in sk for sk in source_keys):
                    rep.excerpts_found += 1
                else:
                    rep.error("`source_excerpt` was not found in the --source-text file(s); it must be copied from the source, not written from memory", line, qid)

        # review / human_checked flags must be real booleans
        for flag in ("review", "human_checked"):
            if flag in item and not isinstance(item[flag], bool):
                rep.error(f"`{flag}` must be true or false (a JSON boolean, not {item[flag]!r})", line, qid)
        if item.get("review") is True:
            rep.review_true += 1
        if item.get("human_checked") is True:
            rep.human_checked += 1

        # duplicates / near-duplicates
        text = item.get("question") or item.get("prompt")
        if isinstance(text, str) and text.strip():
            key = _norm_text(text)
            if key in questions:
                rep.error(f"duplicate question text (same as line {questions[key]})", line, qid)
            questions.setdefault(key, line)
            toks = set(key.split())
            if len(toks) >= 4:
                token_sets.append((line, qid, toks))

        if prof.kind == "mcq" and isinstance(item.get("answer"), str) and item["answer"] in letters:
            answers.append(item["answer"])
            opts = item.get("options")
            if isinstance(opts, list) and len(opts) == prof.n_options and all(isinstance(o, str) for o in opts):
                lengths = [len(o) for o in opts]
                idx = letters.index(item["answer"])
                if lengths[idx] == max(lengths) and lengths.count(max(lengths)) == 1:
                    longest_hits += 1

    # near-duplicates (warning only; capped so huge files stay fast)
    if len(token_sets) <= 2000:
        for i in range(len(token_sets)):
            la, qa, ta = token_sets[i]
            for j in range(i + 1, len(token_sets)):
                lb, qb, tb = token_sets[j]
                if len(ta & tb) / len(ta | tb) >= 0.85:
                    rep.warn(f"question is almost identical to the one on line {la} ({qa})", lb, qb)

    # answer-position balance
    if prof.kind == "mcq" and answers:
        counts = {ch: answers.count(ch) for ch in letters}
        rep.answer_counts = counts
        n = len(answers)
        share = {ch: counts[ch] / n for ch in letters}
        if n >= 20:
            uniform = 1.0 / len(letters)
            for ch in letters:
                if abs(share[ch] - uniform) > balance_tolerance + 1e-9:
                    rep.error(f"answer positions are unbalanced: {ch} is the answer for {counts[ch]} of {n} questions "
                              f"({share[ch]:.0%}); each letter should be near {uniform:.0%} (tolerance +/-{balance_tolerance:.0%})")
            if longest_hits / n > 0.45:
                rep.warn(f"the correct option is the longest one in {longest_hits} of {n} questions; models can exploit this")
        elif n >= 2 and max(share.values()) > 0.5:
            top = max(counts, key=counts.get)
            rep.warn(f"answer {top} is correct for {counts[top]} of {n} questions; a model that always answers {top} would score {share[top]:.0%}")

    # count and human-check requirements
    if expect_count is not None and rep.n_items != expect_count:
        rep.error(f"expected {expect_count} items, found {rep.n_items}")
    if min_checked is not None:
        have = rep.human_checked / rep.n_items if rep.n_items else 0.0
        if have + 1e-9 < min_checked:
            rep.error(f"only {rep.human_checked} of {rep.n_items} items ({have:.0%}) have `human_checked: true`; "
                      f"at least {min_checked:.0%} must be checked by a person against the source")

    # the real loader must accept the file, extra fields included
    if rep.n_items and not any("JSON" in e or "BOM" in e or "UTF-8" in e for e in rep.errors):
        try:
            loaded = BenchmarkLoader.load_jsonl(path)
            if len(loaded) != rep.n_items:
                rep.error(f"loader returned {len(loaded)} items but the file has {rep.n_items}")
        except Exception as e:  # noqa: BLE001 - report whatever the loader rejects
            rep.error(f"the dataset loader cannot read this file: {e}")
    return rep


def format_report(rep: Report) -> str:
    out = [f"== {rep.path}  (type: {rep.bench_type})"]
    out.append(f"items: {rep.n_items}")
    if rep.answer_counts:
        out.append("answer positions: " + "  ".join(f"{k}={v}" for k, v in rep.answer_counts.items()))
    pct = (lambda c: f"{c / rep.n_items:.0%}") if rep.n_items else (lambda c: "n/a")
    out.append(f"review: true: {rep.review_true} of {rep.n_items} ({pct(rep.review_true)})")
    out.append(f"human_checked: true: {rep.human_checked} of {rep.n_items} ({pct(rep.human_checked)})")
    if rep.source_text_used:
        out.append(f"source_excerpt found in --source-text: {rep.excerpts_found} of {rep.excerpts_checked} checked")
    else:
        out.append("source_excerpt not compared with any source text (pass --source-text FILE to do that)")
    for w in rep.warnings:
        out.append(f"WARNING  {w}")
    for e in rep.errors:
        out.append(f"ERROR    {e}")
    out.append(f"RESULT: {'PASS' if rep.ok else 'FAIL'} ({len(rep.errors)} error(s), {len(rep.warnings)} warning(s))")
    return "\n".join(out)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Validate benchmark dataset JSONL files.")
    ap.add_argument("paths", nargs="+", help="dataset file(s), e.g. datasets/kyc_india/sample.jsonl")
    ap.add_argument("--type", dest="bench_type", help=f"benchmark type; default: the folder name. One of: {', '.join(sorted(PROFILES))}")
    ap.add_argument("--require-source", action=argparse.BooleanOptionalAction, default=None,
                    help="require a specific `source` on every item (default: on for kyc_india, rbi_regulations, image_processing)")
    ap.add_argument("--require-excerpt", action=argparse.BooleanOptionalAction, default=None,
                    help="require `source_excerpt` on every item (default: same as --require-source)")
    ap.add_argument("--expect-count", type=int, default=None, help="fail unless the file has exactly this many items")
    ap.add_argument("--min-checked", type=float, default=None, help="fail unless this fraction of items has human_checked: true")
    ap.add_argument("--balance-tolerance", type=float, default=0.10, help="allowed deviation of each answer letter from an equal share (default 0.10)")
    ap.add_argument("--no-collision-check", action="store_true", help="skip the check against ids in the other datasets")
    ap.add_argument("--source-text", action="append", default=None, metavar="FILE",
                    help="plain-text copy of the source document (repeatable). Every source_excerpt must appear in one of them")
    ap.add_argument("--final", action="store_true",
                    help=f"release check: --expect-count {FINAL_COUNT} --min-checked {FINAL_MIN_CHECKED}")
    args = ap.parse_args(argv)

    expect_count = args.expect_count if args.expect_count is not None else (FINAL_COUNT if args.final else None)
    min_checked = args.min_checked if args.min_checked is not None else (FINAL_MIN_CHECKED if args.final else None)
    worst = 0
    for path in args.paths:
        rep = validate_file(
            path, args.bench_type,
            require_source=args.require_source, require_excerpt=args.require_excerpt,
            expect_count=expect_count, min_checked=min_checked,
            balance_tolerance=args.balance_tolerance, check_collisions=not args.no_collision_check,
            source_texts=args.source_text,
        )
        print(format_report(rep))
        print()
        if not rep.ok:
            worst = 1
    return worst


if __name__ == "__main__":
    sys.exit(main())
