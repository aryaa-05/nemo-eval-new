"""Rebalance answer positions in multiple-choice JSONL datasets by shuffling each question's options.

DRY RUN by default: it prints what would change and writes nothing. It never edits a file in place. To get a
rebalanced copy, pass --output-dir; the copy is written to <output-dir>/<dataset folder>/<file name> and the
original is left untouched. Review the copy (and run the validator on it) before replacing anything yourself.

    python src/tools/rebalance_answers.py datasets/gpqa/sample.jsonl datasets/truthfulqa/sample.jsonl
    python src/tools/rebalance_answers.py datasets/gpqa/sample.jsonl --output-dir rebalanced

Only the order of `options` and the matching `answer` letter change. The text of the correct option, the set of
options and every other field stay exactly as they were. Questions whose options refer to each other ("all of the
above", "both A and B") or whose text mentions "option B" are left alone and reported.

Rebalancing changes the dataset, so scores from before and after are not comparable.
"""
import argparse
import json
import os
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

LETTERS = "ABCDEFGHIJ"

_OPTION_REFERS_TO_OTHERS = re.compile(
    r"(?i)\b(?:all|none|neither|both|any) of the (?:above|options)\b"
    r"|\b(?:both|either|neither)\s+\(?[a-j]\)?\s+(?:and|or|nor)\s+\(?[a-j]\)?(?![\w'])"
    r"|\boptions?\s+\(?[a-j]\)?(?![\w'])"
)
_QUESTION_MENTIONS_OPTION = re.compile(r"(?i)\b(?:option|choice)s?\s+\(?[A-J]\)?(?![\w'])")


@dataclass
class ItemPlan:
    index: int
    item_id: Any
    old_letter: Optional[str]
    new_letter: Optional[str]
    skipped: Optional[str] = None          # reason the item is left alone
    order: Optional[List[int]] = None      # new option i is old option order[i]

    @property
    def changed(self) -> bool:
        return self.skipped is None and self.order is not None and self.order != sorted(self.order)


@dataclass
class FilePlan:
    path: str
    items: List[Dict[str, Any]]
    plans: List[ItemPlan] = field(default_factory=list)

    def counts(self, after: bool) -> Dict[str, int]:
        n = max((len(i.get("options", [])) for i in self.items if isinstance(i.get("options"), list)), default=4)
        c = Counter(p.new_letter if after else p.old_letter for p in self.plans)
        return {ch: c.get(ch, 0) for ch in LETTERS[:n]}


def _skip_reason(item: Dict[str, Any]) -> Optional[str]:
    options, answer = item.get("options"), item.get("answer")
    if not isinstance(options, list) or len(options) < 2 or not all(isinstance(o, str) for o in options):
        return "no usable options list"
    if answer not in list(LETTERS[: len(options)]):
        return f"answer {answer!r} is not a valid letter for {len(options)} options"
    if any(_OPTION_REFERS_TO_OTHERS.search(o) for o in options):
        return "an option refers to other options (e.g. 'all of the above')"
    if _QUESTION_MENTIONS_OPTION.search(str(item.get("question", ""))):
        return "the question text mentions a lettered option"
    return None


def plan_file(items: List[Dict[str, Any]], seed: int = 42, path: str = "") -> FilePlan:
    """Decide a new option order and answer letter for every movable item. Does not modify `items`."""
    rng = random.Random(seed)
    fp = FilePlan(path=path, items=items)
    movable: List[int] = []
    for idx, item in enumerate(items):
        reason = _skip_reason(item)
        old = item.get("answer") if isinstance(item.get("answer"), str) else None
        fp.plans.append(ItemPlan(idx, item.get("question_id") or item.get("task_id"), old, old, skipped=reason))
        if reason is None:
            movable.append(idx)

    n_options = max((len(items[i]["options"]) for i in movable), default=0)
    counts = Counter({ch: 0 for ch in LETTERS[:n_options]})
    for p in fp.plans:
        if p.skipped and p.old_letter in counts:
            counts[p.old_letter] += 1  # skipped items keep their letter and still count toward the balance

    rng.shuffle(movable)
    for idx in movable:
        item, plan = items[idx], fp.plans[idx]
        n = len(item["options"])
        usable = {ch: counts[ch] for ch in LETTERS[:n]}
        lowest = min(usable.values())
        target = rng.choice([ch for ch, c in usable.items() if c == lowest])
        counts[target] += 1
        plan.new_letter = target
        old_pos = LETTERS.index(plan.old_letter)
        new_pos = LETTERS.index(target)
        if old_pos == new_pos:
            plan.order = list(range(n))  # already in the right place: leave this question exactly as it is
            continue
        others = [i for i in range(n) if i != old_pos]
        rng.shuffle(others)
        order: List[Optional[int]] = [None] * n
        order[new_pos] = old_pos
        it = iter(others)
        for pos in range(n):
            if order[pos] is None:
                order[pos] = next(it)
        plan.order = order  # type: ignore[assignment]
    return fp


def apply_plan(fp: FilePlan) -> List[Dict[str, Any]]:
    """The rebalanced items (new dicts). Raises AssertionError if anything but order and answer would change."""
    out = []
    for item, plan in zip(fp.items, fp.plans):
        if plan.skipped or not plan.changed:
            out.append(item)
            continue
        new = dict(item)
        new["options"] = [item["options"][i] for i in plan.order]
        new["answer"] = plan.new_letter
        correct_before = item["options"][LETTERS.index(item["answer"])]
        correct_after = new["options"][LETTERS.index(new["answer"])]
        assert correct_before == correct_after, f"{plan.item_id}: correct option text changed"
        assert sorted(new["options"]) == sorted(item["options"]), f"{plan.item_id}: option set changed"
        assert {k: v for k, v in new.items() if k not in ("options", "answer")} == \
               {k: v for k, v in item.items() if k not in ("options", "answer")}, f"{plan.item_id}: another field changed"
        out.append(new)
    return out


def read_dataset(path: str) -> Tuple[List[Dict[str, Any]], List[str], bool, bool, bool]:
    """(items, raw_lines, crlf, final_newline, ensure_ascii) for a JSONL file."""
    with open(path, "rb") as f:
        raw = f.read()
    if raw.startswith(b"\xef\xbb\xbf"):
        raise ValueError("file starts with a byte-order mark; fix that first (see the dataset validator)")
    text = raw.decode("utf-8")
    crlf = "\r\n" in text
    lines = [ln for ln in text.replace("\r\n", "\n").split("\n") if ln.strip()]
    items = []
    for n, ln in enumerate(lines, 1):
        try:
            items.append(json.loads(ln))
        except json.JSONDecodeError as e:
            raise ValueError(f"line {n}: invalid JSON ({e.msg})") from e
    return items, lines, crlf, text.endswith("\n"), (chr(92) + "u") in text


def render(fp: FilePlan, raw_lines: List[str], crlf: bool, final_newline: bool, ensure_ascii: bool) -> str:
    new_items = apply_plan(fp)
    out = []
    for raw, old, new in zip(raw_lines, fp.items, new_items):
        out.append(raw if new is old else json.dumps(new, ensure_ascii=ensure_ascii))
    nl = "\r\n" if crlf else "\n"
    return nl.join(out) + (nl if final_newline else "")


def describe(fp: FilePlan, verbose: bool = True) -> str:
    lines = [f"== {fp.path}  ({len(fp.items)} questions)"]
    fmt = lambda c: "  ".join(f"{k}={v}" for k, v in c.items())
    lines.append(f"answer positions before: {fmt(fp.counts(False))}")
    lines.append(f"answer positions after : {fmt(fp.counts(True))}")
    changed = [p for p in fp.plans if p.changed]
    skipped = [p for p in fp.plans if p.skipped]
    same = len(fp.plans) - len(changed) - len(skipped)
    lines.append(f"questions that would change: {len(changed)}   unchanged: {same}   left alone: {len(skipped)}")
    if verbose:
        for p in changed:
            item = fp.items[p.index]
            before = item["options"]
            after = [before[i] for i in p.order]
            lines.append(f"  {p.item_id}: answer {p.old_letter} -> {p.new_letter}   (correct option: {before[LETTERS.index(p.old_letter)]!r})")
            lines.append(f"      options before: {' | '.join(before)}")
            lines.append(f"      options after : {' | '.join(after)}")
        for p in skipped:
            lines.append(f"  {p.item_id}: LEFT ALONE - {p.skipped}")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Rebalance answer positions by shuffling options. Dry run unless --output-dir is given.")
    ap.add_argument("paths", nargs="+", help="multiple-choice dataset files, e.g. datasets/gpqa/sample.jsonl")
    ap.add_argument("--seed", type=int, default=42, help="shuffle seed; the same seed always gives the same result (default 42)")
    ap.add_argument("--output-dir", help="write rebalanced copies to <dir>/<dataset folder>/<file name>; originals are never modified")
    ap.add_argument("--quiet", action="store_true", help="print only the before/after counts, not every question")
    args = ap.parse_args(argv)

    jobs = []
    for path in args.paths:
        try:
            items, raw_lines, crlf, final_nl, ascii_ = read_dataset(path)
        except (OSError, ValueError) as e:
            print(f"ERROR {path}: {e}")
            return 2
        target = None
        if args.output_dir:
            target = os.path.join(args.output_dir, os.path.basename(os.path.dirname(os.path.abspath(path))), os.path.basename(path))
            if os.path.abspath(target) == os.path.abspath(path):
                print(f"ERROR {path}: refusing to write over the input file; choose a different --output-dir")
                return 2
        jobs.append((path, items, raw_lines, crlf, final_nl, ascii_, target))

    for path, items, raw_lines, crlf, final_nl, ascii_, target in jobs:
        fp = plan_file(items, args.seed, path)
        print(describe(fp, verbose=not args.quiet))
        if target:
            text = render(fp, raw_lines, crlf, final_nl, ascii_)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with open(target, "w", encoding="utf-8", newline="") as f:
                f.write(text)
            print(f"wrote rebalanced copy: {target}   (the original was not modified)\n")
        else:
            print("DRY RUN: nothing was written.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
