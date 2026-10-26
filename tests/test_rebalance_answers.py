import contextlib
import copy
import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
from collections import Counter

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)

from src.benchmarks.loader import BenchmarkLoader
from src.tools.rebalance_answers import apply_plan, main, plan_file, read_dataset, render
from src.tools.validate_dataset import validate_file

LETTERS = "ABCD"


def make_items(n=20, answer="A", **over):
    items = []
    for i in range(1, n + 1):
        options = [f"right {i}", f"wrong one {i}", f"wrong two {i}", f"wrong three {i}"]
        items.append({"question_id": f"q_{i:02d}", "question": f"Synthetic question {i} alpha{i} beta{i} gamma{i}?",
                      "options": options, "answer": answer, **over})
    return items


def correct_text(item):
    return item["options"][LETTERS.index(item["answer"])]


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


class TestPlanning(unittest.TestCase):
    def test_all_a_becomes_balanced_to_within_one(self):
        for n in (4, 15, 20, 41, 100):
            with self.subTest(n=n):
                items = make_items(n)
                counts = Counter(i["answer"] for i in apply_plan(plan_file(items, seed=1)))
                self.assertLessEqual(max(counts.values()) - min(counts.get(c, 0) for c in LETTERS), 1)

    def test_correct_option_text_options_set_and_other_fields_are_preserved(self):
        items = make_items(30)
        for it in items:
            it["topic"] = "t"
        before = copy.deepcopy(items)
        after = apply_plan(plan_file(items, seed=3))
        self.assertEqual(items, before, "the input must not be modified")
        for old, new in zip(before, after):
            self.assertEqual(correct_text(new), correct_text(old))
            self.assertEqual(sorted(new["options"]), sorted(old["options"]))
            self.assertEqual({k: v for k, v in new.items() if k not in ("options", "answer")},
                             {k: v for k, v in old.items() if k not in ("options", "answer")})
            self.assertEqual(list(new.keys()), list(old.keys()))

    def test_deterministic_for_a_seed_and_different_across_seeds(self):
        items = make_items(30)
        a = apply_plan(plan_file(items, seed=7))
        self.assertEqual(a, apply_plan(plan_file(items, seed=7)))
        self.assertNotEqual(a, apply_plan(plan_file(items, seed=8)))

    def test_questions_already_in_the_target_position_are_left_exactly_as_they_are(self):
        items = make_items(20)
        fp = plan_file(items, seed=1)
        after = apply_plan(fp)
        same = [p for p in fp.plans if not p.changed and not p.skipped]
        self.assertTrue(same)
        for p in same:
            self.assertIs(after[p.index], items[p.index])

    def test_items_that_refer_to_other_options_are_skipped_but_counted(self):
        items = make_items(20)
        for it in items[:6]:
            it["options"][3] = "All of the above"
        items[6]["question"] = "Is option B the best choice here?"
        items[7]["options"][2] = "Both A and B"
        fp = plan_file(items, seed=1)
        skipped = {p.index for p in fp.plans if p.skipped}
        self.assertEqual(skipped, set(range(8)))
        after = apply_plan(fp)
        for idx in skipped:
            self.assertIs(after[idx], items[idx])
        # 8 fixed answers stay A, so the 12 movable ones must go to the other letters and not add to A
        self.assertEqual(Counter(i["answer"] for i in after[8:]).get("A", 0), 0)

    def test_invalid_answers_are_skipped_not_crashed_on(self):
        items = make_items(8)
        items[0]["answer"] = "Z"
        del items[1]["options"]
        fp = plan_file(items, seed=1)
        self.assertEqual({p.index for p in fp.plans if p.skipped}, {0, 1})

    def test_ten_option_questions(self):
        items = [{"question_id": f"m_{i}", "question": f"Synthetic ten-option question {i}?",
                  "options": [f"opt{i}_{k}" for k in range(10)], "answer": "A"} for i in range(40)]
        after = apply_plan(plan_file(items, seed=2))
        counts = Counter(i["answer"] for i in after)
        self.assertEqual(set(counts), set("ABCDEFGHIJ"))
        self.assertEqual(counts.most_common(1)[0][1], 4)
        for old, new in zip(items, after):
            self.assertEqual(new["options"]["ABCDEFGHIJ".index(new["answer"])], old["options"][0])


class TestWithTheShippedFiles(unittest.TestCase):
    """Reads gpqa and truthfulqa to plan only; nothing is ever written to them."""

    def test_shipped_files_are_balanced_and_planning_leaves_them_untouched(self):
        expected = {"A": 4, "B": 3, "C": 4, "D": 4}
        for name in ("gpqa", "truthfulqa"):
            path = os.path.join(REPO, "datasets", name, "sample.jsonl")
            before = sha(path)
            items = read_dataset(path)[0]
            self.assertEqual(Counter(i["answer"] for i in items), expected)
            fp = plan_file(items, seed=42, path=path)
            after = apply_plan(fp)
            counts = Counter(i["answer"] for i in after)
            self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)
            for old, new in zip(items, after):
                self.assertEqual(correct_text(new), correct_text(old))
            self.assertEqual(sha(path), before)


class TestCommandLine(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = os.path.join(self.tmp.name, "datasets")
        os.makedirs(os.path.join(self.root, "gpqa"))
        self.path = os.path.join(self.root, "gpqa", "sample.jsonl")
        lines = [json.dumps(it) for it in make_items(20)]
        with open(self.path, "wb") as f:  # CRLF with a final newline, like the shipped files
            f.write(("\r\n".join(lines) + "\r\n").encode("utf-8"))

    def run_cli(self, *args):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main(list(args))
        return code, out.getvalue()

    def test_dry_run_writes_nothing_and_says_so(self):
        before = sha(self.path)
        code, out = self.run_cli(self.path)
        self.assertEqual(code, 0)
        self.assertIn("DRY RUN: nothing was written", out)
        self.assertIn("answer positions before: A=20", out)
        self.assertEqual(sha(self.path), before)
        self.assertEqual(os.listdir(self.tmp.name), ["datasets"])

    def test_output_dir_writes_a_copy_and_never_touches_the_original(self):
        before = sha(self.path)
        out_dir = os.path.join(self.tmp.name, "rebalanced")
        code, out = self.run_cli(self.path, "--output-dir", out_dir, "--quiet")
        self.assertEqual(code, 0)
        self.assertEqual(sha(self.path), before)
        copy_path = os.path.join(out_dir, "gpqa", "sample.jsonl")
        self.assertTrue(os.path.isfile(copy_path))
        raw = open(copy_path, "rb").read()
        self.assertTrue(raw.endswith(b"\r\n"))
        self.assertEqual(raw.count(b"\r\n"), raw.count(b"\n"), "line endings must stay CRLF")
        items = BenchmarkLoader.load_jsonl(copy_path)
        self.assertEqual(len(items), 20)
        self.assertEqual(Counter(i["answer"] for i in items), {c: 5 for c in LETTERS})

    def test_unchanged_questions_are_byte_identical_in_the_copy(self):
        out_dir = os.path.join(self.tmp.name, "rebalanced")
        self.run_cli(self.path, "--output-dir", out_dir, "--quiet")
        old_lines = open(self.path, "rb").read().split(b"\r\n")
        new_lines = open(os.path.join(out_dir, "gpqa", "sample.jsonl"), "rb").read().split(b"\r\n")
        identical = sum(a == b for a, b in zip(old_lines, new_lines))
        self.assertGreaterEqual(identical, 6)  # the ~5 already-A questions, plus the empty tail element

    def test_the_rebalanced_copy_passes_the_validator_balance_check(self):
        out_dir = os.path.join(self.tmp.name, "rebalanced")
        self.run_cli(self.path, "--output-dir", out_dir, "--quiet")
        rep = validate_file(os.path.join(out_dir, "gpqa", "sample.jsonl"), bench_type="gpqa", check_collisions=False)
        self.assertTrue(rep.ok, rep.errors)
        self.assertEqual(rep.answer_counts, {c: 5 for c in LETTERS})
        original = validate_file(self.path, bench_type="gpqa", check_collisions=False)
        self.assertFalse(original.ok)  # 20 x "A" is rejected

    def test_refuses_to_write_over_its_input(self):
        before = sha(self.path)
        code, out = self.run_cli(self.path, "--output-dir", self.root)
        self.assertEqual(code, 2)
        self.assertIn("refusing to write over the input", out)
        self.assertEqual(sha(self.path), before)

    def test_bad_input_is_reported_not_crashed_on(self):
        bad = os.path.join(self.tmp.name, "bad.jsonl")
        with open(bad, "wb") as f:
            f.write(b"{not json}\n")
        self.assertEqual(self.run_cli(bad)[0], 2)
        self.assertEqual(self.run_cli(os.path.join(self.tmp.name, "missing.jsonl"))[0], 2)

    def test_render_preserves_unicode_style_of_the_input(self):
        path = os.path.join(self.tmp.name, "u.jsonl")
        item = make_items(1)[0]
        item["question"] = "café question?"
        with open(path, "wb") as f:
            f.write((json.dumps(item, ensure_ascii=False) + "\n").encode("utf-8"))
        items, raw_lines, crlf, final_nl, ascii_ = read_dataset(path)
        self.assertFalse(ascii_)
        self.assertIn("café", render(plan_file(items), raw_lines, crlf, final_nl, ascii_))


if __name__ == "__main__":
    unittest.main()
