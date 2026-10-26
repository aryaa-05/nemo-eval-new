import contextlib
import io
import json
import os
import sys
import tempfile
import unittest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)

from src.benchmarks.base import get_benchmark
from src.benchmarks.loader import BenchmarkLoader
from src.engine.solvers import get_solver
from src.tools.validate_dataset import format_report, main, validate_file, validate_source

GOOD_SOURCE = "Example Directions on Widget Records, 2001, Paragraph 12(a)"
GOOD_EXCERPT = "A synthetic supporting sentence that only the tests use."
WORDS = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]
LETTERS = "ABCD"


def make_item(i, letter="A", **over):
    item = {
        "question_id": f"kyc_{i:03d}",
        "topic": "synthetic",
        "question": f"Synthetic check {i}: " + " ".join(f"{w}{i}" for w in WORDS) + "?",
        "options": [f"option {c} number {i}" for c in LETTERS],
        "answer": letter,
        "source": GOOD_SOURCE,
        "source_excerpt": GOOD_EXCERPT,
    }
    item.update(over)
    return item


def balanced_items(n=100, **over):
    return [make_item(i, LETTERS[i % 4], **over) for i in range(1, n + 1)]


class TempDatasets:
    """A throwaway datasets/<name>/sample.jsonl tree."""

    def __init__(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = os.path.join(self._tmp.name, "datasets")

    def write(self, name, items=None, raw=None, filename="sample.jsonl"):
        folder = os.path.join(self.root, name)
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, filename)
        with open(path, "wb") as f:
            if raw is not None:
                f.write(raw)
            else:
                f.write(("\n".join(json.dumps(it) for it in items) + "\n").encode("utf-8"))
        return path

    def cleanup(self):
        self._tmp.cleanup()


class ValidatorTestCase(unittest.TestCase):
    def setUp(self):
        self.ds = TempDatasets()
        self.addCleanup(self.ds.cleanup)

    def validate(self, items, name="kyc_india", **kwargs):
        return validate_file(self.ds.write(name, items), **kwargs)

    def assertHasError(self, rep, fragment):
        self.assertTrue(any(fragment in e for e in rep.errors), f"no error containing {fragment!r} in {rep.errors}")

    def assertHasWarning(self, rep, fragment):
        self.assertTrue(any(fragment in w for w in rep.warnings), f"no warning containing {fragment!r} in {rep.warnings}")


class TestValidateSource(unittest.TestCase):
    def test_generic_or_incomplete_sources_are_rejected(self):
        bad = [
            "RBI", "KYC", "rbi kyc", "RBI Master Direction", "Reserve Bank of India", "Master Direction on KYC",
            "Section 5", "RBI, Section 5", "KYC para 3", "RBI Master Direction on KYC, Section 5",
            "TODO section 3", "tbd", "", "   ", None, 12, "https://example.org/page",
            "Example Directions on Widget Records",           # document but no locator
            "Example Directions on Widget Records, Paragraph",  # locator word without a number
        ]
        for value in bad:
            with self.subTest(value=value):
                self.assertIsNotNone(validate_source(value))

    def test_specific_document_plus_locator_is_accepted(self):
        good = [
            GOOD_SOURCE,
            "Section 12 of Example Directions on Widget Records, 2001",
            "Example Act, 1999, Chapter VI, Para 38(a)",
            "RBI/2020-21/99 Example Circular on Widgets, Annex II",
            "Example Textbook of Imaging, 4th ed., Section 3.4",
        ]
        for value in good:
            with self.subTest(value=value):
                self.assertIsNone(validate_source(value))

    def test_error_messages_say_what_to_fix(self):
        self.assertIn("too generic", validate_source("RBI"))
        self.assertIn("no section", validate_source("Example Directions on Widget Records, 2001"))
        self.assertIn("specific document", validate_source("RBI, Section 5"))


class TestSchemaAndFields(ValidatorTestCase):
    def test_a_clean_file_passes(self):
        rep = self.validate(balanced_items(40))
        self.assertTrue(rep.ok, rep.errors)
        self.assertEqual(rep.n_items, 40)
        self.assertEqual(rep.answer_counts, {"A": 10, "B": 10, "C": 10, "D": 10})

    def test_missing_required_fields(self):
        for field in ("question_id", "question", "options", "answer"):
            with self.subTest(field=field):
                item = make_item(1)
                del item[field]
                rep = self.validate([item])
                self.assertFalse(rep.ok)

    def test_wrong_option_count(self):
        rep = self.validate([make_item(1, options=["a1", "b2", "c3", "d4", "e5"])])
        self.assertHasError(rep, "exactly 4 strings")
        rep = self.validate([make_item(1, options=["a1", "b2", "c3"])])
        self.assertHasError(rep, "exactly 4 strings")

    def test_answer_must_be_one_uppercase_valid_letter(self):
        for bad in ("a", "E", "AB", "", "A.", 1, None):
            with self.subTest(answer=bad):
                self.assertHasError(self.validate([make_item(1, answer=bad)]), "`answer` must be exactly one uppercase letter")

    def test_options_must_not_carry_their_own_letter_label(self):
        rep = self.validate([make_item(1, options=["A. first", "B. second", "C. third", "D. fourth"])])
        self.assertHasError(rep, "own letter label")

    def test_duplicate_options_within_an_item(self):
        rep = self.validate([make_item(1, options=["Same text", "same  TEXT", "other one", "last one"])])
        self.assertHasError(rep, "duplicates another option")

    def test_options_that_differ_only_by_punctuation_are_not_duplicates(self):
        rep = self.validate([make_item(1, options=["-3/5", "3/5", "1/5", "-1/5"])])
        self.assertFalse(any("duplicates another option" in e for e in rep.errors))

    def test_whitespace_in_an_option_is_a_warning(self):
        rep = self.validate([make_item(1, options=[" padded", "b two", "c three", "d four"])])
        self.assertHasWarning(rep, "whitespace")

    def test_all_of_the_above_is_a_warning(self):
        rep = self.validate([make_item(1, options=["one", "two", "three", "All of the above"])])
        self.assertHasWarning(rep, "refers to other options")

    def test_unknown_field_is_a_warning_not_an_error(self):
        rep = self.validate([make_item(1, sourse="typo")])
        self.assertHasWarning(rep, "unknown field `sourse`")
        self.assertFalse(any("sourse" in e for e in rep.errors))

    def test_review_flags_must_be_real_booleans(self):
        self.assertHasError(self.validate([make_item(1, review="true")]), "`review` must be true or false")
        self.assertHasError(self.validate([make_item(1, human_checked="yes")]), "`human_checked` must be true or false")

    def test_unknown_benchmark_type(self):
        rep = validate_file(self.ds.write("not_a_benchmark", balanced_items(4)))
        self.assertHasError(rep, "unknown benchmark type")

    def test_type_can_be_given_explicitly(self):
        path = self.ds.write("my_folder", balanced_items(8))
        self.assertTrue(validate_file(path, bench_type="kyc_india").ok)


class TestSourceAndExcerptRules(ValidatorTestCase):
    def test_source_and_excerpt_are_required_for_the_three_new_datasets(self):
        for name in ("kyc_india", "rbi_regulations", "image_processing"):
            with self.subTest(name=name):
                item = make_item(1)
                del item["source"], item["source_excerpt"]
                rep = self.validate([item], name=name)
                self.assertHasError(rep, "`source` is missing")
                self.assertHasError(rep, "`source_excerpt` is missing")

    def test_not_required_for_other_datasets_unless_asked(self):
        item = make_item(1)
        del item["source"], item["source_excerpt"]
        self.assertTrue(self.validate([item], name="gpqa").ok)
        self.assertHasError(self.validate([item], name="gpqa", require_source=True), "`source` is missing")

    def test_a_source_that_is_present_is_always_checked(self):
        self.assertHasError(self.validate([make_item(1, source="RBI")], name="gpqa"), "too generic")

    def test_generic_source_is_an_error_in_the_dataset(self):
        for bad in ("RBI", "KYC", "RBI Master Direction"):
            with self.subTest(source=bad):
                self.assertHasError(self.validate([make_item(1, source=bad)]), "`source`")

    def test_excerpt_length_limits(self):
        self.assertHasError(self.validate([make_item(1, source_excerpt="too short")]), "shorter than")
        self.assertHasError(self.validate([make_item(1, source_excerpt="x" * 601)]), "longer than")
        self.assertTrue(self.validate([make_item(1, source_excerpt="x" * 600)]).ok)

    def test_excerpt_check_can_be_switched_off(self):
        item = make_item(1)
        del item["source_excerpt"]
        self.assertTrue(self.validate([item], require_excerpt=False).ok)


class TestExcerptAgainstSourceText(ValidatorTestCase):
    SOURCE = ("Chapter 1. Example Directions on Widget Records, 2001.\n"
              "12. (a) Every widget shall be recorded in the regis-\ntry within thirty days of manufacture.\n"
              "(b) The registry shall be “available for inspection” at all times.")

    def source_file(self, text=None):
        path = os.path.join(self.ds._tmp.name, "source.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.SOURCE if text is None else text)
        return path

    def test_excerpt_present_in_source_text_passes_despite_line_breaks_and_quotes(self):
        items = [make_item(1, source_excerpt="Every widget shall be recorded in the registry within thirty days of manufacture."),
                 make_item(2, source_excerpt='The registry shall be "available for inspection" at all times.')]
        rep = self.validate(items, source_texts=[self.source_file()])
        self.assertTrue(rep.ok, rep.errors)
        self.assertEqual((rep.excerpts_found, rep.excerpts_checked), (2, 2))
        self.assertIn("source_excerpt found in --source-text: 2 of 2 checked", format_report(rep))

    def test_invented_excerpt_is_rejected(self):
        items = [make_item(1, source_excerpt="Every widget shall be recorded within sixty days of manufacture.")]
        rep = self.validate(items, source_texts=[self.source_file()])
        self.assertHasError(rep, "not found in the --source-text")
        self.assertEqual(rep.excerpts_found, 0)

    def test_unreadable_source_text_is_an_error(self):
        rep = self.validate([make_item(1)], source_texts=[os.path.join(self.ds._tmp.name, "missing.txt")])
        self.assertHasError(rep, "cannot read --source-text")

    def test_without_source_text_the_report_says_excerpts_were_not_compared(self):
        self.assertIn("not compared with any source text", format_report(self.validate([make_item(1)])))

    def test_command_line_flag(self):
        path = self.ds.write("kyc_india", [make_item(1, source_excerpt="Every widget shall be recorded in the registry.")])
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main([path, "--source-text", self.source_file()])
        self.assertEqual(code, 0, out.getvalue())


class TestDuplicatesAndIds(ValidatorTestCase):
    def test_duplicate_question_id(self):
        items = [make_item(1), make_item(2, question_id="kyc_001")]
        self.assertHasError(self.validate(items), "duplicate question_id")

    def test_duplicate_question_text_ignoring_case_and_punctuation(self):
        items = [make_item(1, question="What is the Widget limit?"), make_item(2, question="what is the widget   limit")]
        self.assertHasError(self.validate(items), "duplicate question text")

    def test_near_duplicate_is_a_warning(self):
        base = [f"w{n}" for n in range(14)]
        items = [make_item(1, question=" ".join(base)), make_item(2, question=" ".join(base[:-1] + ["different"]))]
        rep = self.validate(items)
        self.assertHasWarning(rep, "almost identical")
        self.assertFalse(any("duplicate question text" in e for e in rep.errors))

    def test_id_characters(self):
        self.assertHasError(self.validate([make_item(1, question_id="has space")]), "may only contain")

    def test_id_collision_with_another_dataset_is_an_error(self):
        self.ds.write("gpqa", [make_item(1, question_id="shared_01")])
        rep = validate_file(self.ds.write("kyc_india", [make_item(2, question_id="shared_01")]))
        self.assertHasError(rep, "also exists in datasets/gpqa")

    def test_collision_check_can_be_skipped(self):
        self.ds.write("gpqa", [make_item(1, question_id="shared_01")])
        rep = validate_file(self.ds.write("kyc_india", [make_item(2, question_id="shared_01")]), check_collisions=False)
        self.assertTrue(rep.ok, rep.errors)


class TestBalance(ValidatorTestCase):
    def _with_counts(self, counts):
        letters = [c for c, n in counts.items() for _ in range(n)]
        return [make_item(i + 1, letter) for i, letter in enumerate(letters)]

    def test_all_the_same_letter_fails_for_100_items(self):
        rep = self.validate(self._with_counts({"A": 100}))
        self.assertHasError(rep, "answer positions are unbalanced")

    def test_moderate_skew_within_tolerance_passes(self):
        self.assertTrue(self.validate(self._with_counts({"A": 30, "B": 25, "C": 25, "D": 20})).ok)

    def test_skew_beyond_tolerance_fails(self):
        self.assertHasError(self.validate(self._with_counts({"A": 40, "B": 20, "C": 20, "D": 20})), "unbalanced")

    def test_tolerance_is_adjustable(self):
        rep = self.validate(self._with_counts({"A": 40, "B": 20, "C": 20, "D": 20}), balance_tolerance=0.20)
        self.assertTrue(rep.ok, rep.errors)

    def test_small_files_only_warn_about_skew(self):
        rep = self.validate(self._with_counts({"A": 8, "B": 1, "C": 1}))
        self.assertTrue(rep.ok)
        self.assertHasWarning(rep, "would score 80%")

    def test_correct_option_always_longest_is_a_warning(self):
        items = []
        for i in range(1, 41):
            letter = LETTERS[i % 4]
            options = [f"short {c}{i}" for c in LETTERS]
            options[LETTERS.index(letter)] = f"a much much longer correct option text {i}"
            items.append(make_item(i, letter, options=options))
        self.assertHasWarning(self.validate(items), "longest one")


class TestReviewAndHumanChecks(ValidatorTestCase):
    def test_review_true_is_counted_and_printed(self):
        items = balanced_items(40)
        for it in items[:7]:
            it["review"] = True
        items[8]["review"] = False
        rep = self.validate(items)
        self.assertEqual(rep.review_true, 7)
        self.assertIn("review: true: 7 of 40 (18%)", format_report(rep))

    def test_min_checked_is_enforced(self):
        items = balanced_items(100)
        for it in items[:19]:
            it["human_checked"] = True
        self.assertHasError(self.validate(items, min_checked=0.20), "have `human_checked: true`")
        items[19]["human_checked"] = True
        self.assertTrue(self.validate(items, min_checked=0.20).ok)

    def test_expect_count(self):
        self.assertHasError(self.validate(balanced_items(40), expect_count=100), "expected 100 items, found 40")
        self.assertTrue(self.validate(balanced_items(40), expect_count=40).ok)


class TestFileLevelProblems(ValidatorTestCase):
    def test_bom_is_rejected_because_the_loader_cannot_read_it(self):
        raw = b"\xef\xbb\xbf" + "\n".join(json.dumps(i) for i in balanced_items(4)).encode()
        rep = validate_file(self.ds.write("kyc_india", raw=raw))
        self.assertHasError(rep, "byte-order mark")
        with self.assertRaises(json.JSONDecodeError):  # this is the real loader failure the check protects against
            BenchmarkLoader.load_jsonl(self.ds.write("kyc_india", raw=raw))

    def test_invalid_json_reports_the_line_number(self):
        good = json.dumps(make_item(1))
        raw = f"{good}\n{{not json}}\n{json.dumps(make_item(2))}\n".encode()
        rep = validate_file(self.ds.write("kyc_india", raw=raw))
        self.assertTrue(any(e.startswith("line 2:") and "invalid JSON" in e for e in rep.errors), rep.errors)

    def test_non_object_lines_and_blank_lines(self):
        raw = f"{json.dumps(make_item(1))}\n\n[1, 2]\n".encode()
        rep = validate_file(self.ds.write("kyc_india", raw=raw))
        self.assertHasError(rep, "must be a JSON object")
        self.assertEqual(rep.n_items, 1)

    def test_windows_line_endings_are_fine(self):
        raw = ("\r\n".join(json.dumps(i) for i in balanced_items(8)) + "\r\n").encode()
        self.assertTrue(validate_file(self.ds.write("kyc_india", raw=raw)).ok)

    def test_missing_file(self):
        self.assertHasError(validate_file(os.path.join(self.ds.root, "kyc_india", "nope.jsonl"), bench_type="kyc_india"), "file not found")


class TestOtherBenchmarkKinds(ValidatorTestCase):
    def test_mmlu_pro_needs_ten_options_and_letters_up_to_j(self):
        item = {"question_id": "mmlu_x1", "domain": "x", "question": "Synthetic ten-option question text here?",
                "options": [f"opt{n}" for n in range(10)], "answer": "J"}
        self.assertTrue(self.validate([item], name="mmlu_pro").ok)
        self.assertHasError(self.validate([{**item, "options": item["options"][:4]}], name="mmlu_pro"), "exactly 10 strings")

    def test_math_items(self):
        item = {"question_id": "gsm_x1", "question": "Synthetic arithmetic question?", "answer": "42"}
        self.assertTrue(self.validate([item], name="gsm8k").ok)
        self.assertHasError(self.validate([{**item, "answer": ""}], name="gsm8k"), "`answer` must be a non-empty string")

    def test_code_items(self):
        item = {
            "task_id": "Synthetic/0",
            "prompt": 'def double(x):\n    """Return x twice."""\n',
            "canonical_solution": "    return x * 2\n",
            "test": "def check(candidate):\n    assert candidate(2) == 4\n",
            "entry_point": "double",
        }
        self.assertTrue(self.validate([item], name="humaneval").ok)
        self.assertHasError(self.validate([{**item, "entry_point": "triple"}], name="humaneval"), "entry_point")
        self.assertHasError(self.validate([{**item, "test": "assert True"}], name="humaneval"), "def check")
        self.assertHasError(self.validate([{**item, "canonical_solution": "    return (\n"}], name="humaneval"), "not valid Python")


class TestShippedDatasets(unittest.TestCase):
    def test_non_grounded_datasets_have_no_errors(self):
        for name in ("mmlu_pro", "gsm8k", "math_500", "gpqa", "humaneval", "truthfulqa", "custom_byob"):
            with self.subTest(dataset=name):
                rep = validate_file(os.path.join(REPO, "datasets", name, "sample.jsonl"))
                self.assertTrue(rep.ok, rep.errors)

    def test_regulatory_and_image_samples_only_fail_the_new_source_rules(self):
        # The 15-item placeholders predate the `source` rule. Whatever else they get right, the only
        # problems allowed here are the missing source fields; this also keeps passing once they are replaced.
        for name in ("kyc_india", "rbi_regulations", "image_processing"):
            with self.subTest(dataset=name):
                rep = validate_file(os.path.join(REPO, "datasets", name, "sample.jsonl"))
                for err in rep.errors:
                    self.assertTrue("`source`" in err or "`source_excerpt`" in err, err)


class TestExtraFieldsAreAcceptedAndNeverShownToModels(ValidatorTestCase):
    def test_loader_returns_extra_fields_untouched(self):
        path = self.ds.write("kyc_india", [make_item(1, review=True, human_checked=False)])
        item = BenchmarkLoader.load_jsonl(path)[0]
        for key in ("source", "source_excerpt", "review", "human_checked", "topic"):
            self.assertIn(key, item)

    def test_prompts_are_built_without_source_fields(self):
        item = make_item(1, source="Example Directions on Widget Records, 2001, Paragraph 99", source_excerpt="UNIQUE-EXCERPT-MARKER sentence.")
        for solver in ("simple", "cot", "harbor"):
            with self.subTest(solver=solver):
                prompt = get_solver(solver).format_prompt(item, "kyc_india")
                self.assertNotIn("UNIQUE-EXCERPT-MARKER", prompt)
                self.assertNotIn("Paragraph 99", prompt)

    def test_scoring_ignores_extra_fields(self):
        bench = get_benchmark("kyc_india", base_dir=REPO)
        item = make_item(1, "C", review=True)
        correct, _, extracted, _ = bench.evaluate(item, "Answer: C")
        self.assertTrue(correct)
        self.assertEqual(extracted, "C")


class TestCommandLine(ValidatorTestCase):
    def run_cli(self, *args):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main(list(args))
        return code, out.getvalue()

    def test_exit_codes_and_output(self):
        good = self.ds.write("kyc_india", balanced_items(40))
        code, text = self.run_cli(good)
        self.assertEqual(code, 0)
        self.assertIn("RESULT: PASS", text)
        self.assertIn("review: true: 0 of 40 (0%)", text)
        bad = self.ds.write("rbi_regulations", [make_item(1, source="RBI")])
        code, text = self.run_cli(bad)
        self.assertEqual(code, 1)
        self.assertIn("RESULT: FAIL", text)

    def test_several_files_fail_if_any_fails(self):
        good = self.ds.write("kyc_india", balanced_items(40))
        bad = self.ds.write("rbi_regulations", [make_item(1, source="RBI")])
        self.assertEqual(self.run_cli(good, bad)[0], 1)

    def test_final_flag_demands_100_items_and_20_percent_checked(self):
        items = balanced_items(100)
        path = self.ds.write("kyc_india", items)
        code, text = self.run_cli(path, "--final")
        self.assertEqual(code, 1)
        self.assertIn("human_checked: true", text)
        for it in items[:20]:
            it["human_checked"] = True
        path = self.ds.write("kyc_india", items)
        self.assertEqual(self.run_cli(path, "--final")[0], 0)
        short = self.ds.write("rbi_regulations", [dict(it, human_checked=True) for it in balanced_items(40)])
        self.assertEqual(self.run_cli(short, "--final")[0], 1)

    def test_flags_toggle_source_requirements(self):
        item = make_item(1)
        del item["source"], item["source_excerpt"]
        path = self.ds.write("kyc_india", [item])
        self.assertEqual(self.run_cli(path)[0], 1)
        self.assertEqual(self.run_cli(path, "--no-require-source", "--no-require-excerpt")[0], 0)


if __name__ == "__main__":
    unittest.main()
