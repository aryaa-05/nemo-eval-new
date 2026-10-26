import json
import os
import re
import sys
import unittest
from unittest import mock

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)

from src.benchmarks.base import get_benchmark
from src.engine.answer_scoring import score_generation
from src.scoring import math_answers
from src.scoring.math_answers import answers_equal, extract_math_answer, last_boxed, normalize_math_str, score_math

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def _load(name, key):
    with open(os.path.join(FIXTURES, name), encoding="utf-8") as f:
        return json.load(f)[key]


CASES = _load("math_cases.json", "cases")
LIVE = _load("live_math_replies.json", "replies")
GUARDS = [c for c in CASES if c["group"] == "must-stay-wrong"]


def score(case):
    """Every fixture goes through the same score_generation() the runner uses."""
    bench = get_benchmark(case["benchmark"], base_dir=REPO)
    return score_generation(bench, {"answer": case["truth"]}, {"text": case["text"], "finish_reason": "stop"})


def guards_that_became_right():
    return sorted(c["id"] for c in GUARDS if score(c).is_correct)


class TestCuratedMathFixtures(unittest.TestCase):
    def test_fixture_file_shape(self):
        ids = [c["id"] for c in CASES]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertGreaterEqual(len(GUARDS), 15, "the brief asks for at least 15 must-stay-wrong fixtures")
        self.assertGreaterEqual(len([c for c in CASES if c["group"] == "right-format"]), 15)
        self.assertEqual({c["group"] for c in CASES}, {"right-format", "must-stay-wrong"})

    def test_all_curated_cases(self):
        for case in CASES:
            with self.subTest(case=case["id"], name=case["name"]):
                scored = score(case)
                expect = case["expect"]
                self.assertEqual(scored.is_correct, expect["correct"])
                self.assertEqual(scored.extracted, expect["extracted"])
                self.assertEqual(scored.error, expect["error"])
                self.assertEqual(scored.no_answer, expect["error"] == "no_answer_found")
                self.assertFalse(scored.truncated)
                self.assertFalse(scored.api_error)

    def test_must_stay_wrong_fixtures_are_all_expected_wrong(self):
        for case in GUARDS:
            with self.subTest(case=case["id"]):
                self.assertFalse(case["expect"]["correct"])
                extracted = case["expect"]["extracted"]
                if extracted is not None:  # the fixture must not accidentally describe a correct extraction
                    self.assertFalse(answers_equal(extracted, case["truth"]))

    def test_right_format_fixtures_are_all_expected_right(self):
        for case in CASES:
            if case["group"] == "right-format":
                with self.subTest(case=case["id"]):
                    self.assertTrue(case["expect"]["correct"])
                    self.assertTrue(answers_equal(case["expect"]["extracted"], case["truth"]))

    def test_there_is_a_guard_for_each_kind_of_false_positive_in_the_brief(self):
        names = " | ".join(c["name"] for c in GUARDS)
        for needle in ("correct intermediate", "only in the working", "unit that changes the value"):
            self.assertIn(needle, names)


class TestLiveReplyFixtures(unittest.TestCase):
    """Real replies from the pair 2 and pair 3 stand-in runs (pipeline validation, not findings about real models)."""

    def test_file_shape(self):
        self.assertEqual(len(LIVE), 118)
        self.assertEqual({r["benchmark"] for r in LIVE}, {"gsm8k", "math_500"})
        self.assertEqual(len({r["run_id"] for r in LIVE}), 4)

    def test_every_live_reply_scores_as_expected(self):
        for row in LIVE:
            with self.subTest(run=row["run_id"], item=row["question_id"], model=row["model"]):
                scored = score(row)
                self.assertEqual(scored.is_correct, row["expect"]["correct"])
                self.assertEqual(scored.extracted, row["expect"]["extracted"])

    def test_no_reply_that_the_old_scorer_got_right_is_now_wrong(self):
        lost = [(r["run_id"], r["question_id"]) for r in LIVE if r["original_scorer"]["correct"] and not r["expect"]["correct"]]
        self.assertEqual(lost, [])

    def test_only_listed_verdicts_changed(self):
        changed = [r for r in LIVE if r["original_scorer"]["correct"] != r["expect"]["correct"]]
        self.assertTrue(changed)
        self.assertTrue(all(r["expect"]["correct"] for r in changed))  # every change is wrong -> right


class TestHelpers(unittest.TestCase):
    def test_last_boxed_takes_the_last_one_and_handles_nested_braces(self):
        self.assertEqual(last_boxed(r"so \boxed{3} and finally \boxed{\frac{1}{6}}"), r"\frac{1}{6}")
        self.assertIsNone(last_boxed("no box here"))

    def test_normalisation(self):
        self.assertEqual(normalize_math_str(r"\$1{,}680"), "1680")
        self.assertEqual(normalize_math_str("$1,680"), "1680")
        self.assertEqual(normalize_math_str("25%"), "25")
        self.assertEqual(normalize_math_str(r"\frac{1}{6}"), "1/6")
        self.assertEqual(normalize_math_str(r"9\sqrt{3}"), "9*sqrt(3)")
        self.assertEqual(normalize_math_str("9√3"), "9*sqrt(3)")

    def test_equal_values(self):
        for extracted, truth in [("10.00", "10"), (r"\frac{1}{6}", "1/6"), ("0.5", "1/2"), ("1,680", "1680"), ("45", "45.0"),
                                 (r"9\sqrt{3}", "9*sqrt(3)"), ("25%", "25")]:
            with self.subTest(extracted=extracted, truth=truth):
                self.assertTrue(answers_equal(extracted, truth))

    def test_different_values_are_never_equal(self):
        for extracted, truth in [("12", "120"), ("1,200", "120"), ("5", "-5"), ("3", "3.5"), ("0.15", "150"), ("25", "0.25"),
                                 (r"\frac{1}{3}", "1/6"), ("9*sqrt(2)", "9*sqrt(3)"), ("18*sqrt(3)", "9*sqrt(3)"), ("16.80", "1680")]:
            with self.subTest(extracted=extracted, truth=truth):
                self.assertFalse(answers_equal(extracted, truth))

    def test_units_are_never_converted(self):
        self.assertFalse(answers_equal("0.5", "30"))    # 0.5 hours is not 30 minutes
        self.assertFalse(answers_equal("0.15", "150"))  # km is not metres

    def test_empty_inputs(self):
        self.assertIsNone(extract_math_answer(""))
        self.assertIsNone(extract_math_answer(None))
        self.assertIsNone(extract_math_answer("   \n  "))
        self.assertEqual(score_math("Answer: 5", ""), (False, "5"))
        self.assertEqual(score_math("", "5"), (False, None))

    def test_no_blind_last_number_fallback(self):
        # a number deep in the working is not taken when the closing paragraphs state none
        working = ("First add the apples to the pears.\n\nThen subtract the bruised ones, which is 5 + 5 = 10.\n\n"
                   "Now consider the price.\n\nThen think about the tax.\n\nI hope this helps.")
        self.assertIsNone(extract_math_answer(working))
        self.assertIsNone(extract_math_answer("Let me know if you need 10 more examples."))  # closing chatter is skipped

    def test_known_limitation_an_incidental_number_in_the_final_sentence_of_an_unlabelled_reply_is_read(self):
        # Documented in CONTEXT.md. Tightening this loses real right answers (4 of the 118 stored replies), so it is kept:
        # the extractor then returns the number and the comparison with the key decides. It is only a wrong "right"
        # if the key happens to equal that number.
        self.assertEqual(extract_math_answer("The problem has 3 parts and I cannot solve all of them."), "3")


class TestGuardsHaveTeeth(unittest.TestCase):
    """The must-stay-wrong fixtures must fail if the scorer is made lenient. Each mutation below is a plausible
    'quick fix' for under-scoring, and each must turn several guards right."""

    def test_the_real_scorer_turns_no_guard_right(self):
        self.assertEqual(guards_that_became_right(), [])

    def test_substring_equality_is_caught(self):
        def lenient(extracted, truth):
            a, b = normalize_math_str(extracted), normalize_math_str(truth)
            return a in b or b in a

        with mock.patch.object(math_answers, "answers_equal", lenient):
            flipped = guards_that_became_right()
        self.assertGreaterEqual(len(flipped), 5, flipped)

    def test_ignoring_signs_and_decimal_points_is_caught(self):
        def lenient(extracted, truth):
            return re.sub(r"\D", "", extracted) == re.sub(r"\D", "", truth)

        with mock.patch.object(math_answers, "answers_equal", lenient):
            flipped = guards_that_became_right()
        self.assertGreaterEqual(len(flipped), 3, flipped)  # the sign guards W09/W10 and the decimal-point guard W24

    def test_accepting_the_key_anywhere_in_the_reply_is_caught(self):
        # "the key appears somewhere in the text, so call it right"
        def lenient_score(response, truth):
            found = re.search(rf"(?<![\d.,]){re.escape(truth)}(?![\d])", response.replace(",", ""))
            return bool(found), (truth if found else None)

        with mock.patch.object(math_answers, "score_math", lenient_score):
            flipped = guards_that_became_right()
        self.assertGreaterEqual(len(flipped), 5, flipped)

    def test_a_blind_last_number_fallback_is_caught(self):
        original = math_answers.extract_math_answer

        def with_fallback(response):
            found = original(response)
            if found is None and response:
                numbers = re.findall(r"-?\d[\d,]*\.?\d*", response)
                return numbers[-1] if numbers else None
            return found

        with mock.patch.object(math_answers, "extract_math_answer", with_fallback):
            flipped = guards_that_became_right()
            # the no-answer guards (W20-W22) must also notice: they expect no extraction at all
            wrongly_extracted = [c["id"] for c in GUARDS if c["expect"]["extracted"] is None and score(c).extracted is not None]
        self.assertTrue(flipped or wrongly_extracted, "the mutation went unnoticed")
        self.assertTrue(wrongly_extracted, "the no-answer guards should notice a blind fallback")


if __name__ == "__main__":
    unittest.main()
