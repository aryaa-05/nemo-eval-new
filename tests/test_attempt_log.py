import json
import os
import sys
import unittest
from unittest import mock

import requests

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)

from src.engine.runner import EvaluationRunner
from src.gateway.interceptor import InterceptorProxy
from src.gateway.model_client import UnifiedModelClient
from src.storage.schema import ItemResult, RunArtifact

KEY = "sentinel-nvidia-key-attempt-log"
LIVE = {"id": "vendor/live-model", "name": "Live Model", "provider": "nvidia_nim"}


def reply(status=200, content="Answer: A", headers=None):
    r = mock.MagicMock()
    r.status_code = status
    r.headers = headers or {}
    if status >= 400:
        r.raise_for_status.side_effect = requests.exceptions.HTTPError(f"{status} Client Error: Synthetic")
    else:
        r.raise_for_status.return_value = None
    r.json.return_value = {"choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                           "usage": {"prompt_tokens": 5, "completion_tokens": 3}}
    return r


class LiveTestCase(unittest.TestCase):
    def setUp(self):
        for p in (mock.patch.dict(os.environ, {"NVIDIA_API_KEY": KEY}), mock.patch.object(UnifiedModelClient, "_sleep")):
            p.start()
            self.addCleanup(p.stop)

    def generate(self, responses):
        client = UnifiedModelClient(LIVE, InterceptorProxy())  # a new client each time: identical prompts are cached
        with mock.patch("src.gateway.model_client.requests.post", side_effect=responses) as post:
            return client.generate("a prompt"), post


class TestClientAttemptLog(LiveTestCase):
    def test_a_first_try_success_logs_one_200(self):
        out, _ = self.generate([reply(200)])
        self.assertEqual((out["attempts"], out["attempt_log"]), (1, ["200"]))

    def test_a_429_then_success_logs_both_requests_in_order(self):
        out, post = self.generate([reply(429), reply(200)])
        self.assertEqual(out["attempt_log"], ["429", "200"])
        self.assertEqual((out["attempts"], post.call_count), (2, 2))
        self.assertNotIn("error", out)

    def test_a_mix_of_429_and_5xx_before_success(self):
        out, _ = self.generate([reply(429), reply(503), reply(502), reply(200)])
        self.assertEqual(out["attempt_log"], ["429", "503", "502", "200"])
        self.assertEqual(out["attempts"], 4)

    def test_exhausted_retries_log_every_attempt_and_the_final_status(self):
        out, post = self.generate([reply(503)] * 5)
        self.assertEqual(out["attempt_log"], ["503"] * 5)
        self.assertEqual((out["attempts"], post.call_count, out["http_status"]), (5, 5, 503))
        self.assertTrue(out["text"].startswith("API Error"))

    def test_a_read_timeout_then_success(self):
        out, _ = self.generate([requests.exceptions.ReadTimeout("slow"), reply(200)])
        self.assertEqual(out["attempt_log"], ["read_timeout", "200"])
        self.assertEqual(out["attempts"], 2)

    def test_two_read_timeouts_log_both_and_give_up(self):
        out, post = self.generate([requests.exceptions.ReadTimeout("slow")] * 2)
        self.assertEqual(out["attempt_log"], ["read_timeout", "read_timeout"])
        self.assertEqual(post.call_count, 2)
        self.assertIsNone(out["http_status"])

    def test_a_non_retryable_status_is_logged_once(self):
        for status in (400, 401, 403, 404, 410, 422):
            with self.subTest(status=status):
                out, post = self.generate([reply(status)])
                self.assertEqual(out["attempt_log"], [str(status)])
                self.assertEqual((out["attempts"], post.call_count, out["http_status"]), (1, 1, status))

    def test_a_429_that_turns_into_a_401_logs_both(self):
        out, _ = self.generate([reply(429), reply(401)])
        self.assertEqual(out["attempt_log"], ["429", "401"])
        self.assertEqual(out["http_status"], 401)

    def test_a_network_error_is_logged_by_class_name(self):
        out, post = self.generate([requests.exceptions.ConnectionError("no route")])
        self.assertEqual(out["attempt_log"], ["ConnectionError"])
        self.assertEqual(post.call_count, 1)  # not retried
        self.assertIsNone(out["http_status"])

    def test_a_retry_after_that_is_too_long_logs_the_429_that_caused_it(self):
        out, post = self.generate([reply(429, headers={"Retry-After": "9999"})])
        self.assertEqual(out["attempt_log"], ["429"])
        self.assertEqual(post.call_count, 1)
        self.assertEqual(out["http_status"], 429)

    def test_the_log_holds_only_statuses_and_error_names_never_the_key_or_the_text(self):
        out, _ = self.generate([reply(500, content="secret body"), requests.exceptions.ReadTimeout(f"{KEY}"), reply(200)])
        self.assertTrue(all(isinstance(x, str) and len(x) < 40 for x in out["attempt_log"]))
        self.assertNotIn(KEY, json.dumps(out["attempt_log"]))
        self.assertNotIn("secret body", json.dumps(out["attempt_log"]))

    def test_the_log_always_has_one_entry_per_request(self):
        sequences = [[reply(200)], [reply(429), reply(200)], [reply(503)] * 5, [requests.exceptions.ReadTimeout("t")] * 2, [reply(404)]]
        for seq in sequences:
            with self.subTest(n=len(seq)):
                out, post = self.generate(list(seq))
                self.assertEqual(len(out["attempt_log"]), post.call_count)
                self.assertEqual(len(out["attempt_log"]), out["attempts"])


class TestRunnerStoresTheLog(LiveTestCase):
    def run_model(self, responses, limit=3):
        with mock.patch("src.gateway.model_client.requests.post", side_effect=responses):
            return EvaluationRunner(base_dir=REPO).run_evaluation(
                LIVE, ["gpqa"], job_name="real_job", limit_per_benchmark=limit, show_progress=False)

    def test_each_item_keeps_its_own_log(self):
        run = self.run_model([reply(429), reply(200), reply(200), reply(503), reply(503), reply(200)])
        self.assertEqual([i.attempt_log for i in run.item_results], [["429", "200"], ["200"], ["503", "503", "200"]])
        self.assertEqual([i.attempts for i in run.item_results], [2, 1, 3])
        self.assertTrue(all(i.http_status is None for i in run.item_results))  # all three finally succeeded

    def test_a_failed_item_keeps_the_intermediate_statuses_and_the_final_one(self):
        run = self.run_model([reply(500)] * 5 + [reply(200)] * 2)
        first = run.item_results[0]
        self.assertEqual(first.attempt_log, ["500"] * 5)
        self.assertEqual(first.http_status, 500)
        self.assertTrue(first.error.startswith("api_error"), first.error)

    def test_the_log_survives_saving_and_loading(self):
        run = self.run_model([reply(429), reply(200), reply(200), reply(200)])
        again = RunArtifact.model_validate(json.loads(run.model_dump_json()))
        self.assertEqual(again.item_results[0].attempt_log, ["429", "200"])

    def test_the_saved_file_never_contains_the_key(self):
        run = self.run_model([reply(401)] * 5 + [reply(200)] * 5)
        self.assertNotIn(KEY, run.model_dump_json())

    def test_simulated_items_have_an_empty_log(self):
        run = EvaluationRunner(base_dir=REPO).run_evaluation(
            {"id": "meta/llama-3.1-8b-instruct", "name": "T", "provider": "mock"}, ["gpqa"],
            job_name="standin_t", limit_per_benchmark=3, show_progress=False)
        self.assertEqual([i.attempt_log for i in run.item_results], [[], [], []])


class TestLegacyArtifacts(unittest.TestCase):
    def test_an_item_saved_before_the_field_existed_loads_with_an_empty_log(self):
        item = ItemResult(question_id="q", benchmark="gpqa", model_id="m", prompt="p", raw_response="r")
        data = json.loads(item.model_dump_json())
        data.pop("attempt_log")
        self.assertEqual(ItemResult.model_validate(data).attempt_log, [])


if __name__ == "__main__":
    unittest.main()
