import json
import os
import sys
import unittest
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from unittest import mock

import requests

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)

from src.benchmarks.base import get_benchmark
from src.comparison.diff_engine import DiffEngine
from src.engine.runner import EvaluationRunner
from src.gateway.interceptor import InterceptorProxy
from src.gateway.model_client import (
    BACKOFF_BASE_SECONDS,
    DEFAULT_REQUESTS_PER_MINUTE,
    MAX_RETRIES,
    ProviderConfigError,
    UnifiedModelClient,
)
from src.reporting.report_generator import ReportGenerator
from src.storage.schema import RunArtifact

KEY = "sentinel-nvidia-key-rate-limits"
GROQ_KEY = "sentinel-groq-key-rate-limits"


class FakeTime:
    """A clock that only moves when something sleeps (or the test advances it)."""

    def __init__(self):
        self.t = 1000.0
        self.sleeps = []

    def sleep(self, seconds):
        self.sleeps.append(round(seconds, 6))
        self.t += seconds

    def now(self):
        return self.t


def response(status=200, headers=None, content="Answer: C", finish_reason="stop"):
    r = mock.MagicMock()
    r.status_code = status
    r.headers = headers or {}
    if status >= 400:
        r.raise_for_status.side_effect = requests.exceptions.HTTPError(
            f"{status} Client Error: Synthetic for url: https://integrate.api.nvidia.com/v1/chat/completions"
        )
    else:
        r.raise_for_status.return_value = None
    r.json.return_value = {
        "choices": [{"message": {"content": content}, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 7},
    }
    return r


def make_client(provider="nvidia_nim", **extra):
    cfg = {"id": "vendor/model", "name": "M", "provider": provider, **extra}
    client = UnifiedModelClient(cfg, InterceptorProxy())
    fake = FakeTime()
    client._sleep, client._now = fake.sleep, fake.now
    return client, fake


class KeyedTestCase(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict(os.environ, {"NVIDIA_API_KEY": KEY, "GROQ_API_KEY": GROQ_KEY})
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_generate(self, client, responses, prompt="prompt"):
        with mock.patch("src.gateway.model_client.requests.post", side_effect=responses) as post:
            res = client.generate(prompt)
        return res, post


class TestRetries(KeyedTestCase):
    def test_constants_match_the_spec(self):
        self.assertEqual((MAX_RETRIES, BACKOFF_BASE_SECONDS, DEFAULT_REQUESTS_PER_MINUTE), (4, 2.0, 20))

    def test_429_then_success(self):
        client, fake = make_client(requests_per_minute=6000)
        res, post = self.run_generate(client, [response(429), response(200)])
        self.assertNotIn("error", res)
        self.assertEqual(res["attempts"], 2)
        self.assertEqual(post.call_count, 2)
        self.assertEqual(fake.sleeps, [2.0])

    def test_5xx_statuses_are_retried(self):
        for status in (500, 502, 503, 504):
            with self.subTest(status=status):
                client, fake = make_client(requests_per_minute=6000)
                res, post = self.run_generate(client, [response(status), response(200)])
                self.assertNotIn("error", res)
                self.assertEqual(post.call_count, 2)

    def test_gives_up_after_four_retries_with_exponential_backoff(self):
        client, fake = make_client(requests_per_minute=6000)
        res, post = self.run_generate(client, [response(429)] * 10)
        self.assertEqual(post.call_count, 5)  # 1 try + 4 retries
        self.assertEqual(fake.sleeps, [2.0, 4.0, 8.0, 16.0])
        self.assertIn("error", res)
        self.assertEqual(res["attempts"], 5)
        self.assertIn("HTTP 429 after 5 attempt(s), 4 retried", res["error"])
        self.assertEqual(res["total_tokens"], 0)

    def test_a_persistent_5xx_is_reported_with_its_status(self):
        client, fake = make_client(requests_per_minute=6000)
        res, post = self.run_generate(client, [response(503)] * 10)
        self.assertEqual(post.call_count, 5)
        self.assertIn("HTTP 503", res["error"])

    def test_retry_after_seconds_is_honoured_when_longer_than_the_backoff(self):
        for header in ("Retry-After", "retry-after"):
            with self.subTest(header=header):
                client, fake = make_client(requests_per_minute=6000)
                self.run_generate(client, [response(429, {header: "7"}), response(200)])
                self.assertEqual(fake.sleeps, [7.0])

    def test_retry_after_shorter_than_the_backoff_uses_the_backoff(self):
        client, fake = make_client(requests_per_minute=6000)
        self.run_generate(client, [response(429, {"Retry-After": "1"}), response(200)])
        self.assertEqual(fake.sleeps, [2.0])

    def test_retry_after_as_an_http_date(self):
        when = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=30), usegmt=True)
        client, fake = make_client(requests_per_minute=6000)
        self.run_generate(client, [response(429, {"Retry-After": when}), response(200)])
        self.assertEqual(len(fake.sleeps), 1)
        self.assertTrue(27.0 <= fake.sleeps[0] <= 31.0, fake.sleeps)

    def test_unreadable_retry_after_is_ignored(self):
        for junk in ("soon", "", "-5"):
            with self.subTest(header=junk):
                client, fake = make_client(requests_per_minute=6000)
                self.run_generate(client, [response(429, {"Retry-After": junk}), response(200)])
                self.assertEqual(fake.sleeps, [2.0])

    def test_retry_after_beyond_the_maximum_wait_fails_without_waiting(self):
        client, fake = make_client(requests_per_minute=6000)
        res, post = self.run_generate(client, [response(429, {"Retry-After": "300"}), response(200)])
        self.assertEqual(post.call_count, 1)
        self.assertEqual(fake.sleeps, [])
        self.assertIn("Retry-After", res["error"])
        self.assertIn("error", res)

    def test_other_4xx_errors_are_never_retried(self):
        for status in (400, 401, 403, 404, 408, 409, 410, 413, 422):
            with self.subTest(status=status):
                client, fake = make_client(requests_per_minute=6000)
                res, post = self.run_generate(client, [response(status), response(200)])
                self.assertEqual(post.call_count, 1)
                self.assertEqual(fake.sleeps, [])
                self.assertIn("error", res)
                self.assertEqual(res["attempts"], 1)
                self.assertIn(str(status), res["error"])

    def test_network_errors_and_timeouts_are_not_retried(self):
        for exc in (requests.exceptions.Timeout("timed out"), requests.exceptions.ConnectionError("no route")):
            with self.subTest(exc=type(exc).__name__):
                client, fake = make_client(requests_per_minute=6000)
                res, post = self.run_generate(client, [exc, response(200)])
                self.assertEqual(post.call_count, 1)
                self.assertIn("error", res)

    def test_a_read_timeout_is_retried_once_and_can_then_succeed(self):
        client, fake = make_client(requests_per_minute=6000)
        res, post = self.run_generate(client, [requests.exceptions.ReadTimeout("slow"), response(200)])
        self.assertNotIn("error", res)
        self.assertEqual((post.call_count, res["attempts"]), (2, 2))
        self.assertEqual(fake.sleeps, [2.0])

    def test_two_read_timeouts_in_a_row_fail_after_exactly_two_attempts(self):
        client, fake = make_client(requests_per_minute=6000)
        res, post = self.run_generate(client, [requests.exceptions.ReadTimeout("slow")] * 5 + [response(200)])
        self.assertEqual(post.call_count, 2)
        self.assertEqual(res["attempts"], 2)
        self.assertIn("read timed out after 2 attempt(s)", res["error"])

    def test_connect_timeouts_and_connection_errors_are_still_not_retried(self):
        for exc in (requests.exceptions.ConnectTimeout("c"), requests.exceptions.ConnectionError("n"),
                    requests.exceptions.SSLError("s")):
            with self.subTest(exc=type(exc).__name__):
                client, fake = make_client(requests_per_minute=6000)
                res, post = self.run_generate(client, [exc, response(200)])
                self.assertEqual(post.call_count, 1)
                self.assertIn("error", res)

    def test_the_timeout_retry_is_separate_from_the_429_retries(self):
        client, fake = make_client(requests_per_minute=6000)
        res, post = self.run_generate(client, [response(429), requests.exceptions.ReadTimeout("slow"), response(429), response(200)])
        self.assertNotIn("error", res)
        self.assertEqual(res["attempts"], 4)
        self.assertEqual(fake.sleeps, [2.0, 2.0, 4.0])

    def test_the_timeout_retry_is_throttled_like_any_other_request(self):
        client, fake = make_client()  # 3 s between request starts
        self.run_generate(client, [requests.exceptions.ReadTimeout("slow"), response(200)])
        self.assertEqual(fake.sleeps, [2.0, 1.0])

    def test_timeout_errors_do_not_contain_the_key(self):
        client, fake = make_client(requests_per_minute=6000)
        res, _ = self.run_generate(client, [requests.exceptions.ReadTimeout(f"slow {KEY}")] * 2)
        self.assertNotIn(KEY, json.dumps(res))

    def test_redirects_are_not_retried(self):
        client, fake = make_client(requests_per_minute=6000)
        res, post = self.run_generate(client, [response(302), response(200)])
        self.assertEqual(post.call_count, 1)
        self.assertIn("redirect", res["error"])

    def test_a_first_try_success_reports_one_attempt(self):
        client, fake = make_client()
        res, _ = self.run_generate(client, [response(200)])
        self.assertEqual(res["attempts"], 1)

    def test_latency_is_the_last_request_not_the_waiting(self):
        client, fake = make_client(requests_per_minute=6000)
        calls = iter([response(429), response(200)])

        def slow_post(*args, **kwargs):
            fake.t += 0.5  # each request itself takes half a second
            return next(calls)

        with mock.patch("src.gateway.model_client.requests.post", side_effect=slow_post):
            res = client.generate("prompt")
        self.assertEqual(res["latency_sec"], 0.5)
        self.assertEqual(fake.sleeps, [2.0])

    def test_failures_are_not_cached_but_successes_are(self):
        client, fake = make_client(requests_per_minute=6000)
        failed, post = self.run_generate(client, [response(401), response(200)], prompt="p1")
        again, post2 = self.run_generate(client, [response(200)], prompt="p1")
        self.assertNotIn("error", again)
        self.assertEqual(post2.call_count, 1)

    def test_error_messages_never_contain_the_key(self):
        cases = [[response(429)] * 10, [response(401)], [response(302)], [requests.exceptions.Timeout("t")],
                 [response(429, {"Retry-After": "999"})]]
        for responses in cases:
            client, fake = make_client(requests_per_minute=6000)
            res, _ = self.run_generate(client, responses)
            blob = json.dumps(res)
            self.assertNotIn(KEY, blob)
            self.assertNotIn("sentinel-nvidia", blob)

    def test_groq_gets_the_same_retry_behaviour(self):
        client, fake = make_client("groq", requests_per_minute=6000)
        res, post = self.run_generate(client, [response(429, {"retry-after": "3"}), response(200)])
        self.assertNotIn("error", res)
        self.assertEqual(fake.sleeps, [3.0])  # Retry-After (3 s) is longer than the first backoff (2 s)
        self.assertTrue(post.call_args.args[0].startswith("https://api.groq.com/openai/v1/"))

    def test_ollama_retries_too_but_keeps_its_own_endpoint(self):
        client, fake = make_client("ollama")
        ok = mock.MagicMock()
        ok.status_code = 200
        ok.headers = {}
        ok.raise_for_status.return_value = None
        ok.json.return_value = {"response": "hi", "done_reason": "stop", "prompt_eval_count": 1, "eval_count": 1}
        res, post = self.run_generate(client, [response(503), ok])
        self.assertEqual(res["text"], "hi")
        self.assertEqual(res["attempts"], 2)
        self.assertTrue(post.call_args.args[0].endswith("/api/generate"))


class TestThrottle(KeyedTestCase):
    def test_default_is_20_requests_per_minute(self):
        client, _ = make_client()
        self.assertEqual(client.min_interval, 3.0)

    def test_requests_are_spaced_by_the_minimum_interval(self):
        client, fake = make_client()
        for i in range(3):
            self.run_generate(client, [response(200)], prompt=f"prompt {i}")
        self.assertEqual(fake.sleeps, [3.0, 3.0])

    def test_rpm_is_configurable_per_model(self):
        for rpm, interval in ((60, 1.0), (120, 0.5), (30, 2.0), ("40", 1.5)):
            with self.subTest(rpm=rpm):
                client, fake = make_client(requests_per_minute=rpm)
                self.assertEqual(client.min_interval, interval)
                for i in range(2):
                    self.run_generate(client, [response(200)], prompt=f"p{i}")
                self.assertEqual(fake.sleeps, [interval])

    def test_no_wait_when_enough_time_has_passed(self):
        client, fake = make_client()
        self.run_generate(client, [response(200)], prompt="a")
        fake.t += 10
        self.run_generate(client, [response(200)], prompt="b")
        self.assertEqual(fake.sleeps, [])

    def test_cache_hits_are_neither_throttled_nor_sent(self):
        client, fake = make_client()
        _, post = self.run_generate(client, [response(200)], prompt="same")
        _, post2 = self.run_generate(client, [response(200)], prompt="same")
        self.assertEqual(post2.call_count, 0)
        self.assertEqual(fake.sleeps, [])

    def test_every_retry_attempt_respects_the_throttle_too(self):
        client, fake = make_client()  # 3 s between request starts
        self.run_generate(client, [response(429), response(200)])
        # waits 2 s for the backoff, then 1 more second so the retry starts 3 s after the first attempt
        self.assertEqual(fake.sleeps, [2.0, 1.0])

    def test_each_model_has_its_own_throttle(self):
        a, fake_a = make_client()
        b, fake_b = make_client()
        self.run_generate(a, [response(200)], prompt="x")
        self.run_generate(b, [response(200)], prompt="x")
        self.assertEqual(fake_a.sleeps + fake_b.sleeps, [])

    def test_invalid_rpm_is_rejected(self):
        for bad in (0, -1, "fast", True, [], 0.0):
            with self.subTest(rpm=bad), self.assertRaises(ProviderConfigError):
                make_client(requests_per_minute=bad)

    def test_simulator_and_local_ollama_are_not_throttled_by_default(self):
        self.assertEqual(UnifiedModelClient({"id": "m", "provider": "mock"}).min_interval, 0.0)
        self.assertEqual(UnifiedModelClient({"id": "m", "provider": "ollama"}).min_interval, 0.0)
        self.assertEqual(UnifiedModelClient({"id": "m", "provider": "ollama", "requests_per_minute": 10}).min_interval, 6.0)


class TestApiErrorAccounting(KeyedTestCase):
    def _run(self, status_sequence_factory):
        bench = get_benchmark("gpqa", base_dir=REPO)
        truths = [it["answer"] for it in bench.load_items(limit=4)]
        responses = status_sequence_factory(truths)
        model = {"id": "vendor/model", "name": "Live Model", "provider": "nvidia_nim"}
        with mock.patch("src.gateway.model_client.requests.post", side_effect=responses) as post, \
                mock.patch.object(UnifiedModelClient, "_sleep"):
            run = EvaluationRunner(base_dir=REPO).run_evaluation(model, ["gpqa"], job_name="standin_t",
                                                                 limit_per_benchmark=4, show_progress=False)
        return run, post

    @staticmethod
    def _sequence(truths):
        return [response(200, content=f"Answer: {truths[0]}"),
                *[response(429)] * 5,                         # item 2: rate limited through every retry
                response(200, content=f"Answer: {truths[2]}"),
                response(401)]                                # item 4: a client error, no retry

    def test_failed_items_are_incorrect_flagged_and_counted(self):
        run, post = self._run(self._sequence)
        self.assertEqual(run.run_mode, "live")
        self.assertEqual([i.is_correct for i in run.item_results], [True, False, True, False])
        errors = [i.error for i in run.item_results]
        self.assertIsNone(errors[0])
        self.assertTrue(errors[1].startswith("api_error: HTTP 429 after 5 attempt(s)"), errors[1])
        self.assertIsNone(errors[2])
        self.assertTrue(errors[3].startswith("api_error:") and "401" in errors[3], errors[3])
        self.assertEqual([i.attempts for i in run.item_results], [1, 5, 1, 1])
        self.assertEqual(post.call_count, 8)
        summary = run.benchmark_summaries["gpqa"]
        self.assertEqual((summary.api_error_count, run.api_error_count), (2, 2))
        self.assertEqual((summary.truncated_count, summary.no_answer_count), (0, 0))
        self.assertEqual(summary.accuracy, 0.5)

    def test_extracted_answer_is_not_taken_from_an_error_message(self):
        run, _ = self._run(lambda truths: [response(429)] * 20)
        for item in run.item_results:
            self.assertIsNone(item.extracted_answer)
            self.assertFalse(item.is_correct)

    def test_the_report_shows_the_api_error_count(self):
        run, _ = self._run(self._sequence)
        md = ReportGenerator.generate_markdown(DiffEngine.compare_runs(run, run))
        self.assertIn("2 api_error (call failed)", md)
        self.assertEqual(DiffEngine.compare_runs(run, run)["api_error_a"], 2)

    def test_counts_survive_a_json_round_trip_and_old_files_still_load(self):
        run, _ = self._run(self._sequence)
        again = RunArtifact.model_validate_json(run.model_dump_json())
        self.assertEqual(again.api_error_count, 2)
        data = json.loads(run.model_dump_json())
        data.pop("api_error_count")
        for summary in data["benchmark_summaries"].values():
            summary.pop("api_error_count")
        for item in data["item_results"]:
            item.pop("attempts")
        old = RunArtifact.model_validate(data)
        self.assertEqual((old.api_error_count, old.item_results[0].attempts), (0, 1))


if __name__ == "__main__":
    unittest.main()
