import contextlib
import io
import os
import sys
import tempfile
import unittest
from unittest import mock
from urllib.parse import urlparse

import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)

from src.gateway.env_loader import load_env_file
from src.gateway.interceptor import InterceptorProxy
from src.gateway.model_client import (
    AUTO_PROVIDER_PRIORITY,
    DEFAULT_TIMEOUT_SECONDS,
    PROVIDER_SPECS,
    ProviderConfigError,
    UnifiedModelClient,
)

NVIDIA_HOST = "integrate.api.nvidia.com"
OPENAI_HOST = "api.openai.com"
GROQ_HOST = "api.groq.com"
OPENROUTER_HOST = "openrouter.ai"
NVIDIA_SENTINEL = "sentinel-nvidia-key-0001"
OPENAI_SENTINEL = "sentinel-openai-key-0002"
GROQ_SENTINEL = "sentinel-groq-key-0003"
OPENROUTER_SENTINEL = "sentinel-openrouter-key-0004"
ALL_KEYS = {
    "NVIDIA_API_KEY": NVIDIA_SENTINEL,
    "OPENAI_API_KEY": OPENAI_SENTINEL,
    "GROQ_API_KEY": GROQ_SENTINEL,
    "OPENROUTER_API_KEY": OPENROUTER_SENTINEL,
}


def _config(provider, **extra):
    return {"id": "vendor/some-model", "name": "Some Model", "provider": provider, **extra}


def _auto_config(**routes):
    return {
        "id": "vendor/requested-model",
        "name": "Auto Model",
        "provider": "auto",
        "routes": routes,
        "requests_per_minute": 600000,
        "cost_per_1m_input": 0.1,
        "cost_per_1m_output": 0.2,
    }


def _fake_response(content="hi", finish_reason="stop", status=200, **message_extra):
    resp = mock.MagicMock()
    resp.status_code = status
    resp.raise_for_status.return_value = None
    resp.json.return_value = {
        "choices": [{"message": {"content": content, **message_extra}, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 7},
    }
    return resp


class TestEnvLoader(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict(os.environ, {}, clear=False)
        patcher.start()
        self.addCleanup(patcher.stop)
        for name in ("ENVTEST_A", "ENVTEST_B", "ENVTEST_C", "ENVTEST_D", "ENVTEST_E", "ENVTEST_EXISTING"):
            os.environ.pop(name, None)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _write(self, text):
        path = os.path.join(self.tmp.name, ".env")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return path

    def test_missing_file_is_not_an_error(self):
        self.assertEqual(load_env_file(os.path.join(self.tmp.name, "nope.env")), [])

    def test_loads_plain_quoted_export_and_comment_forms(self):
        path = self._write(
            "# a comment\n"
            "\n"
            "ENVTEST_A=plain\n"
            'ENVTEST_B="double quoted # not a comment"\n'
            "export ENVTEST_C='single quoted'\n"
            "ENVTEST_D=value # trailing comment\n"
            "this line is malformed\n"
            "1BAD=nope\n"
        )
        names = load_env_file(path)
        self.assertEqual(sorted(names), ["ENVTEST_A", "ENVTEST_B", "ENVTEST_C", "ENVTEST_D"])
        self.assertEqual(os.environ["ENVTEST_A"], "plain")
        self.assertEqual(os.environ["ENVTEST_B"], "double quoted # not a comment")
        self.assertEqual(os.environ["ENVTEST_C"], "single quoted")
        self.assertEqual(os.environ["ENVTEST_D"], "value")
        self.assertNotIn("1BAD", os.environ)

    def test_never_overrides_an_existing_variable(self):
        os.environ["ENVTEST_EXISTING"] = "from-the-shell"
        path = self._write("ENVTEST_EXISTING=from-the-file\nENVTEST_E=new\n")
        names = load_env_file(path)
        self.assertEqual(os.environ["ENVTEST_EXISTING"], "from-the-shell")
        self.assertEqual(names, ["ENVTEST_E"])

    def test_an_empty_shell_value_still_counts_as_set(self):
        os.environ["ENVTEST_EXISTING"] = ""
        load_env_file(self._write("ENVTEST_EXISTING=from-the-file\n"))
        self.assertEqual(os.environ["ENVTEST_EXISTING"], "")

    def test_never_prints_values_and_returns_only_names(self):
        secret = "super-secret-value-123"
        path = self._write(f"ENVTEST_A={secret}\n")
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            names = load_env_file(path)
        self.assertEqual(out.getvalue() + err.getvalue(), "")
        self.assertEqual(names, ["ENVTEST_A"])
        self.assertNotIn(secret, repr(names))


class TestProviderKeyIsolation(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict(os.environ, ALL_KEYS)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_each_provider_defaults_to_its_own_key_variable(self):
        self.assertEqual(UnifiedModelClient(_config("nvidia_nim")).api_key_env, "NVIDIA_API_KEY")
        self.assertEqual(UnifiedModelClient(_config("openai")).api_key_env, "OPENAI_API_KEY")
        self.assertEqual(UnifiedModelClient(_config("groq")).api_key_env, "GROQ_API_KEY")
        self.assertEqual(UnifiedModelClient(_config("openrouter")).api_key_env, "OPENROUTER_API_KEY")

    def test_provider_cannot_be_pointed_at_another_providers_key(self):
        providers = {"nvidia_nim": "NVIDIA_API_KEY", "openai": "OPENAI_API_KEY", "groq": "GROQ_API_KEY",
                     "openrouter": "OPENROUTER_API_KEY"}
        for provider in providers:
            for other_provider, other_key in providers.items():
                if other_provider == provider:
                    continue
                with self.subTest(provider=provider, key_env=other_key), self.assertRaises(ProviderConfigError):
                    UnifiedModelClient(_config(provider, api_key_env=other_key))

    def test_openrouter_is_pinned_to_its_own_https_host_only(self):
        self.assertEqual(UnifiedModelClient(_config("openrouter")).api_base, "https://openrouter.ai/api/v1")
        bad_bases = [
            "https://api.openai.com/v1",
            "https://integrate.api.nvidia.com/v1",
            "https://api.groq.com/openai/v1",
            "https://evil.example/api/v1",
            "http://openrouter.ai/api/v1",                      # not https
            "https://openrouter.ai.evil.example/api/v1",        # look-alike suffix
            "https://openrouter.ai@evil.example/api/v1",        # userinfo trick
            "https://openrouter.ai:8443/api/v1",                # unexpected port
            "https://sub.openrouter.ai/api/v1",                 # other subdomain
        ]
        for base in bad_bases:
            with self.subTest(api_base=base), self.assertRaises(ProviderConfigError):
                UnifiedModelClient(_config("openrouter", api_base=base))

    def test_openrouter_without_its_key_fails_loudly_and_names_the_variable(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENROUTER_API_KEY", None)
            with self.assertRaises(ProviderConfigError) as ctx:
                UnifiedModelClient(_config("openrouter"))
        self.assertIn("OPENROUTER_API_KEY", str(ctx.exception))
        self.assertNotIn(OPENROUTER_SENTINEL, str(ctx.exception))

    def test_openrouter_request_goes_to_its_own_endpoint_without_following_redirects(self):
        client = UnifiedModelClient(_config("openrouter"), InterceptorProxy())
        with mock.patch("src.gateway.model_client.requests.post", return_value=_fake_response()) as post:
            client.generate("prompt")
        self.assertEqual(post.call_args.args[0], "https://openrouter.ai/api/v1/chat/completions")
        self.assertEqual(post.call_args.kwargs["headers"]["Authorization"], f"Bearer {OPENROUTER_SENTINEL}")
        self.assertFalse(post.call_args.kwargs["allow_redirects"])

    def test_groq_is_pinned_to_its_own_https_host_only(self):
        self.assertEqual(UnifiedModelClient(_config("groq")).api_base, "https://api.groq.com/openai/v1")
        bad_bases = [
            "https://api.openai.com/v1",
            "https://integrate.api.nvidia.com/v1",
            "https://evil.example/openai/v1",
            "http://api.groq.com/openai/v1",                    # not https
            "https://api.groq.com.evil.example/openai/v1",      # look-alike suffix
            "https://api.groq.com@evil.example/openai/v1",      # userinfo trick
            "https://api.groq.com:8443/openai/v1",              # unexpected port
        ]
        for base in bad_bases:
            with self.subTest(api_base=base), self.assertRaises(ProviderConfigError):
                UnifiedModelClient(_config("groq", api_base=base))

    def test_groq_without_its_key_fails_loudly_and_names_the_variable(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GROQ_API_KEY", None)
            with self.assertRaises(ProviderConfigError) as ctx:
                UnifiedModelClient(_config("groq"))
        self.assertIn("GROQ_API_KEY", str(ctx.exception))
        self.assertNotIn(GROQ_SENTINEL, str(ctx.exception))

    def test_groq_request_goes_to_its_own_chat_endpoint_without_following_redirects(self):
        client = UnifiedModelClient(_config("groq"), InterceptorProxy())
        with mock.patch("src.gateway.model_client.requests.post", return_value=_fake_response()) as post:
            client.generate("prompt")
        self.assertEqual(post.call_args.args[0], "https://api.groq.com/openai/v1/chat/completions")
        self.assertEqual(post.call_args.kwargs["headers"]["Authorization"], f"Bearer {GROQ_SENTINEL}")
        self.assertFalse(post.call_args.kwargs["allow_redirects"])

    def test_keyed_provider_rejects_foreign_or_malformed_hosts(self):
        bad_bases = [
            "https://api.openai.com/v1",                       # the other provider's host
            "https://evil.example/v1",
            "http://integrate.api.nvidia.com/v1",              # not https
            "https://integrate.api.nvidia.com.evil.example/v1",  # look-alike suffix
            "https://integrate.api.nvidia.com@evil.example/v1",  # userinfo trick
            "https://integrate.api.nvidia.com:8443/v1",        # unexpected port
        ]
        for base in bad_bases:
            with self.subTest(api_base=base), self.assertRaises(ProviderConfigError):
                UnifiedModelClient(_config("nvidia_nim", api_base=base))

    def test_missing_api_base_uses_the_providers_own_default(self):
        self.assertEqual(urlparse(UnifiedModelClient(_config("nvidia_nim")).api_base).hostname, NVIDIA_HOST)
        self.assertEqual(urlparse(UnifiedModelClient(_config("openai")).api_base).hostname, OPENAI_HOST)

    def test_configs_pair_each_model_with_the_key_of_its_own_host(self):
        with open(os.path.join(REPO, "configs", "models_config.yaml"), encoding="utf-8") as f:
            models = yaml.safe_load(f)["models"]
        expected = {NVIDIA_HOST: "NVIDIA_API_KEY", OPENAI_HOST: "OPENAI_API_KEY", GROQ_HOST: "GROQ_API_KEY",
                    OPENROUTER_HOST: "OPENROUTER_API_KEY"}
        for model_id, cfg in models.items():
            with self.subTest(model=model_id):
                if cfg.get("provider") == "auto":
                    routes = cfg.get("routes")
                    self.assertIsInstance(routes, dict)
                    self.assertTrue(routes, "auto model has no routes")
                    for route_provider, route in routes.items():
                        self.assertIn(route_provider, AUTO_PROVIDER_PRIORITY)
                        self.assertIn(route_provider, PROVIDER_SPECS)
                        if isinstance(route, str):
                            route = {"id": route}
                        self.assertTrue(route.get("id"), "auto route has no served model id")
                        self.assertIsInstance(route.get("substitute", False), bool)
                        if route.get("substitute"):
                            self.assertTrue(route.get("note"), "a substitute route must explain the substitution")
                else:
                    host = urlparse(cfg["api_base"]).hostname
                    self.assertEqual(cfg.get("api_key_env"), expected[host])

    def test_nvidia_key_can_only_ever_reach_the_nvidia_host(self):
        """Sweep every configured model x provider x key variable x endpoint. If any combination lets the
        NVIDIA, OpenAI, Groq or OpenRouter key leave its own host, this fails."""
        with open(os.path.join(REPO, "configs", "models_config.yaml"), encoding="utf-8") as f:
            models = yaml.safe_load(f)["models"]

        base_choices = [
            None,  # use whatever the model config says
            "https://api.openai.com/v1",
            "https://integrate.api.nvidia.com/v1",
            "https://api.groq.com/openai/v1",
            "https://openrouter.ai/api/v1",
            "https://evil.example/v1",
        ]
        key_choices = [None, "NVIDIA_API_KEY", "OPENAI_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY"]
        sent = []  # (url, headers, allow_redirects)

        def capture(url, headers=None, json=None, timeout=None, allow_redirects=True):
            sent.append((url, dict(headers or {}), allow_redirects))
            return _fake_response()

        built = blocked = 0
        with mock.patch("src.gateway.model_client.requests.post", side_effect=capture):
            for cfg in models.values():
                for provider in ("nvidia_nim", "openai", "groq", "openrouter", "ollama"):
                    for key_env in key_choices:
                        for base in base_choices:
                            variant = {**cfg, "provider": provider}
                            variant.pop("api_key_env", None)
                            if key_env:
                                variant["api_key_env"] = key_env
                            if base:
                                variant["api_base"] = base
                            try:
                                client = UnifiedModelClient(variant, InterceptorProxy())
                            except ProviderConfigError:
                                blocked += 1
                                continue
                            built += 1
                            client.generate("prompt")

        self.assertGreater(built, 0)
        self.assertGreater(blocked, 0)
        nvidia_requests = [s for s in sent if f"Bearer {NVIDIA_SENTINEL}" in s[1].get("Authorization", "")]
        openai_requests = [s for s in sent if f"Bearer {OPENAI_SENTINEL}" in s[1].get("Authorization", "")]
        groq_requests = [s for s in sent if f"Bearer {GROQ_SENTINEL}" in s[1].get("Authorization", "")]
        openrouter_requests = [s for s in sent if f"Bearer {OPENROUTER_SENTINEL}" in s[1].get("Authorization", "")]
        self.assertTrue(openrouter_requests, "sweep never exercised the OpenRouter key; the test would pass vacuously")
        for url, headers, allow_redirects in openrouter_requests:
            self.assertEqual(urlparse(url).hostname, OPENROUTER_HOST, f"OpenRouter key sent to {url}")
        self.assertTrue(nvidia_requests, "sweep never exercised the NVIDIA key; the test would pass vacuously")
        self.assertTrue(openai_requests, "sweep never exercised the OpenAI key; the test would pass vacuously")
        self.assertTrue(groq_requests, "sweep never exercised the Groq key; the test would pass vacuously")
        for url, headers, allow_redirects in nvidia_requests:
            self.assertEqual(urlparse(url).hostname, NVIDIA_HOST, f"NVIDIA key sent to {url}")
        for url, headers, allow_redirects in openai_requests:
            self.assertEqual(urlparse(url).hostname, OPENAI_HOST, f"OpenAI key sent to {url}")
        for url, headers, allow_redirects in groq_requests:
            self.assertEqual(urlparse(url).hostname, GROQ_HOST, f"Groq key sent to {url}")
        for url, headers, allow_redirects in sent:
            if "Authorization" in headers:
                self.assertFalse(allow_redirects, f"keyed request to {url} was allowed to follow redirects")
            else:
                self.assertNotIn("sentinel", " ".join(str(v) for v in headers.values()))
        # ollama is keyless: nothing it sends may carry a key
        ollama_requests = [s for s in sent if urlparse(s[0]).path.endswith("/api/generate")]
        self.assertTrue(ollama_requests)
        for url, headers, _ in ollama_requests:
            self.assertNotIn("Authorization", headers)

    def test_redirects_are_never_followed(self):
        client = UnifiedModelClient(_config("nvidia_nim"), InterceptorProxy())
        with mock.patch("src.gateway.model_client.requests.post", return_value=_fake_response(status=302)) as post:
            res = client.generate("prompt")
        self.assertFalse(post.call_args.kwargs["allow_redirects"])
        self.assertIn("error", res)


class TestAutoProvider(unittest.TestCase):
    def test_auto_uses_the_first_route_whose_key_is_set_in_priority_order(self):
        cfg = _auto_config(
            openrouter={"id": "served/openrouter-model"},
            groq={"id": "served/groq-model"},
            nvidia_nim={"id": "served/nim-model"},
        )
        with mock.patch.dict(os.environ, {"GROQ_API_KEY": GROQ_SENTINEL}, clear=True):
            self.assertEqual(UnifiedModelClient(cfg).provider, "groq")
        with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": OPENROUTER_SENTINEL}, clear=True):
            self.assertEqual(UnifiedModelClient(cfg).provider, "openrouter")
        with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": OPENROUTER_SENTINEL,
                                          "NVIDIA_API_KEY": NVIDIA_SENTINEL}, clear=True):
            self.assertEqual(UnifiedModelClient(cfg).provider, "nvidia_nim")

    def test_auto_records_the_served_model_and_the_substitution(self):
        cfg = _auto_config(openrouter={
            "id": "served/replacement-model",
            "substitute": True,
            "note": "The exact variant is not hosted.",
        })
        with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": OPENROUTER_SENTINEL}, clear=True):
            client = UnifiedModelClient(cfg, InterceptorProxy())
            self.assertEqual(client.provider, "openrouter")
            self.assertEqual(client.requested_model_id, "vendor/requested-model")
            self.assertEqual(client.model_id, "served/replacement-model")
            self.assertEqual(client.served_model_id, "served/replacement-model")
            self.assertTrue(client.route_substitute)
            self.assertEqual(client.route_note, "The exact variant is not hosted.")
            self.assertEqual(client.api_base, "https://openrouter.ai/api/v1")
            self.assertEqual(client.api_key_env, "OPENROUTER_API_KEY")
            with mock.patch("src.gateway.model_client.requests.post", return_value=_fake_response()) as post:
                client.generate("prompt")
        self.assertEqual(post.call_args.args[0], "https://openrouter.ai/api/v1/chat/completions")
        self.assertEqual(post.call_args.kwargs["json"]["model"], "served/replacement-model")
        self.assertEqual(post.call_args.kwargs["headers"]["Authorization"], f"Bearer {OPENROUTER_SENTINEL}")

    def test_auto_without_any_route_key_names_the_keys_and_never_falls_back_to_mock(self):
        cfg = _auto_config(
            openrouter={"id": "served/openrouter-model"},
            nvidia_nim={"id": "served/nim-model"},
        )
        with mock.patch.dict(os.environ, {}, clear=True), self.assertRaises(ProviderConfigError) as ctx:
            UnifiedModelClient(cfg)
        message = str(ctx.exception)
        self.assertIn("OPENROUTER_API_KEY", message)
        self.assertIn("NVIDIA_API_KEY", message)
        self.assertIn("Refusing to fall back to the simulator", message)

    def test_auto_requires_a_routes_map_and_only_known_routes(self):
        with self.assertRaises(ProviderConfigError):
            UnifiedModelClient({"id": "vendor/m", "provider": "auto"})
        with self.assertRaises(ProviderConfigError):
            UnifiedModelClient({"id": "vendor/m", "provider": "auto", "routes": {"huggingface": {"id": "x"}}})
        with self.assertRaises(ProviderConfigError):
            UnifiedModelClient({"id": "vendor/m", "provider": "auto", "routes": {"openrouter": {}}})

    def test_auto_cache_is_keyed_by_the_requested_model_not_the_served_model(self):
        a = _auto_config(openrouter={"id": "shared/served-model"})
        b = {**a, "id": "vendor/other-requested-model"}
        cache = InterceptorProxy()
        with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": OPENROUTER_SENTINEL}, clear=True), \
                mock.patch("src.gateway.model_client.requests.post", return_value=_fake_response()) as post:
            UnifiedModelClient(a, cache).generate("same prompt")
            UnifiedModelClient(b, cache).generate("same prompt")
        self.assertEqual(post.call_count, 2, "two requested models sharing a served model must not share cache entries")


class TestListModels(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict(os.environ, ALL_KEYS)
        patcher.start()
        self.addCleanup(patcher.stop)

    @staticmethod
    def _listing(ids, status=200):
        resp = mock.MagicMock()
        resp.status_code = status
        resp.json.return_value = {"data": [{"id": i, "object": "model"} for i in ids]}
        return resp

    def test_returns_sorted_ids_from_the_providers_own_host_only(self):
        client = UnifiedModelClient(_config("nvidia_nim"), InterceptorProxy())
        with mock.patch("src.gateway.model_client.requests.get", return_value=self._listing(["b/two", "a/one"])) as get:
            self.assertEqual(client.list_models(), ["a/one", "b/two"])
        url = get.call_args.args[0]
        self.assertEqual(urlparse(url).hostname, NVIDIA_HOST)
        self.assertTrue(url.endswith("/models"))
        self.assertEqual(get.call_args.kwargs["headers"]["Authorization"], f"Bearer {NVIDIA_SENTINEL}")
        self.assertFalse(get.call_args.kwargs["allow_redirects"])
        self.assertEqual(get.call_args.kwargs["timeout"], 120.0)

    def test_the_key_is_never_sent_to_another_host(self):
        for provider, sentinel, host in (("nvidia_nim", NVIDIA_SENTINEL, NVIDIA_HOST), ("openai", OPENAI_SENTINEL, OPENAI_HOST),
                                         ("groq", GROQ_SENTINEL, GROQ_HOST), ("openrouter", OPENROUTER_SENTINEL, OPENROUTER_HOST)):
            client = UnifiedModelClient(_config(provider), InterceptorProxy())
            with mock.patch("src.gateway.model_client.requests.get", return_value=self._listing(["x"])) as get:
                client.list_models()
            self.assertEqual(urlparse(get.call_args.args[0]).hostname, host)
            self.assertIn(sentinel, get.call_args.kwargs["headers"]["Authorization"])

    def test_http_errors_and_redirects_raise_without_leaking_the_key(self):
        client = UnifiedModelClient(_config("nvidia_nim"), InterceptorProxy())
        for status in (401, 429, 302):
            with self.subTest(status=status), mock.patch("src.gateway.model_client.requests.get", return_value=self._listing([], status)):
                with self.assertRaises(RuntimeError) as ctx:
                    client.list_models()
                self.assertNotIn(NVIDIA_SENTINEL, str(ctx.exception))

    def test_not_available_for_mock_or_ollama(self):
        for provider in ("mock", "ollama"):
            with self.subTest(provider=provider), self.assertRaises(ProviderConfigError):
                UnifiedModelClient(_config(provider)).list_models()


class TestModelRequestSettings(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict(os.environ, {"NVIDIA_API_KEY": NVIDIA_SENTINEL})
        patcher.start()
        self.addCleanup(patcher.stop)

    def _call(self, cfg):
        client = UnifiedModelClient(cfg, InterceptorProxy())
        with mock.patch("src.gateway.model_client.requests.post", return_value=_fake_response()) as post:
            client.generate("prompt")
        return post.call_args

    def test_default_timeout_is_120_seconds(self):
        self.assertEqual(DEFAULT_TIMEOUT_SECONDS, 120.0)
        self.assertEqual(self._call(_config("nvidia_nim")).kwargs["timeout"], 120.0)

    def test_timeout_is_configurable_per_model(self):
        self.assertEqual(self._call(_config("nvidia_nim", timeout_seconds=300)).kwargs["timeout"], 300.0)

    def test_invalid_timeout_is_rejected(self):
        for bad in (0, -5, "abc", None):
            with self.subTest(timeout=bad), self.assertRaises(ProviderConfigError):
                UnifiedModelClient(_config("nvidia_nim", timeout_seconds=bad))

    def test_max_tokens_is_configurable_per_model(self):
        call = self._call(_config("nvidia_nim", params={"temperature": 0.1, "max_tokens": 777, "top_p": 0.95}))
        self.assertEqual(call.kwargs["json"]["max_tokens"], 777)

    def test_ollama_uses_the_same_timeout_setting(self):
        client = UnifiedModelClient(_config("ollama", timeout_seconds=45), InterceptorProxy())
        resp = mock.MagicMock()
        resp.status_code = 200
        resp.raise_for_status.return_value = None
        resp.json.return_value = {"response": "hi", "done_reason": "length", "prompt_eval_count": 1, "eval_count": 1}
        with mock.patch("src.gateway.model_client.requests.post", return_value=resp) as post:
            res = client.generate("prompt")
        self.assertEqual(post.call_args.kwargs["timeout"], 45.0)
        self.assertEqual(res["finish_reason"], "length")


class TestResponseCapture(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict(os.environ, {"NVIDIA_API_KEY": NVIDIA_SENTINEL})
        patcher.start()
        self.addCleanup(patcher.stop)

    def _generate(self, response):
        client = UnifiedModelClient(_config("nvidia_nim"), InterceptorProxy())
        with mock.patch("src.gateway.model_client.requests.post", return_value=response):
            return client.generate("prompt")

    def test_null_content_with_length_finish_is_captured_not_crashed(self):
        res = self._generate(_fake_response(content=None, finish_reason="length"))
        self.assertNotIn("error", res)
        self.assertEqual(res["text"], "")
        self.assertEqual(res["finish_reason"], "length")

    def test_separate_reasoning_content_is_kept_apart_from_the_answer(self):
        res = self._generate(_fake_response(content="Answer: C", reasoning_content="thinking about B"))
        self.assertEqual(res["text"], "Answer: C")
        self.assertEqual(res["reasoning_text"], "thinking about B")

    def test_failed_calls_are_not_cached(self):
        client = UnifiedModelClient(_config("nvidia_nim"), InterceptorProxy())
        with mock.patch("src.gateway.model_client.requests.post", side_effect=RuntimeError("boom")) as post, \
                mock.patch.object(UnifiedModelClient, "_sleep"):  # the per-model throttle would otherwise really wait
            first = client.generate("same prompt")
            client.generate("same prompt")
        self.assertIn("error", first)
        self.assertEqual(post.call_count, 2)


if __name__ == "__main__":
    unittest.main()
