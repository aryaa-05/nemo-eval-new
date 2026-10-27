import os
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import requests
from typing import Dict, Any, List, Optional, Tuple
from urllib.parse import urlparse
from src.gateway.interceptor import InterceptorProxy
from src.gateway.mock_models import MockModelEngine

DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_REQUESTS_PER_MINUTE = 20      # per model, live providers only (ollama: only when configured)
MAX_RETRIES = 4                       # extra attempts after a 429 or 5xx, so at most 5 requests per item
MAX_TIMEOUT_RETRIES = 1               # a read timeout is retried once (other network errors are not retried)
BACKOFF_BASE_SECONDS = 2.0            # waits of 2, 4, 8, 16 seconds between attempts
MAX_RETRY_WAIT_SECONDS = 120.0        # a longer Retry-After is not waited for; the item fails instead

# A key is only ever sent to the hosts listed for its own provider.
PROVIDER_SPECS = {
    "nvidia_nim": {
        "api_key_env": "NVIDIA_API_KEY",
        "api_base": "https://integrate.api.nvidia.com/v1",
        "hosts": ("integrate.api.nvidia.com",),
    },
    "openai": {
        "api_key_env": "OPENAI_API_KEY",
        "api_base": "https://api.openai.com/v1",
        "hosts": ("api.openai.com",),
    },
    "groq": {
        "api_key_env": "GROQ_API_KEY",
        "api_base": "https://api.groq.com/openai/v1",
        "hosts": ("api.groq.com",),
    },
    "openrouter": {
        "api_key_env": "OPENROUTER_API_KEY",
        "api_base": "https://openrouter.ai/api/v1",
        "hosts": ("openrouter.ai",),
    },
}
KEY_ENV_OWNERS = {spec["api_key_env"]: provider for provider, spec in PROVIDER_SPECS.items()}
OLLAMA_DEFAULT_BASE = "http://localhost:11434"

# provider: "auto" picks the first route whose key is present, in this order (free tiers before paid).
AUTO_PROVIDER = "auto"
AUTO_PROVIDER_PRIORITY = ("nvidia_nim", "groq", "openrouter", "openai")


class ProviderConfigError(RuntimeError):
    """A model is configured for a provider that cannot be used (missing API key, unsupported provider, bad endpoint)."""


class UnifiedModelClient:
    KEYED_PROVIDERS = tuple(PROVIDER_SPECS)
    KEYLESS_PROVIDERS = ("ollama",)

    def __init__(self, model_config: Dict[str, Any], interceptor: Optional[InterceptorProxy] = None, seed: int = 42):
        requested_id = model_config["id"]
        self.config = model_config
        self.requested_model_id = requested_id
        self.model_id = requested_id
        self.served_model_id = requested_id
        self.model_name = model_config.get("name", requested_id)
        self.route_note = None
        self.route_substitute = False
        self.provider = model_config.get("provider", "mock")
        self.params = model_config.get("params", {"temperature": 0.1, "max_tokens": 1024, "top_p": 0.95})
        self.interceptor = interceptor or InterceptorProxy()
        self.seed = seed
        self.timeout = self._read_timeout(model_config)
        self._next_slot = 0.0  # earliest time (on the monotonic clock) the next request may start
        self.api_key_env = None
        self.api_key = ""
        self.api_base = model_config.get("api_base")
        self._cache_model_id = requested_id  # requested id, so two routes to one model do not share cached costs

        if self.provider == AUTO_PROVIDER:
            resolved = self._resolve_auto_route(model_config)
            self.provider = resolved["provider"]
            self.model_id = resolved["id"]
            self.served_model_id = resolved["id"]
            self.route_note = resolved["note"]
            self.route_substitute = resolved["substitute"]
            self.api_base = resolved["api_base"]
            self.api_key_env = resolved["api_key_env"]
            self.config = {
                **model_config,
                "provider": self.provider,
                "id": self.model_id,
                "api_base": self.api_base,
                "api_key_env": self.api_key_env,
            }
            model_config = self.config

        self.min_interval = self._read_min_interval(model_config, self.provider)

        if self.provider == "mock":
            return

        if self.provider not in self.KEYED_PROVIDERS + self.KEYLESS_PROVIDERS:
            raise ProviderConfigError(
                f"Model '{self.model_id}' uses provider '{self.provider}', which is not implemented. "
                f"Supported: mock, {', '.join(self.KEYED_PROVIDERS + self.KEYLESS_PROVIDERS)}. "
                f"Refusing to fall back to the simulator."
            )

        if self.provider in self.KEYLESS_PROVIDERS:
            self.api_base = (self.api_base or OLLAMA_DEFAULT_BASE).rstrip("/")
            return

        spec = PROVIDER_SPECS[self.provider]
        self.api_key_env = model_config.get("api_key_env") or spec["api_key_env"]
        owner = KEY_ENV_OWNERS.get(self.api_key_env)
        if owner is not None and owner != self.provider:
            raise ProviderConfigError(
                f"Model '{self.model_id}' (provider '{self.provider}') is configured to use {self.api_key_env}, "
                f"which belongs to provider '{owner}'. A key is only sent to its own provider."
            )
        self.api_base = (self.api_base or spec["api_base"]).rstrip("/")
        self._check_endpoint(spec)

        self.api_key = os.environ.get(self.api_key_env, "")
        if not self.api_key:
            raise ProviderConfigError(
                f"Model '{self.model_id}' is set to live provider '{self.provider}' but environment variable "
                f"{self.api_key_env} is not set. Set it in your shell or in .env, or switch the model to provider 'mock'. "
                f"Refusing to fall back to the simulator."
            )

    def _resolve_auto_route(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Pick the first configured route whose API key is present. Never guesses a provider.

        A route is `routes: {provider: {id: served-model-id, note: ..., substitute: true/false}}`. The requested
        model id stays in the run artifact; the served id and note are recorded separately so a substitution is visible.
        """
        requested = model_config.get("id", "?")
        routes = model_config.get("routes")
        if not isinstance(routes, dict) or not routes:
            raise ProviderConfigError(
                f"Model '{requested}' uses provider 'auto' but has no 'routes:' map. Add at least one route "
                f"(for example openrouter: {{id: 'vendor/model'}}). Refusing to fall back to the simulator."
            )
        unknown = sorted(set(routes) - set(AUTO_PROVIDER_PRIORITY))
        if unknown:
            raise ProviderConfigError(
                f"Model '{requested}' has unsupported auto route(s) {', '.join(unknown)}. Supported routes: "
                f"{', '.join(AUTO_PROVIDER_PRIORITY)}."
            )

        tried = []
        for provider in AUTO_PROVIDER_PRIORITY:
            if provider not in routes:
                continue
            route = routes[provider]
            if isinstance(route, str):
                route = {"id": route}
            if not isinstance(route, dict) or not route.get("id"):
                raise ProviderConfigError(
                    f"Model '{requested}' has an invalid auto route for '{provider}'; it must give an 'id'."
                )
            spec = PROVIDER_SPECS[provider]
            key_env = spec["api_key_env"]
            tried.append(f"{provider} ({key_env})")
            if not os.environ.get(key_env):
                continue
            return {
                "provider": provider,
                "id": str(route["id"]),
                "api_base": spec["api_base"],
                "api_key_env": key_env,
                "note": str(route.get("note") or ""),
                "substitute": bool(route.get("substitute", False)),
            }

        key_names = ", ".join(PROVIDER_SPECS[p]["api_key_env"] for p in AUTO_PROVIDER_PRIORITY if p in routes)
        raise ProviderConfigError(
            f"Model '{requested}' uses provider 'auto' with route(s) {', '.join(tried) or 'none'}, but none of "
            f"their API keys is set ({key_names}). Set one in your shell or in .env. "
            f"Refusing to fall back to the simulator."
        )

    @staticmethod
    def _read_timeout(model_config: Dict[str, Any]) -> float:
        raw = model_config.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS)
        try:
            timeout = float(raw)
        except (TypeError, ValueError):
            timeout = 0.0
        if timeout <= 0:
            raise ProviderConfigError(
                f"Model '{model_config.get('id')}' has invalid timeout_seconds={raw!r}; it must be a positive number."
            )
        return timeout

    @staticmethod
    def _read_min_interval(model_config: Dict[str, Any], provider: str) -> float:
        """Seconds to keep between request starts. Keyed providers default to 20 requests/minute."""
        raw = model_config.get("requests_per_minute")
        if raw is None:
            raw = DEFAULT_REQUESTS_PER_MINUTE if provider in PROVIDER_SPECS else None
        if raw is None:
            return 0.0
        try:
            rpm = float(raw) if not isinstance(raw, bool) else 0.0
        except (TypeError, ValueError):
            rpm = 0.0
        if rpm <= 0:
            raise ProviderConfigError(
                f"Model '{model_config.get('id')}' has invalid requests_per_minute={raw!r}; it must be a positive number."
            )
        return 60.0 / rpm

    # Time goes through these two methods so tests can run without really waiting.
    def _sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    def _now(self) -> float:
        return time.monotonic()

    def _throttle(self) -> None:
        if not self.min_interval:
            return
        now = self._now()
        start = max(now, self._next_slot)
        if start > now:
            self._sleep(start - now)
        self._next_slot = start + self.min_interval

    @staticmethod
    def _retry_after_seconds(resp: Any) -> Optional[float]:
        """The Retry-After header as seconds (it may be a number of seconds or an HTTP date), or None."""
        try:
            items = list(resp.headers.items())
        except Exception:
            return None
        raw = next((v for k, v in items if str(k).lower() == "retry-after"), None)
        if not isinstance(raw, (str, int, float)) or isinstance(raw, bool):
            return None
        text = str(raw).strip()
        try:
            value = float(text)
            return value if value >= 0 else None
        except ValueError:
            pass
        try:
            when = parsedate_to_datetime(text)
            if when.tzinfo is None:
                when = when.replace(tzinfo=timezone.utc)
            return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError):
            return None

    def _post_with_retries(self, url: str, **kwargs: Any) -> Tuple[Any, int, float, List[str]]:
        """POST, retrying only HTTP 429 and 5xx (up to MAX_RETRIES times, exponential backoff, Retry-After honoured).

        Returns (response, attempts, seconds taken by the last attempt, attempt_log). The attempt_log has one entry per
        request, in order: the HTTP status ("503", "200"), "read_timeout", or the name of a network error class.
        The response may still be an error status when retries ran out or the status is not retryable; other 4xx errors
        are never retried. A read timeout is retried once; other network errors (connection errors, connect timeouts)
        are not retried: they raise, carrying `attempts` and `attempt_log`.
        """
        attempts = 0
        timeout_retries = 0   # read-timeout retries and 429/5xx retries have separate budgets
        http_retries = 0
        log: List[str] = []
        while True:
            attempts += 1
            self._throttle()
            started = self._now()
            try:
                resp = requests.post(url, **kwargs)
            except requests.exceptions.ReadTimeout as e:
                log.append("read_timeout")
                if timeout_retries >= MAX_TIMEOUT_RETRIES:
                    err = RuntimeError(f"read timed out after {attempts} attempt(s) (limit {self.timeout:.0f}s each)")
                    err.attempts = attempts  # type: ignore[attr-defined]
                    err.attempt_log = list(log)  # type: ignore[attr-defined]
                    raise err from e
                timeout_retries += 1
                self._sleep(BACKOFF_BASE_SECONDS)
                continue
            except requests.exceptions.RequestException as e:
                log.append(type(e).__name__)
                e.attempts = attempts  # type: ignore[attr-defined]
                e.attempt_log = list(log)  # type: ignore[attr-defined]
                raise
            elapsed = self._now() - started
            status = resp.status_code
            log.append(str(status))
            retryable = status == 429 or (isinstance(status, int) and 500 <= status < 600)
            if not retryable or http_retries >= MAX_RETRIES:
                return resp, attempts, elapsed, log
            wait = BACKOFF_BASE_SECONDS * (2 ** http_retries)
            http_retries += 1
            retry_after = self._retry_after_seconds(resp)
            if retry_after is not None:
                if retry_after > MAX_RETRY_WAIT_SECONDS:
                    err = RuntimeError(
                        f"HTTP {status}; the server asked to wait {retry_after:.0f}s (Retry-After), more than the "
                        f"{MAX_RETRY_WAIT_SECONDS:.0f}s maximum; gave up after {attempts} attempt(s)"
                    )
                    err.attempts = attempts  # type: ignore[attr-defined]
                    err.http_status = status  # type: ignore[attr-defined]
                    err.attempt_log = list(log)  # type: ignore[attr-defined]
                    raise err
                wait = max(wait, retry_after)
            self._sleep(wait)

    def _check_endpoint(self, spec: Dict[str, Any]) -> None:
        parsed = urlparse(self.api_base)
        host = (parsed.hostname or "").lower()
        if (
            parsed.scheme != "https"
            or host not in spec["hosts"]
            or parsed.username is not None
            or parsed.port not in (None, 443)
        ):
            raise ProviderConfigError(
                f"Model '{self.model_id}' (provider '{self.provider}') has api_base '{self.api_base}', which is not an "
                f"https endpoint on {', '.join(spec['hosts'])}. Refusing to send {self.api_key_env} to another host."
            )

    def list_model_entries(self) -> List[Dict[str, Any]]:
        """The provider's own description of each model it serves (GET {api_base}/models), under the same key and
        host rules. This is one request; it is not retried."""
        if self.provider not in self.KEYED_PROVIDERS:
            raise ProviderConfigError(f"list_models is only available for {', '.join(self.KEYED_PROVIDERS)}.")
        headers = {"Authorization": f"Bearer {self.api_key}"}
        resp = requests.get(f"{self.api_base}/models", headers=headers, timeout=self.timeout, allow_redirects=False)
        if 300 <= resp.status_code < 400:
            raise RuntimeError(f"unexpected HTTP {resp.status_code} redirect; not following it")
        if resp.status_code >= 400:
            raise RuntimeError(f"model list request failed with HTTP {resp.status_code}")
        return [m for m in resp.json().get("data", []) if isinstance(m, dict) and "id" in m]

    def list_models(self) -> List[str]:
        """IDs of the models the provider currently serves."""
        return sorted(m["id"] for m in self.list_model_entries())

    @property
    def run_mode(self) -> str:
        """'simulated' when responses come from the mock engine, 'live' when a real provider serves them."""
        return "simulated" if self.provider == "mock" else "live"

    def generate(self, prompt: str, benchmark_id: str = "general", ground_truth: str = "", item_meta: Dict[str, Any] = None) -> Dict[str, Any]:
        # 1. Check Interceptor Cache (seed only affects the simulator, so only key on it there)
        cache_params = {**self.params, "seed": self.seed} if self.provider == "mock" else self.params
        cached = self.interceptor.get_cached(self._cache_model_id, prompt, cache_params)
        if cached:
            return cached

        # 2. Provider Routing
        if self.provider == "mock":
            res = MockModelEngine.generate_response(
                model_id=self.model_id,
                prompt=prompt,
                benchmark_id=benchmark_id,
                ground_truth=ground_truth,
                item_meta=item_meta,
                seed=self.seed
            )
        elif self.provider in self.KEYED_PROVIDERS:
            res = self._call_openai_compatible(prompt)
        else:
            res = self._call_ollama(prompt)

        # 3. Store in Interceptor Cache (failed calls are not cached, so a retry can succeed)
        if not res.get("error"):
            self.interceptor.set_cached(self._cache_model_id, prompt, cache_params, res)
        return res

    def _call_openai_compatible(self, prompt: str) -> Dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.params.get("temperature", 0.1),
            "max_tokens": self.params.get("max_tokens", 1024),
            "top_p": self.params.get("top_p", 0.95)
        }
        url = f"{self.api_base}/chat/completions"
        start_t = self._now()
        attempts = 1
        log: List[str] = []
        try:
            # Never follow redirects: the Authorization header must not leave the provider's own host.
            resp, attempts, latency, log = self._post_with_retries(
                url, headers=headers, json=payload, timeout=self.timeout, allow_redirects=False
            )
            self._raise_for_failed_status(resp, attempts, log)
            data = resp.json()
            choice = data["choices"][0]
            message = choice.get("message") or {}
            text = message.get("content") or ""
            reasoning = message.get("reasoning_content") or message.get("reasoning") or None
            usage = data.get("usage") or {}
            p_tokens = usage.get("prompt_tokens", len(prompt.split()) + 20)
            c_tokens = usage.get("completion_tokens", len(text.split()) + 10)
            cost_in = self.config.get("cost_per_1m_input", 0.5)
            cost_out = self.config.get("cost_per_1m_output", 0.5)
            cost = (p_tokens / 1e6 * cost_in) + (c_tokens / 1e6 * cost_out)
            return {
                "text": text,
                "reasoning_text": reasoning,
                "finish_reason": choice.get("finish_reason"),
                "latency_sec": round(latency, 3),
                "attempts": attempts,
                "attempt_log": log,
                "prompt_tokens": p_tokens,
                "completion_tokens": c_tokens,
                "total_tokens": p_tokens + c_tokens,
                "cost": round(cost, 6)
            }
        except Exception as e:
            return {
                "text": f"API Error: {e}",
                "latency_sec": round(self._now() - start_t, 3),
                "attempts": getattr(e, "attempts", attempts),
                "attempt_log": getattr(e, "attempt_log", log),
                "http_status": getattr(e, "http_status", None),  # None for timeouts and other network errors
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost": 0.0,
                "error": str(e)
            }

    @staticmethod
    def _raise_for_failed_status(resp: Any, attempts: int, log: Optional[List[str]] = None) -> None:
        """Raise for a failed reply, tagging the exception with `http_status`, `attempts` and `attempt_log`."""
        status = resp.status_code
        log = list(log or [])
        if 300 <= status < 400:
            err = RuntimeError(f"unexpected HTTP {status} redirect; not following it")
            err.http_status = status  # type: ignore[attr-defined]
            err.attempt_log = log  # type: ignore[attr-defined]
            raise err
        if status == 429 or 500 <= status < 600:
            err = RuntimeError(f"HTTP {status} after {attempts} attempt(s), {attempts - 1} retried")
            err.attempts = attempts  # type: ignore[attr-defined]
            err.http_status = status  # type: ignore[attr-defined]
            err.attempt_log = log  # type: ignore[attr-defined]
            raise err
        try:
            resp.raise_for_status()  # any other 4xx: raised at once, never retried
        except requests.exceptions.HTTPError as e:
            e.http_status = status  # type: ignore[attr-defined]
            e.attempt_log = log  # type: ignore[attr-defined]
            raise

    def _call_ollama(self, prompt: str) -> Dict[str, Any]:
        url = f"{self.api_base}/api/generate"
        payload = {"model": self.model_id, "prompt": prompt, "stream": False}
        start_t = self._now()
        attempts = 1
        log: List[str] = []
        try:
            resp, attempts, latency, log = self._post_with_retries(url, json=payload, timeout=self.timeout)
            self._raise_for_failed_status(resp, attempts, log)
            data = resp.json()
            text = data.get("response", "")
            return {
                "text": text,
                "reasoning_text": data.get("thinking") or None,
                "finish_reason": data.get("done_reason"),
                "latency_sec": round(latency, 3),
                "attempts": attempts,
                "attempt_log": log,
                "prompt_tokens": data.get("prompt_eval_count", len(prompt.split())),
                "completion_tokens": data.get("eval_count", len(text.split())),
                "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
                "cost": 0.0
            }
        except Exception as e:
            return {"text": f"Ollama Error: {e}", "latency_sec": round(self._now() - start_t, 3),
                    "attempts": getattr(e, "attempts", attempts), "attempt_log": getattr(e, "attempt_log", log),
                    "http_status": getattr(e, "http_status", None), "error": str(e), "total_tokens": 0, "cost": 0.0}
