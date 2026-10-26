import hashlib
import time
from typing import Dict, Any, Optional

class InterceptorProxy:
    def __init__(self, cache_enabled: bool = True):
        self.cache_enabled = cache_enabled
        self._cache: Dict[str, Dict[str, Any]] = {}

    def _get_cache_key(self, model_id: str, prompt: str, params: Dict[str, Any]) -> str:
        serialized = f"{model_id}|{prompt}|{sorted(params.items())}"
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def get_cached(self, model_id: str, prompt: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self.cache_enabled:
            return None
        k = self._get_cache_key(model_id, prompt, params)
        return self._cache.get(k)

    def set_cached(self, model_id: str, prompt: str, params: Dict[str, Any], result: Dict[str, Any]) -> None:
        if not self.cache_enabled:
            return
        k = self._get_cache_key(model_id, prompt, params)
        self._cache[k] = result
