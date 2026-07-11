from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


Message = dict[str, str]


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model: str
    base_url: str | None = None
    api_key_env: str | None = None
    temperature: float = 0.0
    max_tokens: int = 2048
    timeout: float = 120.0
    retries: int = 2
    json_mode: bool = False
    dimensions: int | None = None


class LLMClient:
    def __init__(self, config: LLMConfig):
        provider = config.provider.strip().lower()
        if provider not in {"openai-compatible", "ollama"}:
            raise ValueError("provider must be 'openai-compatible' or 'ollama'")
        self.config = LLMConfig(
            provider=provider,
            model=config.model,
            base_url=config.base_url,
            api_key_env=config.api_key_env,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout=config.timeout,
            retries=config.retries,
            json_mode=config.json_mode,
            dimensions=config.dimensions,
        )

    def chat(self, messages: list[Message], *, expect_json: bool = False) -> str:
        if self.config.provider == "ollama":
            return self._chat_ollama(messages)
        return self._chat_openai_compatible(messages, expect_json=expect_json)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if self.config.provider != "openai-compatible":
            raise ValueError("Embedding backend must be openai-compatible")
        return self._embed_openai_compatible(texts)

    def metadata(self) -> dict[str, Any]:
        return {
            "provider": self.config.provider,
            "model": self.config.model,
            "base_url": self.config.base_url,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "json_mode": self.config.json_mode,
            "dimensions": self.config.dimensions,
        }


    def _embed_openai_compatible(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        url = _openai_embeddings_url(self.config.base_url)
        payload: dict[str, Any] = {
            "model": self.config.model,
            "input": texts,
        }
        if self.config.dimensions is not None:
            payload["dimensions"] = self.config.dimensions
        headers = {"Content-Type": "application/json"}
        api_key = _api_key(self.config.api_key_env)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        data = _request_json(url, payload, headers, self.config.timeout, self.config.retries)
        try:
            rows = data["data"]
            if not isinstance(rows, list):
                raise TypeError("data must be a list")
            ordered = sorted(rows, key=lambda row: int(row.get("index", 0)))
            embeddings = [row["embedding"] for row in ordered]
            if len(embeddings) != len(texts):
                raise RuntimeError(f"Expected {len(texts)} embeddings, got {len(embeddings)}")
            return [[float(value) for value in embedding] for embedding in embeddings]
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"Unexpected OpenAI-compatible embeddings response shape: {data!r}") from exc

    def _chat_openai_compatible(self, messages: list[Message], *, expect_json: bool) -> str:
        url = _openai_chat_url(self.config.base_url)
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if expect_json and self.config.json_mode:
            payload["response_format"] = {"type": "json_object"}
        headers = {"Content-Type": "application/json"}
        api_key = _api_key(self.config.api_key_env)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        data = _request_json(url, payload, headers, self.config.timeout, self.config.retries)
        try:
            return str(data["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected OpenAI-compatible response shape: {data!r}") from exc

    def _chat_ollama(self, messages: list[Message]) -> str:
        base_url = (self.config.base_url or "http://localhost:11434").rstrip("/")
        payload = {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            },
        }
        data = _request_json(
            f"{base_url}/api/chat",
            payload,
            {"Content-Type": "application/json"},
            self.config.timeout,
            self.config.retries,
        )
        try:
            return str(data["message"]["content"])
        except (KeyError, TypeError) as exc:
            raise RuntimeError(f"Unexpected Ollama response shape: {data!r}") from exc


def _api_key(env_name: str | None) -> str | None:
    if not env_name:
        return None
    value = os.environ.get(env_name)
    return value.strip() if value else None


def _openai_chat_url(base_url: str | None) -> str:
    base = (base_url or "http://localhost:8000/v1").rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def _openai_embeddings_url(base_url: str | None) -> str:
    base = (base_url or "https://api.openai.com/v1").rstrip("/")
    if base.endswith("/embeddings"):
        return base
    if base.endswith("/v1"):
        return f"{base}/embeddings"
    return f"{base}/v1/embeddings"


def _request_json(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
    retries: int,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read().decode("utf-8")
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                raise RuntimeError(f"Expected JSON object response from {url}, got {type(parsed).__name__}")
            return parsed
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"HTTP {exc.code} from {url}: {details}")
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, RuntimeError) as exc:
            last_error = exc
        if attempt < retries:
            time.sleep(min(2**attempt, 8))
    raise RuntimeError(f"LLM request failed after {retries + 1} attempts: {last_error}") from last_error

