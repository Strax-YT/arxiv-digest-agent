"""One narrow LLM interface, four free backends, plus a mock for tests.

Deliberately built on raw HTTP + `requests` instead of four vendor SDKs: the
surface we need is "send messages, get text back", and vendor SDKs churn.

All providers share:
  * bounded retries with exponential backoff, honouring `Retry-After` on 429
  * a `complete_json` helper that strips code fences, salvages unbalanced
    braces, and does exactly one repair round-trip before giving up
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from typing import Any

import requests

from ..config import API_KEY_ENV, Settings

log = logging.getLogger(__name__)

RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


class LLMError(RuntimeError):
    pass


class LLMUnavailable(LLMError):
    """Provider unreachable / not configured — callers may degrade instead of dying."""


# --------------------------------------------------------------------- #
# base
# --------------------------------------------------------------------- #
class BaseLLM:
    name = "base"

    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self.calls = 0

    # -- subclasses implement this -- #
    def _request(self, system: str, user: str, json_mode: bool, temperature: float) -> str:
        raise NotImplementedError

    def complete(
        self,
        user: str,
        system: str = "You are a precise research assistant.",
        *,
        json_mode: bool = False,
        temperature: float | None = None,
    ) -> str:
        temp = self.s.temperature if temperature is None else temperature
        last: Exception | None = None
        for attempt in range(1, self.s.llm_retries + 1):
            try:
                self.calls += 1
                return self._request(system, user, json_mode, temp)
            except LLMUnavailable:
                raise
            except Exception as exc:  # noqa: BLE001
                last = exc
                if attempt == self.s.llm_retries:
                    break
                delay = min(30.0, (2**attempt) + random.uniform(0, 0.8))
                retry_after = getattr(exc, "retry_after", None)
                if retry_after:
                    delay = min(60.0, float(retry_after))
                log.warning("%s call failed (%s); retrying in %.1fs", self.name, exc, delay)
                time.sleep(delay)
        raise LLMError(f"{self.name} failed after {self.s.llm_retries} attempts: {last}")

    def complete_json(
        self,
        user: str,
        system: str = "You reply with JSON only.",
        *,
        temperature: float | None = None,
    ) -> Any:
        raw = self.complete(user, system, json_mode=True, temperature=temperature)
        try:
            return parse_json_loose(raw)
        except ValueError as exc:
            log.warning("JSON parse failed (%s); attempting one repair round-trip", exc)
            repair = (
                "The following text was supposed to be a single valid JSON document but "
                f"could not be parsed ({exc}). Return ONLY the corrected JSON, no prose, "
                "no markdown fences.\n\n<<<\n" + raw[:8000] + "\n>>>"
            )
            fixed = self.complete(repair, "You repair malformed JSON. Output JSON only.", json_mode=True)
            return parse_json_loose(fixed)


def parse_json_loose(text: str) -> Any:
    """Tolerant JSON extraction: fences, preamble, trailing commas, truncation."""
    if text is None:
        raise ValueError("empty response")
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()
    start = min([i for i in (cleaned.find("{"), cleaned.find("[")) if i != -1], default=-1)
    if start == -1:
        raise ValueError("no JSON object found in response")
    cleaned = cleaned[start:]
    for candidate in (cleaned, _balance(cleaned), re.sub(r",\s*([}\]])", r"\1", _balance(cleaned))):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise ValueError("could not parse JSON from model output")


def _balance(text: str) -> str:
    """Close brackets a truncated generation left open (drops a dangling string)."""
    stack: list[str] = []
    in_string = False
    escape = False
    last_safe = len(text)
    for i, ch in enumerate(text):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]" and stack:
            stack.pop()
            if not stack:
                last_safe = i + 1
    if in_string:
        text = text[: text.rfind('"')] if '"' in text else text
        return text + '"' + "".join(reversed(stack))
    if not stack:
        return text[:last_safe]
    return text + "".join(reversed(stack))


def _raise_for_status(resp: requests.Response, provider: str) -> None:
    if resp.status_code < 400:
        return
    err = LLMError(f"{provider} HTTP {resp.status_code}: {resp.text[:300]}")
    if resp.status_code in RETRYABLE_STATUS:
        err.retry_after = resp.headers.get("Retry-After")  # type: ignore[attr-defined]
    raise err


# --------------------------------------------------------------------- #
# providers
# --------------------------------------------------------------------- #
class OpenAICompatLLM(BaseLLM):
    """Groq and OpenRouter both speak the OpenAI chat-completions dialect."""

    def __init__(self, settings: Settings, name: str) -> None:
        super().__init__(settings)
        self.name = name
        if not settings.api_key:
            raise LLMUnavailable(
                f"{name} needs {API_KEY_ENV.get(name, 'an API key')} in the environment "
                f"(free tier is fine). Or run with --provider ollama for a fully local setup."
            )

    def _request(self, system: str, user: str, json_mode: bool, temperature: float) -> str:
        headers = {"Authorization": f"Bearer {self.s.api_key}", "Content-Type": "application/json"}
        if self.name == "openrouter":
            headers["HTTP-Referer"] = "https://github.com/local/arxiv-digest-agent"
            headers["X-Title"] = "arxiv-digest-agent"
        body: dict[str, Any] = {
            "model": self.s.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": temperature,
            "max_tokens": self.s.max_tokens,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        resp = requests.post(
            f"{self.s.base_url}/chat/completions", headers=headers, json=body, timeout=self.s.llm_timeout
        )
        _raise_for_status(resp, self.name)
        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError) as exc:
            raise LLMError(f"{self.name}: unexpected response shape: {str(data)[:300]}") from exc


class OllamaLLM(BaseLLM):
    name = "ollama"

    def _request(self, system: str, user: str, json_mode: bool, temperature: float) -> str:
        body: dict[str, Any] = {
            "model": self.s.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            "options": {"temperature": temperature, "num_predict": self.s.max_tokens},
        }
        if json_mode:
            body["format"] = "json"
        try:
            resp = requests.post(f"{self.s.base_url}/api/chat", json=body, timeout=self.s.llm_timeout)
        except requests.ConnectionError as exc:
            raise LLMUnavailable(
                f"Cannot reach Ollama at {self.s.base_url}. Start it with `ollama serve` and "
                f"`ollama pull {self.s.model}`, or pick a hosted free tier with --provider groq."
            ) from exc
        if resp.status_code == 404:
            raise LLMUnavailable(f"Ollama has no model {self.s.model!r}. Run: ollama pull {self.s.model}")
        _raise_for_status(resp, self.name)
        return resp.json().get("message", {}).get("content", "")


class GeminiLLM(BaseLLM):
    name = "gemini"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        if not settings.api_key:
            raise LLMUnavailable("gemini needs GOOGLE_API_KEY (AI Studio free tier).")

    def _request(self, system: str, user: str, json_mode: bool, temperature: float) -> str:
        url = f"{self.s.base_url}/models/{self.s.model}:generateContent"
        gen: dict[str, Any] = {"temperature": temperature, "maxOutputTokens": self.s.max_tokens}
        if json_mode:
            gen["responseMimeType"] = "application/json"
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": gen,
        }
        resp = requests.post(
            url,
            params={"key": self.s.api_key},
            json=body,
            timeout=self.s.llm_timeout,
            headers={"Content-Type": "application/json"},
        )
        _raise_for_status(resp, self.name)
        data = resp.json()
        candidates = data.get("candidates") or []
        if not candidates:
            raise LLMError(f"gemini returned no candidates: {str(data)[:300]}")
        parts = candidates[0].get("content", {}).get("parts", [])
        return "".join(p.get("text", "") for p in parts)


class MockLLM(BaseLLM):
    """Deterministic stand-in so the whole graph is testable with no network."""

    name = "mock"

    def __init__(self, settings: Settings, responses: dict[str, str] | None = None) -> None:
        super().__init__(settings)
        self.responses = responses or {}
        self.prompts: list[str] = []

    def _request(self, system: str, user: str, json_mode: bool, temperature: float) -> str:
        self.prompts.append(user)
        for key, value in self.responses.items():
            if key in user or key in system:
                return value
        return "{}" if json_mode else "MOCK"


# --------------------------------------------------------------------- #
def build_llm(settings: Settings) -> BaseLLM:
    provider = settings.provider.lower()
    if provider == "ollama":
        return OllamaLLM(settings)
    if provider in {"groq", "openrouter"}:
        return OpenAICompatLLM(settings, provider)
    if provider == "gemini":
        return GeminiLLM(settings)
    if provider == "mock":
        return MockLLM(settings)
    raise LLMError(f"unknown provider {provider!r}; expected ollama|groq|gemini|openrouter|mock")
