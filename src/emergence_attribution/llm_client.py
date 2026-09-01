"""The project's only OpenAI-compatible API boundary."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI


class LLMConfigurationError(RuntimeError):
    pass


@dataclass(frozen=True)
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    model: str


_REQUIRED = {
    "base_url",
    "api_key",
    "model",
    "temperature",
    "max_tokens",
    "timeout",
    "max_retries",
}


def load_llm_config(path: Path, *, require_key: bool = True) -> dict[str, Any]:
    if not path.is_file():
        raise LLMConfigurationError(f"LLM configuration file does not exist: {path}")
    config = json.loads(path.read_text(encoding="utf-8"))
    missing = sorted(_REQUIRED - set(config))
    if missing:
        raise LLMConfigurationError(f"LLM configuration fields are missing: {missing}")
    if not str(config["model"]).strip():
        raise LLMConfigurationError("model must be configured")
    if require_key:
        key = str(config["api_key"]).strip()
        if not key or key.lower() in {"replace_me", "your_api_key", "none", "null"}:
            raise LLMConfigurationError(
                "formal semantic generation requires a real API key; no fallback response is permitted"
            )
    return config


def redacted_llm_config(config: dict[str, Any]) -> dict[str, Any]:
    result = {key: value for key, value in config.items() if key != "api_key"}
    result["api_key"] = "***REDACTED***" if str(config.get("api_key", "")).strip() else ""
    result["api_key_present"] = bool(str(config.get("api_key", "")).strip())
    return result


def public_config_hash(config: dict[str, Any]) -> str:
    public = {key: value for key, value in config.items() if key != "api_key"}
    payload = json.dumps(public, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class OpenAICompatibleClient:
    def __init__(self, config: dict[str, Any]):
        self.config = dict(config)
        kwargs: dict[str, Any] = {
            "api_key": config["api_key"],
            "timeout": float(config["timeout"]),
            "max_retries": int(config["max_retries"]),
        }
        if str(config.get("base_url", "")).strip():
            kwargs["base_url"] = str(config["base_url"]).strip()
        self._client = OpenAI(**kwargs)

    def complete_json(self, system: str, user: str) -> LLMResponse:
        response = self._client.chat.completions.create(
            model=str(self.config["model"]),
            temperature=float(self.config["temperature"]),
            max_tokens=int(self.config["max_tokens"]),
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        text = response.choices[0].message.content or ""
        usage = response.usage
        return LLMResponse(
            text=text,
            input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            model=str(getattr(response, "model", self.config["model"])),
        )

