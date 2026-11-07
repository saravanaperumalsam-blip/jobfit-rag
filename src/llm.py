"""Thin wrapper around any OpenAI-compatible chat API (Groq, OpenAI, Gemini, Ollama)."""
from __future__ import annotations

import re

from . import config

# Name of the model that produced the most recent reply (main model or a backup).
last_model: str | None = None


class LLMUnavailable(Exception):
    """No model could answer (rate limit, network problem, bad key, ...).
    The message is short and safe to show to the user."""


def is_configured() -> bool:
    if config.LLM_PROVIDER in ("", "none"):
        return False
    return config.LLM_PROVIDER == "ollama" or bool(config.LLM_API_KEY)


def describe() -> str:
    if not is_configured():
        return "No LLM configured (extractive mode)"
    _, model = _settings()
    return f"{config.LLM_PROVIDER} / {model}"


def _settings() -> tuple[str, str]:
    base_url, model = config.PROVIDER_DEFAULTS.get(config.LLM_PROVIDER, ("", ""))
    return config.LLM_BASE_URL or base_url, config.LLM_MODEL or model


def _models() -> list[str]:
    _, main = _settings()
    return [main] + [m for m in config.LLM_FALLBACK_MODELS if m != main]


def _short_reason(exc: Exception) -> str:
    """Turn an API error into one readable line (no tracebacks, no account ids)."""
    from openai import APIConnectionError, AuthenticationError, NotFoundError, RateLimitError
    text = str(exc)
    if isinstance(exc, RateLimitError):
        period = "daily" if "per day" in text.lower() else "per-minute"
        wait = re.search(r"try again in ([0-9hms.]+)", text)
        return f"{period} usage limit reached" + (f", try again in {wait.group(1)}" if wait else "")
    if isinstance(exc, AuthenticationError):
        return "the API key was rejected; check LLM_API_KEY in .env"
    if isinstance(exc, NotFoundError):
        return "model not available for this key"
    if isinstance(exc, APIConnectionError):
        return "could not reach the LLM service (check your internet connection)"
    return exc.__class__.__name__


def chat(system: str, user: str, temperature: float = 0.0) -> str:
    """Send one system + user message and return the reply text.

    Tries the main model, then each backup model if a model is rate-limited or not
    available. Raises LLMUnavailable with a short reason if none of them can answer."""
    global last_model
    from openai import (APIConnectionError, APIStatusError, AuthenticationError, NotFoundError,
                        OpenAI, RateLimitError)

    base_url, _ = _settings()
    client = OpenAI(base_url=base_url, api_key=config.LLM_API_KEY or "ollama", max_retries=1)
    reasons = []
    for model in _models():
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            )
        except (RateLimitError, NotFoundError) as exc:  # try the next model
            reasons.append(f"{model}: {_short_reason(exc)}")
            continue
        except (AuthenticationError, APIConnectionError) as exc:  # same for every model
            raise LLMUnavailable(_short_reason(exc)) from exc
        except APIStatusError as exc:
            reasons.append(f"{model}: {_short_reason(exc)}")
            continue
        last_model = model
        text = response.choices[0].message.content or ""
        # Some reasoning models (e.g. Qwen) include their thinking in <think> tags.
        return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    raise LLMUnavailable("; ".join(reasons) or "no model configured")
