"""LLM provider chain with graceful failover.

The classifier tries providers in order. If one is unavailable (no API key) it
is skipped; if a call fails after retries, we fall through to the next provider.

Default chain:
  1. openai                  — OpenAI direct, primary model (e.g. gpt-5-mini)
  2. openrouter:mirror       — the *same* model via OpenRouter; survives an
                               OpenAI-side outage with identical behaviour
  3. openrouter:deepseek-nex — a *different* vendor's model via OpenRouter
                               (DeepSeek V3.1 Nex-N1); survives the primary
                               model being unavailable everywhere

OpenRouter is OpenAI-API-compatible, so the same AsyncOpenAI client works with
a different base_url + key.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from openai import AsyncOpenAI

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Cross-vendor last resort, reached only if both OpenAI direct and the
# OpenRouter mirror of the primary model fail. Not an OpenAI model, so it takes
# `temperature` (not `reasoning_effort`) and we don't assume structured-output
# support (the plain-completion + normalize() path handles it). Configurable —
# swap for any OpenRouter model id.
CROSS_VENDOR_FALLBACK_MODEL = "nex-agi/deepseek-v3.1-nex-n1"


@dataclass(frozen=True)
class Provider:
    name: str                       # short label, recorded in results
    model: str                      # provider-specific model id
    api_key_env: str                # env var holding this provider's key
    base_url: Optional[str] = None  # None => OpenAI default endpoint
    structured_output: bool = True  # whether to use the JSON-schema parse path

    @property
    def is_available(self) -> bool:
        return bool(os.environ.get(self.api_key_env))


def build_provider_chain(primary_model: str) -> list[Provider]:
    """Construct the ordered provider chain for a given primary model."""
    chain = [
        Provider("openai", primary_model, "OPENAI_API_KEY"),
        Provider(
            "openrouter:mirror",
            f"openai/{primary_model}",
            "OPENROUTER_API_KEY",
            base_url=OPENROUTER_BASE_URL,
        ),
        Provider(
            "openrouter:deepseek-nex",
            CROSS_VENDOR_FALLBACK_MODEL,
            "OPENROUTER_API_KEY",
            base_url=OPENROUTER_BASE_URL,
            structured_output=False,
        ),
    ]
    return chain


def available_providers(chain: list[Provider]) -> list[Provider]:
    """Keep only providers whose API key is present in the environment."""
    return [p for p in chain if p.is_available]


def make_client(provider: Provider) -> AsyncOpenAI:
    """Build an AsyncOpenAI client pointed at the provider's endpoint."""
    kwargs: dict = {"api_key": os.environ[provider.api_key_env]}
    if provider.base_url:
        kwargs["base_url"] = provider.base_url
        # Optional, recommended by OpenRouter for attribution; harmless elsewhere.
        kwargs["default_headers"] = {"X-Title": "sentiment-eval"}
    return AsyncOpenAI(**kwargs)
