from __future__ import annotations

import re
from typing import Literal, NamedTuple, Optional

from openai import AsyncOpenAI
from pydantic import BaseModel
from tenacity import (
    AsyncRetrying,
    stop_after_attempt,
    wait_random_exponential,
)

from .config import EvalConfig, request_overrides_for
from .prompts import load_prompt
from .providers import (
    Provider,
    available_providers,
    build_provider_chain,
    make_client,
)

UNPARSEABLE = "unparseable"
ParsedLabel = Literal["positive", "neutral", "negative", "unparseable"]


class Prediction(NamedTuple):
    """A single classification result, tagged with what produced it.

    `provider` is the chain tier that answered (e.g. 'openai'); `model` is the
    exact model id it used (e.g. 'gpt-5-mini'). Both are recorded per row so a
    reviewer can see whether any failover changed the model mid-run.
    """
    label: ParsedLabel
    raw: str
    provider: str
    model: str

# Loaded from prompts/system_prompt.txt (see sentiment_eval/prompts.py).
SYSTEM_PROMPT = load_prompt("system_prompt")


class SentimentResponse(BaseModel):
    """Schema for OpenAI structured-output. Constrains the model to the enum."""
    sentiment: Literal["positive", "neutral", "negative"]


_KEYWORD_RE = re.compile(r"\b(positive|neutral|negative)\b", re.IGNORECASE)


def normalize(raw: str) -> ParsedLabel:
    """Backstop parser for free-text outputs.

    Why this exists: not every provider supports structured output (the
    cross-vendor fallback uses a plain completion), and even structured calls
    can fall back to text. A junior eval that crashes on 'Positive!' would be
    the embarrassing failure mode.
    """
    if not raw:
        return UNPARSEABLE
    m = _KEYWORD_RE.search(raw)
    if not m:
        return UNPARSEABLE
    return m.group(1).lower()  # type: ignore[return-value]


class NoProvidersAvailable(RuntimeError):
    """Raised at construction if no provider has its API key set."""


class Classifier:
    """Classifies snippets, failing over across a chain of LLM providers."""

    def __init__(self, cfg: EvalConfig) -> None:
        self._cfg = cfg
        chain = available_providers(build_provider_chain(cfg.model))
        if not chain:
            raise NoProvidersAvailable(
                "No API keys found. Set OPENAI_API_KEY (and optionally "
                "OPENROUTER_API_KEY for fallback) in your environment / .env."
            )
        self._providers = chain
        self._clients: dict[str, AsyncOpenAI] = {p.name: make_client(p) for p in chain}

    @property
    def provider_names(self) -> list[str]:
        return [p.name for p in self._providers]

    async def classify(self, snippet: str) -> Prediction:
        """Classify one snippet, tagging the result with provider + model.

        Tries each provider in order; transient errors are retried within a
        provider, then we fail over to the next. Raises only if every provider
        is exhausted.
        """
        last_err: Optional[Exception] = None
        for provider in self._providers:
            try:
                return await self._classify_with(provider, snippet)
            except Exception as e:  # provider exhausted its retries — fail over
                last_err = e
                continue
        raise RuntimeError(
            f"all providers failed ({', '.join(self.provider_names)}): {last_err}"
        )

    async def _classify_with(self, provider: Provider, snippet: str) -> Prediction:
        client = self._clients[provider.name]
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": snippet},
        ]
        overrides = request_overrides_for(
            provider.model, self._cfg.temperature, self._cfg.reasoning_effort
        )

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self._cfg.max_retries),
            wait=wait_random_exponential(min=1, max=10),
            reraise=True,
        ):
            with attempt:
                if provider.structured_output:
                    label, raw = await self._call_structured(
                        client, provider.model, messages, overrides
                    )
                else:
                    label, raw = await self._call_plain(
                        client, provider.model, messages, overrides
                    )
                return Prediction(label, raw, provider.name, provider.model)
        raise RuntimeError("unreachable")  # tenacity reraises before this

    async def _call_structured(
        self, client: AsyncOpenAI, model: str, messages: list, overrides: dict
    ) -> tuple[ParsedLabel, str]:
        resp = await client.beta.chat.completions.parse(
            model=model,
            timeout=self._cfg.request_timeout_s,
            messages=messages,
            response_format=SentimentResponse,
            **overrides,
        )
        msg = resp.choices[0].message
        raw = msg.content or ""
        if msg.parsed is not None:
            return msg.parsed.sentiment, raw
        return normalize(raw), raw  # structured output refused → regex backstop

    async def _call_plain(
        self, client: AsyncOpenAI, model: str, messages: list, overrides: dict
    ) -> tuple[ParsedLabel, str]:
        # For providers without reliable structured-output support: ask for one
        # word and lean on normalize().
        resp = await client.chat.completions.create(
            model=model,
            timeout=self._cfg.request_timeout_s,
            messages=messages,
            **overrides,
        )
        raw = resp.choices[0].message.content or ""
        return normalize(raw), raw

    async def aclose(self) -> None:
        for client in self._clients.values():
            await client.close()
