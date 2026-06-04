"""Tests for the provider failover chain.

These pin down the two things that matter: the chain is built in the right
order with the right per-provider request semantics, and providers without an
API key are dropped (graceful degradation).
"""
from sentiment_eval.config import request_overrides_for
from sentiment_eval.providers import (
    available_providers,
    build_provider_chain,
)


def test_chain_order_and_models():
    chain = build_provider_chain("gpt-5-mini")
    assert [p.name for p in chain] == [
        "openai",
        "openrouter:mirror",
        "openrouter:deepseek-nex",
    ]
    # The mirror routes the *same* model through OpenRouter.
    assert chain[1].model == "openai/gpt-5-mini"
    # The cross-vendor backstop is a different vendor and skips structured output.
    assert chain[2].model.startswith("nex-agi/")
    assert chain[2].structured_output is False


def test_primary_model_is_passed_through():
    chain = build_provider_chain("gpt-4o-mini")
    assert chain[0].model == "gpt-4o-mini"
    assert chain[1].model == "openai/gpt-4o-mini"


def test_only_providers_with_keys_are_available(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    chain = build_provider_chain("gpt-5-mini")
    avail = available_providers(chain)
    assert [p.name for p in avail] == ["openai"]  # OpenRouter dropped


def test_all_providers_available_when_both_keys_set(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    avail = available_providers(build_provider_chain("gpt-5-mini"))
    assert len(avail) == 3


def test_overrides_track_the_provider_model():
    # Reasoning model (mirror) -> reasoning_effort; cross-vendor Claude -> temperature.
    mirror = request_overrides_for("openai/gpt-5-mini", None, "minimal")
    assert mirror == {"reasoning_effort": "minimal"}

    cross = request_overrides_for("nex-agi/deepseek-v3.1-nex-n1", None, "minimal")
    assert cross == {"temperature": 0.0}
