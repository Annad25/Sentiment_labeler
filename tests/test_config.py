"""Tests for model-capability resolution and request-param derivation.

The registry decides whether a model gets `reasoning_effort` (reasoning
models) or `temperature` (classic chat models). Getting this wrong produces
an API error at runtime, so it's worth pinning down.
"""
from sentiment_eval.config import EvalConfig, resolve_model


def test_known_reasoning_models():
    assert resolve_model("gpt-5-mini").reasoning is True
    assert resolve_model("gpt-5").reasoning is True
    assert resolve_model("o3-mini").reasoning is True


def test_known_chat_models():
    assert resolve_model("gpt-4o-mini").reasoning is False
    assert resolve_model("gpt-4.1").reasoning is False


def test_case_insensitive():
    assert resolve_model("GPT-5-MINI").reasoning is True


def test_snapshot_suffix_is_stripped():
    # Dated snapshot ids must resolve to their base model.
    assert resolve_model("gpt-5-mini-2025-08-07").reasoning is True
    assert resolve_model("gpt-4o-mini-2024-07-18").reasoning is False


def test_unknown_model_falls_back_to_heuristic():
    # Not in the registry, but the name pattern still tells us the family.
    assert resolve_model("gpt-5-turbo-experimental").reasoning is True
    assert resolve_model("o9").reasoning is True
    assert resolve_model("some-random-chat-model").reasoning is False


def test_request_overrides_reasoning_model():
    cfg = EvalConfig(model="gpt-5-mini")  # temperature=None, reasoning_effort="minimal"
    overrides = cfg.request_overrides()
    assert overrides == {"reasoning_effort": "minimal"}
    assert "temperature" not in overrides  # would error against a reasoning model


def test_request_overrides_chat_model_defaults_to_temp_zero():
    # Switching to a chat model without touching other fields must still
    # produce a valid request: temperature, not reasoning_effort.
    cfg = EvalConfig(model="gpt-4o-mini")
    overrides = cfg.request_overrides()
    assert overrides == {"temperature": 0.0}
    assert "reasoning_effort" not in overrides
