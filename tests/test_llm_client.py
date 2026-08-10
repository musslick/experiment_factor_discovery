from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.discovery.llm_client import LLMClient, LLMError, OllamaLLMClient


def _anthropic_client(content):
    client = LLMClient.__new__(LLMClient)
    client.model = "test-model"
    client.client = SimpleNamespace(
        messages=SimpleNamespace(create=Mock(return_value=SimpleNamespace(content=content)))
    )
    return client


def _ollama_client(model, chat):
    client = OllamaLLMClient.__new__(OllamaLLMClient)
    client.model = model
    client._client = SimpleNamespace(chat=chat)
    return client


def _response(content):
    return SimpleNamespace(message=SimpleNamespace(content=content))


def test_anthropic_concatenates_text_blocks():
    client = _anthropic_client(
        [
            SimpleNamespace(type="text", text="first"),
            SimpleNamespace(type="thinking", thinking="hidden"),
            SimpleNamespace(type="text", text="second"),
        ]
    )

    assert client.complete("system", "user") == "first\nsecond"


def test_anthropic_raises_when_response_has_no_text():
    client = _anthropic_client([SimpleNamespace(type="thinking", thinking="hidden")])

    with pytest.raises(LLMError, match="no text block"):
        client.complete("system", "user")


def test_ollama_disables_thinking_for_normal_models():
    chat = Mock(return_value=_response("OK"))
    client = _ollama_client("qwen3.5:9b", chat)

    assert client.complete("system", "user") == "OK"
    assert chat.call_args.kwargs["think"] is False


def test_ollama_uses_low_thinking_for_gpt_oss():
    chat = Mock(return_value=_response("OK"))
    client = _ollama_client("gpt-oss:20b", chat)

    assert client.complete("system", "user") == "OK"
    assert chat.call_args.kwargs["think"] == "low"


def test_ollama_retries_blank_content_three_times(monkeypatch):
    chat = Mock(side_effect=[_response(None), _response("  "), _response("")])
    client = _ollama_client("test-model", chat)
    monkeypatch.setattr("src.discovery.llm_client.time.sleep", lambda _: None)

    with pytest.raises(LLMError, match="no content"):
        client.complete("system", "user")
    assert chat.call_count == 3


def test_ollama_does_not_retry_errors():
    chat = Mock(side_effect=RuntimeError("permanent failure"))
    client = _ollama_client("test-model", chat)

    with pytest.raises(LLMError, match="permanent failure"):
        client.complete("system", "user")
    assert chat.call_count == 1
