from __future__ import annotations

from types import SimpleNamespace

import pytest

from hojichar.async_filters.openai import AsyncChatAPI
from hojichar.core.models import Document


# Disable actual API health checks
@pytest.fixture(autouse=True)
def disable_api_check(monkeypatch):
    monkeypatch.setattr(AsyncChatAPI, "check_api_alive", lambda self, endpoint_url, model_id: None)


# Helper to create a dummy ChatCompletion-like response
class DummyMessage:
    def __init__(self, content):
        self.content = content


class DummyChoice:
    def __init__(self, message):
        self.message = message


class DummyChatCompletion:
    def __init__(self, choices):
        self.choices = choices


@pytest.mark.asyncio
async def test_apply_success_default_message_generator(monkeypatch):
    # Prepare document
    doc = Document(text="Hello, world!")

    # Stub the OpenAI client
    async def fake_create(*args, **kwargs):
        msg = DummyMessage(content="Response content")
        choice = DummyChoice(message=msg)
        return DummyChatCompletion(choices=[choice])

    api = AsyncChatAPI(model_id="test-model")
    # Inject stub client
    api._openai_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))
    )

    result = await api.apply(doc)
    assert result is doc
    assert result.extras[api.output_key] == "Response content"


@pytest.mark.asyncio
async def test_apply_custom_message_generator(monkeypatch):
    # Custom generator that wraps text
    custom_gen = lambda doc: [  # noqa E731
        {"role": "system", "content": "sys"},
        {"role": "user", "content": doc.text},
    ]
    doc = Document(text="Check me")

    async def fake_create(*args, **kwargs):
        # Ensure messages passed correctly
        assert kwargs.get("messages") == custom_gen(doc)
        msg = DummyMessage(content="Ok")
        return DummyChatCompletion(choices=[DummyChoice(msg)])

    api = AsyncChatAPI(model_id="m", message_generator=custom_gen)
    api._openai_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))
    )

    out = await api.apply(doc)
    assert out.extras["llm_output"] == "Ok"


@pytest.mark.asyncio
async def test_apply_no_choices_raises():
    doc = Document(text="No choice")

    async def fake_create(*args, **kwargs):
        return DummyChatCompletion(choices=[])

    api = AsyncChatAPI(model_id="m")
    api._openai_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))
    )

    with pytest.raises(RuntimeError):
        await api.apply(doc)


@pytest.mark.asyncio
async def test_retry_mechanism(monkeypatch):
    doc = Document(text="Retry test")
    calls = []

    async def fake_create(*args, **kwargs):
        # Fail first two calls, succeed on third
        calls.append(1)
        if len(calls) < 3:
            raise Exception("Temporary failure")
        return DummyChatCompletion(choices=[DummyChoice(DummyMessage("Done"))])

    api = AsyncChatAPI(model_id="retry-model", retry_count=5)
    api._openai_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))
    )

    result = await api.apply(doc)
    assert result.extras["llm_output"] == "Done"
    assert len(calls) == 3


@pytest.mark.asyncio
async def test_custom_output_key(monkeypatch):
    doc = Document(text="Key test")

    async def fake_create(*args, **kwargs):
        return DummyChatCompletion(choices=[DummyChoice(DummyMessage("X"))])

    api = AsyncChatAPI(model_id="m", output_key="custom_key")
    api._openai_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))
    )

    out = await api.apply(doc)
    assert "custom_key" in out.extras
    assert out.extras["custom_key"] == "X"
