from __future__ import annotations

from types import SimpleNamespace

import pytest

import hojichar.async_filters.openai as openai_module
from hojichar.async_filters.openai import AsyncChatAPI
from hojichar.core.models import Document


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


def test_retry_count_is_delegated_to_openai_sdk():
    api = AsyncChatAPI(model_id="retry-model", retry_count=5)
    assert api._openai_client.max_retries == 4


def test_retry_count_must_include_initial_attempt():
    with pytest.raises(ValueError, match="retry_count must be at least 1"):
        AsyncChatAPI(model_id="retry-model", retry_count=0)


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


def test_constructor_does_not_check_api(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("constructor must not perform an API check")

    monkeypatch.setattr(AsyncChatAPI, "check_api_alive", fail_if_called)
    AsyncChatAPI(model_id="m")


def test_explicit_endpoint_takes_precedence_over_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_ENDPOINT_URL", "https://environment.example/v1")
    api = AsyncChatAPI(
        model_id="m",
        openai_endpoint_url="https://explicit.example/v1",
    )
    assert str(api._openai_client.base_url) == "https://explicit.example/v1/"


@pytest.mark.asyncio
async def test_validate_api_success():
    async def fake_list():
        return SimpleNamespace(data=[SimpleNamespace(id="m")])

    api = AsyncChatAPI(model_id="m")
    api._openai_client = SimpleNamespace(models=SimpleNamespace(list=fake_list))
    await api.validate_api()


@pytest.mark.asyncio
async def test_validate_api_model_not_found():
    async def fake_list():
        return SimpleNamespace(data=[SimpleNamespace(id="another-model")])

    api = AsyncChatAPI(model_id="m")
    api._openai_client = SimpleNamespace(models=SimpleNamespace(list=fake_list))
    with pytest.raises(ValueError, match="Model 'm' not found"):
        await api.validate_api()


def test_check_api_alive_is_deprecated(monkeypatch):
    class DummyResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"data": [{"id": "m"}]}

    class DummyClient:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get(self, url):
            return DummyResponse()

    monkeypatch.setattr(openai_module.httpx, "Client", DummyClient)
    api = AsyncChatAPI(model_id="m")
    with pytest.warns(DeprecationWarning, match="validate_api"):
        api.check_api_alive(endpoint_url="https://example.test/v1", model_id="m")
