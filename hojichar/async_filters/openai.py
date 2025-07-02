from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Callable, Iterable

import httpx
from openai import AsyncOpenAI, DefaultAioHttpClient
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessageParam,
)
from tenacity import AsyncRetrying, before_sleep_log, stop_after_attempt, wait_random_exponential

from hojichar import AsyncFilter, Document

DEFAULT_OPENAI_ENDPOINT_URL = "https://api.openai.com/v1"


class AsyncChatAPI(AsyncFilter):
    def __init__(
        self,
        model_id: str,
        *,
        output_key: str = "llm_output",
        max_concurrent_requests: int = 128,
        openai_endpoint_url: str | None = None,
        openai_api_key: str | None = None,
        message_generator: Callable[[Document], Iterable[ChatCompletionMessageParam]]
        | Callable[[Document], list[dict[str, str]]] = lambda doc: [
            {"role": "user", "content": doc.text}
        ],
        timeout: httpx.Timeout | None = None,
        retry_count: int = 5,
        api_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        """
        This filter sends requests to an OpenAI-compatible API and processes the responses asynchronously.
        Set environment variables:
        - `OPENAI_ENDPOINT_URL` to the OpenAI-compatible API endpoint URL. e.g. `https://your_openai_api/v1`.
          - Your API must have the Chat Completion endpoint implemented in the `/v1/chat/completions` path with specified `model_id`
          - By default, it will use the OpenAI API endpoint `https://api.openai.com/v1`.
        - `OPENAI_API_KEY` to the API key for authentication.

        Args:
            model_id (str): The ID of the OpenAI model or your custom model deployed on the API. `<endpoint_url>/models` must
            output_key (str): The key under which the API response will be stored in the `Document.extras`
            max_concurrent_requests (int): The maximum number of concurrent requests to the API. If set the larger value, the throughput may increase.
            openai_endpoint_url (str | None): The URL of the OpenAI-compatible API endpoint. If not set, it will use the `OPENAI_ENDPOINT_URL` environment variable.
            message_generator: A callable that generates the messages to be sent to the API from Document class. The generated value is inserted into the `messages` parameter of the OpenAI API request. e.g. `lambda doc: [{"system": "You are awesome assistant."}, {"role": "user", "content": doc.text}]`.
            timeout (httpx.Timeout | None): The timeout settings for the HTTP client. If not set, it will use httpx.Timeout(connect=5.0, write=30, read=180, pool=30)
            api_kwargs (dict[str, Any] | None): Additional keyword arguments to pass to the OpenAI API request.
        """
        super().__init__(**kwargs)

        self.model_id = model_id
        endpoint_url = os.getenv("OPENAI_ENDPOINT_URL", openai_endpoint_url)
        if endpoint_url is None:
            msg = (
                "OPENAI_ENDPOINT_URL environment variable is not set. "
                "Please set it to the OpenAI API endpoint URL., e.g. 'https://your_openai_api/v1'"
            )
            self.logger.warning(msg)
            endpoint_url = DEFAULT_OPENAI_ENDPOINT_URL

        self.api_kwargs = api_kwargs or {}
        self.output_key = output_key
        self.message_generator = message_generator
        self._retry_count = retry_count

        self._semaphore = asyncio.Semaphore(max_concurrent_requests)

        self._http_client = DefaultAioHttpClient(
            limits=httpx.Limits(
                max_connections=max_concurrent_requests,
                max_keepalive_connections=max_concurrent_requests,
                keepalive_expiry=60,
            ),
            timeout=timeout or httpx.Timeout(connect=5.0, write=30, read=180, pool=30),
            follow_redirects=True,
        )
        _api_key = os.getenv("OPENAI_API_KEY", openai_api_key)
        if not _api_key:
            msg = (
                "OPENAI_API_KEY environment variable is not set. "
                "Using the dummy API key 'sk-dummy'..."
            )
            self.logger.warning(msg)
            _api_key = "sk-dummy"
        self._openai_client = AsyncOpenAI(
            api_key=_api_key,
            base_url=endpoint_url,
            http_client=self._http_client,
        )

        self.check_api_alive(
            endpoint_url=endpoint_url,
            model_id=model_id,
        )

    async def apply(self, document: Document) -> Document:
        async for attempt in AsyncRetrying(
            reraise=True,
            wait=wait_random_exponential(multiplier=1, max=5.0),
            stop=stop_after_attempt(self._retry_count),
            before_sleep=before_sleep_log(self.logger, logging.WARNING),
        ):
            with attempt:
                async with self._semaphore:
                    response: ChatCompletion = await self._openai_client.chat.completions.create(
                        model=self.model_id,
                        messages=self.message_generator(document),  # type: ignore
                        **self.api_kwargs,
                    )

        if not response.choices:
            raise RuntimeError("ChatCompletion returned no choices")
        document.extras[self.output_key] = response.choices[0].message.content
        return document

    async def shutdown(self) -> None:
        await super().shutdown()
        await self._http_client.aclose()
        await self._openai_client.close()

    def check_api_alive(self, endpoint_url: str, model_id: str) -> None:
        # Check model is deployed
        with httpx.Client() as c:
            try:
                r = c.get(f"{endpoint_url}/models")
                r.raise_for_status()
                r = r.json()
            except httpx.HTTPError as e:
                msg = (
                    f"Failed to fetch models from your API Endpoint '{endpoint_url}/models'. "
                    "Please check the endpoint URL and ensure your server is running."
                )
                raise ValueError(msg) from e
            if not any(model_info["id"] == model_id for model_info in r["data"]):
                msg = (
                    f"Model '{model_id}' not found in the OpenAI API Endpoint '{endpoint_url}/models'."
                    "Please check the model name and ensure it is available in your API."
                )
                raise ValueError(msg)
