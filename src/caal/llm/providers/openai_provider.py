"""OpenAI-compatible LLM provider implementation.

Provides integration with any OpenAI-compatible API endpoint, enabling CAAL
to work with inference servers like LM Studio, vLLM, LocalAI, FastFlowLM,
text-generation-webui, and others.

Uses the official openai Python library for async API calls.
"""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING, Any

from openai import AsyncOpenAI

from .base import LLMProvider, LLMResponse, ToolCall

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

__all__ = ["OpenAIProvider"]

logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    """OpenAI-compatible LLM provider.

    Works with any server implementing the OpenAI chat completions API,
    including LM Studio, vLLM, LocalAI, FastFlowLM, and text-generation-webui.

    Features:
        - Async API calls via openai.AsyncOpenAI
        - Tool/function calling support (requires model support)
        - Optional API key authentication

    Args:
        model: Model name as recognized by the server
        base_url: Server URL (e.g., "http://localhost:8080/v1")
        api_key: Optional API key (some servers don't require auth)
        temperature: Sampling temperature (0.0-2.0)
        max_tokens: Maximum tokens to generate
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key or os.environ.get("OPENAI_COMPAT_API_KEY")
        self._temperature = temperature
        self._max_tokens = max_tokens

        if not self._base_url:
            raise ValueError(
                "Base URL required for OpenAI-compatible provider. "
                "Set OPENAI_COMPAT_HOST environment variable or pass base_url parameter."
            )

        # Use "not-needed" as placeholder if no API key - some servers require
        # the header to be present even if they don't validate it
        self._client = AsyncOpenAI(
            base_url=self._base_url,
            api_key=self._api_key or "not-needed",
        )

        logger.debug(f"OpenAIProvider initialized: {model} at {base_url}")

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def model(self) -> str:
        return self._model

    @property
    def temperature(self) -> float:
        return self._temperature

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Execute non-streaming chat completion.

        Args:
            messages: List of message dicts
            tools: Optional tool definitions
            **kwargs: Additional options

        Returns:
            Normalized LLMResponse
        """
        request_kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "stream": False,
        }

        if tools:
            request_kwargs["tools"] = tools
            request_kwargs["tool_choice"] = "auto"

        response = await self._client.chat.completions.create(**request_kwargs)

        # Extract from OpenAI response format
        message = response.choices[0].message

        # Extract tool calls if present
        tool_calls: list[ToolCall] = []
        if message.tool_calls:
            for tc in message.tool_calls:
                # OpenAI returns arguments as JSON string
                args = self.parse_tool_arguments(tc.function.arguments)
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=args,
                    )
                )

        return LLMResponse(content=message.content, tool_calls=tool_calls)

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Execute streaming chat completion.

        Args:
            messages: List of message dicts
            tools: Optional tool definitions (for validation of tool_calls in history)
            **kwargs: Additional options

        Yields:
            String chunks of response content
        """
        request_kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "stream": True,
        }

        # Include tools if provided (for validation of tool_calls in message history)
        # Set tool_choice="none" to prevent tool calls in streaming mode
        if tools:
            request_kwargs["tools"] = tools
            request_kwargs["tool_choice"] = "none"

        stream = await self._client.chat.completions.create(**request_kwargs)

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def parse_tool_arguments(self, arguments: Any) -> dict[str, Any]:
        """Parse tool arguments from JSON string.

        OpenAI-compatible APIs return tool call arguments as a JSON string.

        Args:
            arguments: JSON string or dict of arguments

        Returns:
            Parsed arguments dict
        """
        if isinstance(arguments, str):
            try:
                return json.loads(arguments)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse tool arguments: {arguments}")
                return {}
        if isinstance(arguments, dict):
            return arguments
        return {}

    def format_tool_result(
        self,
        content: str,
        tool_call_id: str | None,
        tool_name: str,
    ) -> dict[str, Any]:
        """Format tool result message for OpenAI-compatible API.

        Args:
            content: Tool execution result as string
            tool_call_id: ID of the tool call being responded to
            tool_name: Name of the tool that was called

        Returns:
            Message dict for OpenAI API
        """
        return {
            "role": "tool",
            "content": content,
            "tool_call_id": tool_call_id,
            "name": tool_name,
        }
