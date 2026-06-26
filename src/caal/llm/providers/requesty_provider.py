"""Requesty LLM provider implementation.

Provides access to many cloud models through Requesty's unified,
OpenAI-compatible LLM gateway. Inherits from OpenAICompatibleProvider
since Requesty exposes an OpenAI-compatible API.

See: https://docs.requesty.ai
"""

from __future__ import annotations

import logging

from .openai_compatible_provider import OpenAICompatibleProvider

__all__ = ["RequestyProvider"]

logger = logging.getLogger(__name__)

REQUESTY_BASE_URL = "https://router.requesty.ai/v1"


class RequestyProvider(OpenAICompatibleProvider):
    """Requesty cloud LLM provider.

    Provides access to many models including GPT-4, Claude, Llama, Mistral,
    and many others through a single API. Inherits all chat, streaming, and
    tool handling from OpenAICompatibleProvider.

    Differences from base OpenAICompatibleProvider:
        - Fixed base URL (router.requesty.ai)
        - API key is required
        - Attribution headers for model providers

    Args:
        model: Requesty model ID (e.g., "openai/gpt-4o-mini", "anthropic/claude-sonnet-4-5")
        api_key: Requesty API key (required)
        temperature: Sampling temperature (0.0-2.0)
        max_tokens: Maximum tokens to generate

    See: https://app.requesty.ai/router/list
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> None:
        if not api_key:
            raise ValueError(
                "Requesty API key required. Pass api_key parameter "
                "or configure in settings."
            )

        super().__init__(
            model=model,
            base_url=REQUESTY_BASE_URL,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Add Requesty-specific attribution headers
        self._client.default_headers.update({
            "HTTP-Referer": "https://github.com/caal-project/caal",
            "X-Title": "CAAL Voice Assistant",
        })

        logger.debug(f"RequestyProvider initialized: {model}")

    @property
    def provider_name(self) -> str:
        return "requesty"
