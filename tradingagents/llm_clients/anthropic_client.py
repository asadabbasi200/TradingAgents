"""Anthropic Claude client with content normalization and prompt caching."""
from typing import Any, List, Optional

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_anthropic import ChatAnthropic

from .base_client import BaseLLMClient, normalize_content
from .validators import validate_model


def inject_cache_markers(messages: List[BaseMessage]) -> List[BaseMessage]:
    """Mark the system message (if any) as cacheable via cache_control.

    Anthropic cached prefixes must be identical across calls. We mark the
    system message as the single cached block; analyst reports baked into
    the system prompt get a ~70% input-token discount on repeat calls.
    """
    if not messages or not isinstance(messages[0], SystemMessage):
        return messages
    sys = messages[0]
    content = sys.content
    if isinstance(content, str):
        blocks = [{"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}]
    elif isinstance(content, list):
        blocks = list(content)
        if blocks and isinstance(blocks[-1], dict):
            blocks[-1] = {**blocks[-1], "cache_control": {"type": "ephemeral"}}
    else:
        return messages
    new_sys = SystemMessage(content=blocks, additional_kwargs=sys.additional_kwargs)
    return [new_sys, *messages[1:]]


_PASSTHROUGH_KWARGS = (
    "timeout", "max_retries", "api_key", "max_tokens",
    "callbacks", "http_client", "http_async_client", "effort",
)


class NormalizedChatAnthropic(ChatAnthropic):
    """ChatAnthropic with normalized content output and prompt caching.

    Claude models with extended thinking or tool use return content as a
    list of typed blocks. This normalizes to string for consistent
    downstream handling. Prompt caching is injected on the system message
    so stable prefixes are discounted on subsequent calls.
    """

    def invoke(self, input, config=None, **kwargs):
        if isinstance(input, list):
            input = inject_cache_markers(input)
        return normalize_content(super().invoke(input, config, **kwargs))


class AnthropicClient(BaseLLMClient):
    """Client for Anthropic Claude models."""

    def __init__(self, model: str, base_url: Optional[str] = None, **kwargs):
        super().__init__(model, base_url, **kwargs)

    def get_llm(self) -> Any:
        """Return configured ChatAnthropic instance."""
        self.warn_if_unknown_model()
        llm_kwargs = {"model": self.model}

        if self.base_url:
            llm_kwargs["base_url"] = self.base_url

        for key in _PASSTHROUGH_KWARGS:
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]

        return NormalizedChatAnthropic(**llm_kwargs)

    def validate_model(self) -> bool:
        """Validate model for Anthropic."""
        return validate_model("anthropic", self.model)
