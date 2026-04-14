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
    """ChatAnthropic with normalized content output, prompt caching, retry, and budgeting.

    Claude models with extended thinking or tool use return content as a
    list of typed blocks. This normalizes to string for consistent
    downstream handling. Prompt caching is injected on the system message
    so stable prefixes are discounted on subsequent calls. When a
    ``retry_policy`` is supplied, the underlying invoke is wrapped in
    ``with_retries`` for transparent backoff. When a ``context_budgeter``
    is supplied, oversized message histories are compressed before the
    call.
    """

    def __init__(self, *, retry_policy=None, context_budgeter=None, **data):
        super().__init__(**data)
        # Pydantic BaseModel rejects plain instance attrs; bypass via object.__setattr__.
        object.__setattr__(self, "_retry_policy", retry_policy)
        object.__setattr__(self, "_context_budgeter", context_budgeter)

    def invoke(self, input, config=None, **kwargs):
        from tradingagents.reliability.retry import with_retries

        if isinstance(input, list):
            if getattr(self, "_context_budgeter", None) is not None:
                input = self._context_budgeter.ensure_under_ceiling(input)
            input = inject_cache_markers(input)

        def _do_invoke():
            return normalize_content(
                super(NormalizedChatAnthropic, self).invoke(input, config, **kwargs)
            )

        if getattr(self, "_retry_policy", None) is not None:
            return with_retries(_do_invoke, policy=self._retry_policy)
        return _do_invoke()


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

        # Anthropic's `effort` parameter is only supported on Sonnet / Opus.
        # Haiku rejects it with a 400 invalid_request_error, so strip it.
        if "effort" in llm_kwargs and "haiku" in self.model.lower():
            llm_kwargs.pop("effort")

        # Reliability primitives — passed to our subclass __init__, not ChatAnthropic.
        if "retry_policy" in self.kwargs:
            llm_kwargs["retry_policy"] = self.kwargs["retry_policy"]
        if "context_budgeter" in self.kwargs:
            llm_kwargs["context_budgeter"] = self.kwargs["context_budgeter"]

        return NormalizedChatAnthropic(**llm_kwargs)

    def validate_model(self) -> bool:
        """Validate model for Anthropic."""
        return validate_model("anthropic", self.model)
