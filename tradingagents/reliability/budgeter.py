"""Context size guard and lazy history summarization."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List, Optional

log = logging.getLogger(__name__)

# Rough token estimate — 4 chars per token for English. Accurate enough for
# a pre-flight budget check; we don't need exact.
_CHARS_PER_TOKEN = 4.0


def estimate_tokens(text: str | List[Any]) -> int:
    if isinstance(text, list):
        return sum(estimate_tokens(_content_of(x)) for x in text)
    if not isinstance(text, str):
        text = str(text)
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


def _content_of(msg: Any) -> str:
    if isinstance(msg, dict):
        return str(msg.get("content", ""))
    if hasattr(msg, "content"):
        return str(msg.content)
    return str(msg)


def _total_tokens(messages: List[Any]) -> int:
    return sum(estimate_tokens(_content_of(m)) for m in messages)


_SUMMARIZE_PROMPT = (
    "You are compressing prior debate turns for another agent to read. "
    "Preserve: the main claims, any numeric evidence cited, and disagreements. "
    "Omit pleasantries. Output at most 400 words. Debate turns:\n\n"
)


def compress_history(
    summarizer_client: Any,
    middle_messages: List[Any],
) -> str:
    """Call the summarizer on a list of messages, return a compact summary."""
    joined = "\n\n".join(f"[{m.get('role', 'msg')}]: {_content_of(m)}" for m in middle_messages)
    resp = summarizer_client.invoke(_SUMMARIZE_PROMPT + joined)
    if hasattr(resp, "content"):
        return resp.content
    return str(resp)


@dataclass
class ContextBudgeter:
    ceiling: int
    summarizer_client: Optional[Any] = None
    keep_last_n: int = 4

    def ensure_under_ceiling(self, messages: List[Any]) -> List[Any]:
        """If messages exceed the ceiling, compress the middle turns.

        Layout after compression:
          [system (if present), summary_message, last_n turns...]

        If summarization itself fails, fall back to hard truncation:
          [system, last_n turns...]
        """
        if _total_tokens(messages) <= self.ceiling:
            return messages
        if not messages:
            return messages

        system_msgs: List[Any] = []
        tail = messages
        if isinstance(messages[0], dict) and messages[0].get("role") == "system":
            system_msgs = [messages[0]]
            tail = messages[1:]
        elif hasattr(messages[0], "type") and messages[0].type == "system":
            system_msgs = [messages[0]]
            tail = messages[1:]

        if len(tail) <= self.keep_last_n:
            return messages  # nothing to compress

        to_compress = tail[: -self.keep_last_n]
        keep = tail[-self.keep_last_n :]

        if self.summarizer_client is None:
            return system_msgs + keep  # hard truncation

        try:
            summary = compress_history(self.summarizer_client, to_compress)
        except Exception as e:  # noqa: BLE001 — budgeter must never raise upward
            log.warning("summarization failed, falling back to truncation: %s", e)
            return system_msgs + keep

        summary_msg = {"role": "user", "content": f"[Prior turns compressed]\n{summary}"}
        return system_msgs + [summary_msg] + keep
