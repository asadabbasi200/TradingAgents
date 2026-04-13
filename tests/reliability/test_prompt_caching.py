from langchain_core.messages import HumanMessage, SystemMessage

from tradingagents.llm_clients.anthropic_client import inject_cache_markers


def test_system_message_gets_cache_control():
    msgs = [SystemMessage(content="stable analyst report here..." * 300),
            HumanMessage(content="what next?")]
    out = inject_cache_markers(msgs)
    # The system message content should now be a list of blocks with
    # cache_control on the last block.
    sys_content = out[0].content
    assert isinstance(sys_content, list)
    assert any(
        isinstance(b, dict) and b.get("cache_control", {}).get("type") == "ephemeral"
        for b in sys_content
    )
    # Human message is left alone
    assert isinstance(out[1].content, str)


def test_no_system_message_is_noop():
    msgs = [HumanMessage(content="hi")]
    out = inject_cache_markers(msgs)
    assert out == msgs
