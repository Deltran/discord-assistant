"""Tests for MiniMax provider and turn normalization."""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.providers.minimax import normalize_messages, create_llm


def test_normalize_already_alternating():
    msgs = [
        SystemMessage(content="system"),
        HumanMessage(content="hello"),
        AIMessage(content="hi"),
        HumanMessage(content="bye"),
    ]
    result = normalize_messages(msgs)
    assert len(result) == 4
    assert isinstance(result[0], SystemMessage)
    assert isinstance(result[1], HumanMessage)
    assert isinstance(result[2], AIMessage)
    assert isinstance(result[3], HumanMessage)


def test_normalize_consecutive_human():
    msgs = [
        SystemMessage(content="system"),
        HumanMessage(content="hello"),
        HumanMessage(content="world"),
        AIMessage(content="hi"),
    ]
    result = normalize_messages(msgs)
    assert len(result) == 3
    assert isinstance(result[1], HumanMessage)
    assert "hello" in result[1].content
    assert "world" in result[1].content


def test_normalize_consecutive_ai():
    msgs = [
        SystemMessage(content="system"),
        HumanMessage(content="hello"),
        AIMessage(content="hi"),
        AIMessage(content="there"),
    ]
    result = normalize_messages(msgs)
    assert len(result) == 3
    assert isinstance(result[2], AIMessage)
    assert "hi" in result[2].content
    assert "there" in result[2].content


def test_normalize_must_start_with_human_after_system():
    msgs = [
        SystemMessage(content="system"),
        AIMessage(content="unprompted"),
        HumanMessage(content="hello"),
    ]
    result = normalize_messages(msgs)
    non_system = [m for m in result if not isinstance(m, SystemMessage)]
    assert isinstance(non_system[0], HumanMessage)


def test_create_llm_returns_chat_openai():
    from unittest.mock import patch
    import os

    env = {"MINIMAX_API_KEY": "test-key", "DISCORD_TOKEN": "t", "MONITORING_CHANNEL_ID": "1"}
    with patch.dict(os.environ, env, clear=False):
        from importlib import reload
        import src.settings as settings_mod
        reload(settings_mod)
        settings = settings_mod.Settings()
        llm = create_llm(settings)
        assert llm.model_name == "MiniMax-M2.5"
