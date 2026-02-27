"""Tests for SQLite message log."""

import pytest

from src.memory.store import MessageStore


@pytest.fixture
async def store(tmp_path):
    db_path = tmp_path / "messages.sqlite"
    s = MessageStore(db_path)
    await s.initialize()
    yield s
    await s.close()


@pytest.mark.asyncio
async def test_store_and_retrieve(store):
    await store.save_message(
        channel_id="ch-1", user_id="user-1", user_name="Alice",
        content="Hello", is_bot=False, bot_name=None,
    )
    messages = await store.get_messages(channel_id="ch-1", limit=10)
    assert len(messages) == 1
    assert messages[0]["content"] == "Hello"
    assert messages[0]["user_name"] == "Alice"


@pytest.mark.asyncio
async def test_store_bot_message(store):
    await store.save_message(
        channel_id="ch-1", user_id="bot-1", user_name="OtherBot",
        content="Automated", is_bot=True, bot_name="OtherBot",
    )
    messages = await store.get_messages(channel_id="ch-1", limit=10)
    assert messages[0]["is_bot"] is True
    assert messages[0]["bot_name"] == "OtherBot"


@pytest.mark.asyncio
async def test_search_messages(store):
    await store.save_message(
        channel_id="ch-1", user_id="u1", user_name="Alice",
        content="The weather is nice today", is_bot=False, bot_name=None,
    )
    await store.save_message(
        channel_id="ch-1", user_id="u1", user_name="Alice",
        content="I like pizza", is_bot=False, bot_name=None,
    )
    results = await store.search_messages(query="weather", limit=10)
    assert len(results) == 1
    assert "weather" in results[0]["content"]


@pytest.mark.asyncio
async def test_get_messages_respects_limit(store):
    for i in range(20):
        await store.save_message(
            channel_id="ch-1", user_id="u1", user_name="Alice",
            content=f"Message {i}", is_bot=False, bot_name=None,
        )
    messages = await store.get_messages(channel_id="ch-1", limit=5)
    assert len(messages) == 5


@pytest.mark.asyncio
async def test_get_messages_returns_chronological(store):
    await store.save_message(
        channel_id="ch-1", user_id="u1", user_name="Alice",
        content="First", is_bot=False, bot_name=None,
    )
    await store.save_message(
        channel_id="ch-1", user_id="u1", user_name="Alice",
        content="Second", is_bot=False, bot_name=None,
    )
    messages = await store.get_messages(channel_id="ch-1", limit=10)
    assert messages[0]["content"] == "First"
    assert messages[1]["content"] == "Second"
