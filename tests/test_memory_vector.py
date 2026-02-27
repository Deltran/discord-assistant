"""Tests for vector store with hybrid search."""

import pytest

from src.memory.vector import VectorMemory


@pytest.fixture
def vector_memory(tmp_path):
    return VectorMemory(persist_dir=tmp_path / "vectors")


def test_add_and_search(vector_memory):
    vector_memory.add(
        text="The weather in Austin is sunny today",
        metadata={"channel_id": "ch-1", "user_name": "Alice", "timestamp": "2026-01-01"},
    )
    vector_memory.add(
        text="I enjoy programming in Python",
        metadata={"channel_id": "ch-1", "user_name": "Bob", "timestamp": "2026-01-02"},
    )
    results = vector_memory.search("what's the weather like", k=1)
    assert len(results) == 1
    assert "weather" in results[0]["text"].lower() or "sunny" in results[0]["text"].lower()


def test_search_includes_metadata(vector_memory):
    vector_memory.add(
        text="Important decision was made",
        metadata={"channel_id": "ch-1", "user_name": "Alice", "timestamp": "2026-02-01"},
    )
    results = vector_memory.search("decision", k=1)
    assert results[0]["metadata"]["user_name"] == "Alice"


def test_search_empty_store(vector_memory):
    results = vector_memory.search("anything", k=5)
    assert results == []


def test_add_multiple_and_search(vector_memory):
    for i in range(10):
        vector_memory.add(
            text=f"Document number {i} about topic {i % 3}",
            metadata={"index": str(i)},
        )
    results = vector_memory.search("topic", k=3)
    assert len(results) == 3
