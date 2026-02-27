"""Core agent with session management.

Uses in-memory session store. Later phases add full LangGraph graph
with sub-agent routing and durable checkpointing.
"""

from __future__ import annotations

import logging

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.memory.compaction import compact_messages, should_compact
from src.memory.operational import OperationalMemory
from src.memory.vector import VectorMemory
from src.providers.minimax import normalize_messages

logger = logging.getLogger(__name__)


class CoreAgent:
    def __init__(
        self,
        *,
        llm: ChatOpenAI,
        system_prompt: str,
        max_session_messages: int = 50,
        vector_memory: VectorMemory | None = None,
        operational_memory: OperationalMemory | None = None,
    ):
        self.llm = llm
        self.system_prompt = system_prompt
        self.max_session_messages = max_session_messages
        self.vector_memory = vector_memory
        self.operational_memory = operational_memory
        self._sessions: dict[str, list[BaseMessage]] = {}

    def _get_session(self, session_id: str) -> list[BaseMessage]:
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        return self._sessions[session_id]

    def _build_system_prompt(self) -> str:
        """Build system prompt, appending operational memory if available."""
        prompt = self.system_prompt
        if self.operational_memory is None:
            return prompt

        mem = self.operational_memory.read_all()
        sections = []
        if mem["safety_rules"].strip():
            sections.append(mem["safety_rules"])
        if mem["preferences"].strip():
            sections.append(mem["preferences"])
        if mem["operational_notes"].strip():
            sections.append(mem["operational_notes"])

        if sections:
            prompt += "\n\n" + "\n\n".join(sections)
        return prompt

    def _search_vector_context(self, query: str) -> list[dict]:
        """Search vector memory for relevant prior context."""
        if self.vector_memory is None:
            return []
        try:
            return self.vector_memory.search(query, k=5)
        except Exception:
            logger.exception("Vector search failed")
            return []

    def _index_message(self, text: str, metadata: dict) -> None:
        """Index a message in the vector store."""
        if self.vector_memory is None:
            return
        try:
            self.vector_memory.add(text=text, metadata=metadata)
        except Exception:
            logger.exception("Vector indexing failed")

    async def invoke(self, *, session_id: str, user_message: str, user_name: str) -> str:
        session = self._get_session(session_id)

        user_msg = HumanMessage(content=f"[{user_name}]: {user_message}")
        session.append(user_msg)

        # Build system prompt with operational memory
        system_prompt = self._build_system_prompt()

        # Search vector memory for relevant context
        retrieved = self._search_vector_context(user_message)

        messages: list[BaseMessage] = [SystemMessage(content=system_prompt)]

        if retrieved:
            context_text = "\n".join(
                f"- {r['text']}" for r in retrieved
            )
            messages.append(
                HumanMessage(content=f"[Retrieved context]:\n{context_text}")
            )

        messages.extend(session)
        normalized = normalize_messages(messages)

        response = await self.llm.ainvoke(normalized)

        ai_msg = AIMessage(content=response.content)
        session.append(ai_msg)

        # Index the user message in the vector store
        self._index_message(
            text=f"[{user_name}]: {user_message}",
            metadata={"session_id": session_id, "user_name": user_name},
        )

        if should_compact(session, max_messages=self.max_session_messages):
            compacted = await compact_messages(session, llm=self.llm)
            session.clear()
            session.extend(compacted)

        return response.content
