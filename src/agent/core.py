"""Core agent with session management and agentic tool loop.

Uses in-memory session store with tool-calling support via run_tool_loop.
"""

from __future__ import annotations

import logging

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from openai import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    RateLimitError,
)

from src.agent.tool_loop import run_tool_loop
from src.memory.compaction import compact_messages, should_compact
from src.memory.operational import OperationalMemory
from src.memory.vector import VectorMemory
from src.providers.minimax import normalize_messages

logger = logging.getLogger(__name__)


class LLMProviderError(Exception):
    """Raised when the LLM provider cannot fulfill a request."""

    def __init__(self, message: str, *, recoverable: bool = True):
        super().__init__(message)
        self.recoverable = recoverable


class CoreAgent:
    def __init__(
        self,
        *,
        llm: ChatOpenAI,
        system_prompt: str,
        max_session_messages: int = 50,
        vector_memory: VectorMemory | None = None,
        operational_memory: OperationalMemory | None = None,
        tools: list | None = None,
        skill_registry=None,
    ):
        self.tools = tools or []
        self.skill_registry = skill_registry

        # Bind tools to LLM if any are provided
        if self.tools:
            self.llm = llm.bind_tools(self.tools)
        else:
            self.llm = llm

        self._raw_llm = llm
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
        """Build system prompt, appending operational memory and skill index."""
        prompt = self.system_prompt
        if self.operational_memory is not None:
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

        if self.skill_registry is not None:
            index = self.skill_registry.get_skill_index()
            if index and index != "No skills available.":
                prompt += (
                    "\n\n## Available Skills\n"
                    "When a request matches a skill trigger, use `dispatch_skill`.\n\n"
                    f"{index}"
                )

        if self.tools:
            tool_names = [t.name for t in self.tools]
            prompt += (
                "\n\n## Tools\n"
                f"You have access to the following tools: {', '.join(tool_names)}.\n"
                "Use them when needed to fulfill user requests. "
                "Call tools by including tool_calls in your response.\n\n"
                "If you have the `create_skill` tool and a user's request would benefit from "
                "a reusable capability that doesn't exist yet, create a new skill for it. "
                "This lets you learn and improve over time."
            )

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

    async def invoke(
        self,
        *,
        session_id: str,
        user_message: str,
        user_name: str,
        on_tool_call=None,
    ) -> str:
        session = self._get_session(session_id)

        user_msg = HumanMessage(content=f"[{user_name}]: {user_message}")
        session.append(user_msg)

        # Build system prompt with operational memory + skill index
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

        try:
            if self.tools:
                response = await run_tool_loop(
                    llm=self.llm,
                    messages=normalized,
                    tools=self.tools,
                    on_tool_call=on_tool_call,
                )
            else:
                response = await self.llm.ainvoke(normalized)
        except AuthenticationError as e:
            logger.error("MiniMax authentication failed: %s", e)
            session.pop()
            raise LLMProviderError(
                "Authentication with MiniMax failed. The API key may be invalid or expired.",
                recoverable=False,
            ) from e
        except RateLimitError as e:
            logger.warning("MiniMax rate limit hit: %s", e)
            session.pop()
            raise LLMProviderError(
                "MiniMax rate limit reached. Please try again in a moment.",
                recoverable=True,
            ) from e
        except APIConnectionError as e:
            logger.error("Cannot reach MiniMax API: %s", e)
            session.pop()
            raise LLMProviderError(
                "Cannot reach the MiniMax API. The service may be down.",
                recoverable=True,
            ) from e
        except APIStatusError as e:
            logger.error("MiniMax API error (status %s): %s", e.status_code, e)
            session.pop()
            raise LLMProviderError(
                f"MiniMax returned an error (HTTP {e.status_code}). "
                "The service may be experiencing issues.",
                recoverable=e.status_code >= 500,
            ) from e

        ai_msg = AIMessage(content=response.content)
        session.append(ai_msg)

        # Index the user message in the vector store
        self._index_message(
            text=f"[{user_name}]: {user_message}",
            metadata={"session_id": session_id, "user_name": user_name},
        )

        if should_compact(session, max_messages=self.max_session_messages):
            compacted = await compact_messages(session, llm=self._raw_llm)
            session.clear()
            session.extend(compacted)

        return response.content
