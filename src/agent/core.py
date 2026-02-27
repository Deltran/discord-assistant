"""Core agent with session management.

Uses in-memory session store. Later phases add full LangGraph graph
with sub-agent routing and durable checkpointing.
"""

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.memory.compaction import compact_messages, should_compact
from src.providers.minimax import normalize_messages


class CoreAgent:
    def __init__(self, *, llm: ChatOpenAI, system_prompt: str, max_session_messages: int = 50):
        self.llm = llm
        self.system_prompt = system_prompt
        self.max_session_messages = max_session_messages
        self._sessions: dict[str, list[BaseMessage]] = {}

    def _get_session(self, session_id: str) -> list[BaseMessage]:
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        return self._sessions[session_id]

    async def invoke(self, *, session_id: str, user_message: str, user_name: str) -> str:
        session = self._get_session(session_id)

        user_msg = HumanMessage(content=f"[{user_name}]: {user_message}")
        session.append(user_msg)

        full_messages = [SystemMessage(content=self.system_prompt)] + session
        normalized = normalize_messages(full_messages)

        response = await self.llm.ainvoke(normalized)

        ai_msg = AIMessage(content=response.content)
        session.append(ai_msg)

        if should_compact(session, max_messages=self.max_session_messages):
            compacted = await compact_messages(session, llm=self.llm)
            session.clear()
            session.extend(compacted)

        return response.content
