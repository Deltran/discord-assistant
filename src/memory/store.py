"""SQLite message log — full history, never deleted."""

from datetime import datetime, timezone
from pathlib import Path

import aiosqlite


class MessageStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def initialize(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                channel_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                user_name TEXT NOT NULL,
                content TEXT NOT NULL,
                is_bot INTEGER NOT NULL DEFAULT 0,
                bot_name TEXT
            )
        """)
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_channel ON messages(channel_id, timestamp)"
        )
        await self._db.commit()

    async def close(self):
        if self._db:
            await self._db.close()

    async def save_message(
        self, *, channel_id: str, user_id: str, user_name: str,
        content: str, is_bot: bool, bot_name: str | None,
    ) -> int:
        assert self._db is not None
        cursor = await self._db.execute(
            """INSERT INTO messages (timestamp, channel_id, user_id, user_name, content, is_bot, bot_name)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (datetime.now(timezone.utc).isoformat(), channel_id, user_id,
             user_name, content, int(is_bot), bot_name),
        )
        await self._db.commit()
        return cursor.lastrowid

    async def get_messages(self, *, channel_id: str, limit: int = 50) -> list[dict]:
        assert self._db is not None
        cursor = await self._db.execute(
            "SELECT * FROM messages WHERE channel_id = ? ORDER BY timestamp DESC LIMIT ?",
            (channel_id, limit),
        )
        rows = await cursor.fetchall()
        return [self._row_to_dict(row) for row in reversed(rows)]

    async def search_messages(self, *, query: str, limit: int = 20) -> list[dict]:
        assert self._db is not None
        cursor = await self._db.execute(
            "SELECT * FROM messages WHERE content LIKE ? ORDER BY timestamp DESC LIMIT ?",
            (f"%{query}%", limit),
        )
        rows = await cursor.fetchall()
        return [self._row_to_dict(row) for row in rows]

    @staticmethod
    def _row_to_dict(row) -> dict:
        return {
            "id": row["id"],
            "timestamp": row["timestamp"],
            "channel_id": row["channel_id"],
            "user_id": row["user_id"],
            "user_name": row["user_name"],
            "content": row["content"],
            "is_bot": bool(row["is_bot"]),
            "bot_name": row["bot_name"],
        }
