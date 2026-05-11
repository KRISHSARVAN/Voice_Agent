"""MongoDB persistence for chat turns (session history)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from motor.motor_asyncio import AsyncIOMotorCollection


@dataclass(frozen=True)
class ChatTurnDoc:
    id: str
    session_id: str
    user_text: str
    bot_response: str
    created_at: datetime


async def ensure_chat_indexes(collection: AsyncIOMotorCollection) -> None:
    await collection.create_index([("session_id", 1), ("created_at", 1)])


async def insert_chat_turn(
    collection: AsyncIOMotorCollection,
    *,
    session_id: str,
    user_text: str,
    bot_response: str,
    created_at: datetime | None = None,
) -> ChatTurnDoc:
    if created_at is None:
        created_at = datetime.now(timezone.utc)
    elif created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    else:
        created_at = created_at.astimezone(timezone.utc)
    created_at = created_at.replace(microsecond=0)
    doc: dict[str, Any] = {
        "session_id": session_id,
        "user_text": user_text,
        "bot_response": bot_response,
        "created_at": created_at,
    }
    result = await collection.insert_one(doc)
    return ChatTurnDoc(
        id=str(result.inserted_id),
        session_id=session_id,
        user_text=user_text,
        bot_response=bot_response,
        created_at=created_at,
    )


async def list_turns_for_session(
    collection: AsyncIOMotorCollection,
    session_id: str,
    *,
    limit: int = 200,
) -> list[ChatTurnDoc]:
    limit = max(1, min(limit, 500))
    cursor = (
        collection.find({"session_id": session_id})
        .sort([("created_at", 1), ("_id", 1)])
        .limit(limit)
    )
    out: list[ChatTurnDoc] = []
    async for row in cursor:
        created = row["created_at"]
        if isinstance(created, datetime) and created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        elif isinstance(created, datetime):
            created = created.astimezone(timezone.utc)
        else:
            created = datetime.now(timezone.utc)
        out.append(
            ChatTurnDoc(
                id=str(row["_id"]),
                session_id=str(row["session_id"]),
                user_text=str(row["user_text"]),
                bot_response=str(row["bot_response"]),
                created_at=created,
            )
        )
    return out
