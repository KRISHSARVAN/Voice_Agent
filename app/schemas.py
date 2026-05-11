"""Pydantic request/response models."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, field_validator


class ChatMessage(BaseModel):
    role: str = Field(description='One of: "system", "user", "assistant".')
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1)
    session_id: str | None = Field(
        default=None,
        description="Client session id for history; a new id is returned when omitted.",
    )
    top_k: int | None = Field(default=None, ge=1, le=40)
    include_sources: bool = True
    language_code: str = Field(
        default="en-IN",
        description="BCP-47 language code detected by STT (e.g. hi-IN, gu-IN, en-IN). "
                    "Used to translate the query for retrieval and to instruct the LLM "
                    "to reply in the user's language.",
    )

    @field_validator("messages")
    @classmethod
    def last_must_be_user(cls, v: list[ChatMessage]) -> list[ChatMessage]:
        if v[-1].role != "user":
            raise ValueError('Last message must have role="user"')
        return v




class SourceChunk(BaseModel):
    title: str | None = None
    source: str | None = None
    content_excerpt: str
    chroma_distance: float | None = Field(
        default=None,
        description="Chroma distance for this hit; lower usually means more similar (depends on metric).",
    )


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceChunk] = Field(default_factory=list)
    session_id: str | None = Field(
        default=None,
        description="Session id used for this turn (echoed or newly assigned).",
    )


class ConversationTurn(BaseModel):
    id: str
    session_id: str
    user_text: str
    bot_response: str
    created_at: datetime
