"""WebSocket voice pipeline endpoint.

``/ws/voice`` keeps one persistent, bidirectional connection alive for the
duration of a call session, replacing the per-turn HTTP POST + StreamingResponse
pattern.  Benefits over the HTTP approach:

* No TCP/TLS handshake overhead on every voice turn.
* True full-duplex: the client can send an ``interrupt`` message *while* the
  server is still streaming audio, and the server cancels TTS/LLM immediately.
* Binary frames for audio → no base64 encoding/decoding overhead (~33 % smaller).
* LLM token events streamed to the client for live transcript display.

─────────────────────────────────────────────────────────────────────────────
Protocol
─────────────────────────────────────────────────────────────────────────────
Client → Server  (JSON text frames)
  {"type": "query",     "session_id": "...", "messages": [...],
   "language_code": "en-IN", "top_k": 6}
  {"type": "transcribe", "audio_wav_base64": "<base64>"}   # 16 kHz mono WAV
  {"type": "synthesize", "text": "...", "language_code": "en-IN"}
  {"type": "interrupt"}
  {"type": "ping"}

Server → Client
  <binary frame>          raw WAV bytes — TTS chunk (query + synthesize paths)
  {"type": "token",       "text": "..."}      streaming LLM token (query only)
  {"type": "done",        "session_id": "...", "answer": "..."}
  {"type": "transcribe_done", "text": "...", "language_code": "en-IN"}
  {"type": "synthesize_done"}
  {"type": "interrupted"}
  {"type": "error",       "message": "..."}
  {"type": "pong"}
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import uuid

from fastapi import WebSocket, WebSocketDisconnect

from app.db import insert_chat_turn
from app.rag_service import astream_rag_chat
from app.stt_service import transcribe_audio
from app.tts_service import (
    DEFAULT_VOICE_SPEAKER,
    _strip_markdown,
    synthesize_chunk,
    synthesize_speech_stream,
)

logger = logging.getLogger(__name__)

_SENTENCE_RE = re.compile(r"(?<=[.!?।])\s+")


async def handle_ws_voice(websocket: WebSocket) -> None:
    """Drive one persistent WebSocket voice session."""
    await websocket.accept()

    settings = websocket.app.state.settings
    vs = websocket.app.state.vectorstore
    mongo_coll = websocket.app.state.mongo_collection

    session_id: str = ""
    active_task: asyncio.Task | None = None
    # One Bulbul voice for the whole browser call (greeting + every reply).
    voice_speaker = DEFAULT_VOICE_SPEAKER

    # ── low-level send helpers ────────────────────────────────────────────────

    async def _send(data: dict) -> None:
        try:
            await websocket.send_text(json.dumps(data))
        except Exception:
            pass

    async def _send_audio(wav: bytes) -> None:
        try:
            await websocket.send_bytes(wav)
        except Exception:
            pass

    async def _cancel(task: asyncio.Task | None) -> None:
        if task and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    # ── voice pipeline ────────────────────────────────────────────────────────

    async def _transcribe(audio_b64: str) -> None:
        """Decode WAV from base64 and return Sarvam STT result."""
        try:
            raw = base64.b64decode(audio_b64, validate=False)
        except Exception:
            await _send({"type": "error", "message": "Invalid base64 audio payload"})
            return
        if not raw:
            await _send({"type": "transcribe_done", "text": "", "language_code": "en-IN"})
            return
        try:
            text, lang = await transcribe_audio(
                raw,
                api_key=settings.stt_api_key,
                content_type="audio/wav",
                filename="recording.wav",
            )
        except asyncio.CancelledError:
            await _send({"type": "interrupted"})
            raise
        except Exception as exc:
            logger.exception("WS transcribe failed")
            msg = str(exc) if settings.environment == "development" else "Transcription failed"
            await _send({"type": "error", "message": msg})
            return

        await _send(
            {"type": "transcribe_done", "text": text or "", "language_code": lang or "en-IN"}
        )

    async def _synthesize(text: str, language_code: str) -> None:
        """Stream TTS audio as binary WAV frames (welcome / short prompts)."""
        if not settings.tts_api_key:
            await _send({"type": "error", "message": "TTS is not configured on this server."})
            return
        if not text or not text.strip():
            await _send({"type": "error", "message": "synthesize text must not be empty."})
            return
        try:
            async for wav_chunk in synthesize_speech_stream(
                text.strip(),
                api_key=settings.tts_api_key,
                language_code=language_code or "en-IN",
                speaker=voice_speaker,
            ):
                await _send_audio(wav_chunk)
            await _send({"type": "synthesize_done"})
        except asyncio.CancelledError:
            await _send({"type": "interrupted"})
            raise
        except Exception:
            logger.exception("WS synthesize failed")
            await _send(
                {
                    "type": "error",
                    "message": "TTS synthesis failed",
                }
            )

    async def _pipeline(messages: list, language_code: str, top_k: int | None) -> None:
        """Run RAG → LLM stream → parallel TTS for one user turn."""
        if not settings.tts_api_key:
            await _send({"type": "error", "message": "TTS is not configured on this server."})
            return

        # Resolve the latest user message for persistence
        user_text = ""
        for m in reversed(messages):
            role = m.get("role") if isinstance(m, dict) else m[0]
            content = m.get("content") if isinstance(m, dict) else m[1]
            if role == "user":
                user_text = content
                break

        lc_msgs = [
            (m["role"], m["content"]) if isinstance(m, dict) else (str(m[0]), str(m[1]))
            for m in messages
        ]

        pairs_out: list = []
        answer_parts: list[str] = []
        sentence_buf = ""
        tts_tasks: list[asyncio.Task] = []
        yielded = 0

        try:
            async for token in astream_rag_chat(
                vectorstore=vs,
                settings=settings,
                messages=lc_msgs,
                top_k=top_k,
                language_code=language_code,
                pairs_out=pairs_out,
            ):
                answer_parts.append(token)
                sentence_buf += token

                # Send token for live UI display
                await _send({"type": "token", "text": token})

                # Fire TTS for each completed sentence immediately
                parts = _SENTENCE_RE.split(sentence_buf)
                if len(parts) > 1:
                    for sentence in parts[:-1]:
                        clean = _strip_markdown(sentence).strip()
                        if clean:
                            tts_tasks.append(
                                asyncio.ensure_future(
                                    synthesize_chunk(
                                        clean, settings.tts_api_key, language_code, voice_speaker
                                    )
                                )
                            )
                    sentence_buf = parts[-1]

                # Eagerly deliver already-finished chunks (preserving order)
                while yielded < len(tts_tasks) and tts_tasks[yielded].done():
                    try:
                        await _send_audio(tts_tasks[yielded].result())
                    except Exception:
                        logger.exception("TTS chunk %d failed (eager)", yielded)
                    yielded += 1

        except asyncio.CancelledError:
            for t in tts_tasks[yielded:]:
                if not t.done():
                    t.cancel()
            await _send({"type": "interrupted"})
            return
        except Exception:
            logger.exception("LLM stream failed in WS pipeline")
            await _send({"type": "error", "message": "LLM pipeline failed"})
            return

        # Flush the final sentence fragment (no trailing punctuation)
        if sentence_buf.strip():
            clean = _strip_markdown(sentence_buf).strip()
            if clean:
                tts_tasks.append(
                    asyncio.ensure_future(
                        synthesize_chunk(
                            clean, settings.tts_api_key, language_code, voice_speaker
                        )
                    )
                )

        # Deliver remaining TTS chunks in sentence order
        for i in range(yielded, len(tts_tasks)):
            try:
                await _send_audio(await tts_tasks[i])
            except asyncio.CancelledError:
                await _send({"type": "interrupted"})
                return
            except Exception:
                logger.exception("TTS chunk %d failed", i)

        full_answer = "".join(answer_parts)
        await _send({"type": "done", "session_id": session_id, "answer": full_answer})

        logger.info("[WS] [USER] %s", user_text)
        logger.info("[WS] [BOT]  %s", full_answer)

        try:
            await insert_chat_turn(
                mongo_coll,
                session_id=session_id,
                user_text=user_text,
                bot_response=full_answer,
            )
        except Exception:
            logger.exception("Failed to persist WS turn session_id=%s", session_id)

    # ── main receive loop ─────────────────────────────────────────────────────

    try:
        while True:
            raw = await websocket.receive()

            if raw.get("type") == "websocket.disconnect":
                break

            text = raw.get("text")
            if not text:
                continue

            try:
                msg = json.loads(text)
            except json.JSONDecodeError:
                await _send({"type": "error", "message": "Invalid JSON"})
                continue

            msg_type = msg.get("type", "")

            if msg_type == "query":
                # Cancel any in-flight pipeline before starting a new one
                await _cancel(active_task)
                session_id = msg.get("session_id") or session_id or str(uuid.uuid4())
                messages = msg.get("messages", [])
                language_code = msg.get("language_code", "en-IN")
                top_k = msg.get("top_k")

                if not messages:
                    await _send({"type": "error", "message": "No messages provided"})
                    continue

                active_task = asyncio.ensure_future(
                    _pipeline(messages, language_code, top_k)
                )

            elif msg_type == "transcribe":
                await _cancel(active_task)
                audio_field = msg.get("audio_wav_base64") or msg.get("audio_base64") or ""
                if not isinstance(audio_field, str) or not audio_field.strip():
                    await _send({"type": "error", "message": "Missing audio_wav_base64"})
                    continue
                active_task = asyncio.ensure_future(_transcribe(audio_field))

            elif msg_type == "synthesize":
                await _cancel(active_task)
                tts_text = msg.get("text") or ""
                lang = msg.get("language_code", "en-IN")
                active_task = asyncio.ensure_future(_synthesize(str(tts_text), str(lang)))

            elif msg_type == "interrupt":
                await _cancel(active_task)
                active_task = None
                await _send({"type": "interrupted"})

            elif msg_type == "ping":
                await _send({"type": "pong"})

    except WebSocketDisconnect:
        logger.info("WS client disconnected session_id=%s", session_id)
    except Exception:
        logger.exception("WS handler error session_id=%s", session_id)
    finally:
        await _cancel(active_task)
