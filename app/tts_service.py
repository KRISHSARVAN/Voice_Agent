"""Deepgram Aura-2 text-to-speech service."""

from __future__ import annotations

import asyncio
import logging
import re
from typing import AsyncIterator

import httpx

logger = logging.getLogger(__name__)

TTS_MODEL = "aura-2-thalia-en"   # default voice; swap to e.g. aura-2-arcas-en for male
_DEEPGRAM_TTS_URL = "https://api.deepgram.com/v1/speak"
_MAX_CHUNK_CHARS = 2000  # Deepgram TTS supports long input; split only for parallelism


def _strip_markdown(text: str) -> str:
    """Remove markdown formatting so TTS reads clean prose."""
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'#{1,6}\s*', '', text)
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    text = re.sub(r'`{1,3}[^`]*`{1,3}', '', text)
    text = re.sub(r'^\s*[-*]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n{2,}', '. ', text)
    return text.strip()


def _split_into_chunks(text: str, max_chars: int = _MAX_CHUNK_CHARS) -> list[str]:
    """Split text at sentence boundaries so each chunk stays under max_chars."""
    sentences = re.split(r'(?<=[.!?।])\s+', text)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if not sentence:
            continue
        if len(current) + len(sentence) + 1 <= max_chars:
            current = (current + " " + sentence).strip() if current else sentence
        else:
            if current:
                chunks.append(current)
            while len(sentence) > max_chars:
                chunks.append(sentence[:max_chars])
                sentence = sentence[max_chars:]
            current = sentence
    if current:
        chunks.append(current)
    return chunks or [text]


async def _deepgram_tts(text: str, api_key: str) -> bytes:
    """Call Deepgram Aura-2 TTS API and return MP3 bytes."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            _DEEPGRAM_TTS_URL,
            headers={
                "Authorization": f"Token {api_key}",
                "Content-Type": "application/json",
            },
            json={"text": text},
            params={"model": TTS_MODEL},
        )
        response.raise_for_status()
        return response.content


def pick_speaker() -> str:
    """Return the active Deepgram voice model name."""
    return TTS_MODEL


async def synthesize_chunk(
    text: str,
    api_key: str,
    language_code: str = "en-IN",
    speaker: str | None = None,
) -> bytes:
    """Synthesize a single text chunk and return MP3 bytes."""
    return await _deepgram_tts(text, api_key)


async def synthesize_speech(
    text: str,
    api_key: str,
    language_code: str = "en-IN",
) -> bytes:
    """Convert full text to speech and return merged MP3 bytes."""
    if not api_key:
        raise RuntimeError("DEEPGRAM_API_KEY is not set.")
    parts: list[bytes] = [chunk async for chunk in synthesize_speech_stream(text, api_key, language_code)]
    return b"".join(parts)


async def synthesize_speech_stream(
    text: str,
    api_key: str,
    language_code: str = "en-IN",
) -> AsyncIterator[bytes]:
    """Synthesize speech and yield MP3 chunks in sentence order.

    All sentence chunks are fired to Deepgram in parallel and yielded in
    the original order so the caller can start playing chunk 1 while
    chunks 2, 3, … are still synthesizing in the background.
    """
    if not api_key:
        raise RuntimeError("DEEPGRAM_API_KEY is not set.")

    clean_text = _strip_markdown(text)
    chunks = _split_into_chunks(clean_text)

    tasks = [
        asyncio.ensure_future(_deepgram_tts(chunk, api_key))
        for chunk in chunks
    ]

    total_bytes = 0
    for i, task in enumerate(tasks):
        mp3_bytes = await task
        total_bytes += len(mp3_bytes)
        logger.info("TTS chunk %d/%d ready [bytes=%d]", i + 1, len(tasks), len(mp3_bytes))
        yield mp3_bytes

    logger.info(
        "TTS stream done [model=%s, chunks=%d, text_chars=%d, total_bytes=%d]",
        TTS_MODEL, len(chunks), len(clean_text), total_bytes,
    )
