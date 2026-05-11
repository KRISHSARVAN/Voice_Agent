"""Cartesia Sonic text-to-speech service."""

from __future__ import annotations

import asyncio
import io
import logging
import random
import re
import wave
from typing import AsyncIterator

logger = logging.getLogger(__name__)

TTS_DEFAULT_LANGUAGE = "en-IN"

# Map BCP-47 language codes → Cartesia language identifiers.
_BCP47_TO_CARTESIA_LANG: dict[str, str] = {
    "en-IN": "en",
    "en-US": "en",
    "en-GB": "en",
    "hi-IN": "hi",
    "gu-IN": "gu",
}

# Cartesia voice IDs recommended for voice agents (stable, realistic voices).
# Browse and preview more voices at https://play.cartesia.ai/voices
_VOICES: list[dict] = [
    {"id": "f786b574-daa5-4673-aa0c-cbe3e8534c02", "name": "Katie"},       # female, American English
    {"id": "a5136bf9-224c-4d76-b823-52bd5efcffcc", "name": "Jameson"},     # male, American English
    {"id": "694f9389-aac1-45b6-b726-9d9369183238", "name": "Barbershop Man"},  # male
    {"id": "b7d50908-b17c-442d-ad8d-810c63997ed9", "name": "California Girl"}, # female
    {"id": "c2ac25f9-ecc4-4f56-9095-651354df60c0", "name": "Helpful Woman"},   # female
    {"id": "eda5bbff-1ff1-4886-8ef1-4e69a0e29b80", "name": "Friendly Brazilian Man"}, # male
]

# Cartesia model — use base name to always stay on the latest stable snapshot.
_TTS_MODEL = "sonic-3.5"

# Keep each TTS request under 2000 chars.
_MAX_CHUNK_CHARS = 2000


def _strip_markdown(text: str) -> str:
    """Remove markdown so TTS reads clean prose (no asterisks, brackets, etc.)."""
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


def _combine_wav(wav_parts: list[bytes]) -> bytes:
    """Concatenate multiple same-format WAV byte strings into one."""
    if len(wav_parts) == 1:
        return wav_parts[0]
    all_frames: list[bytes] = []
    params = None
    for part in wav_parts:
        with wave.open(io.BytesIO(part), "rb") as wf:
            if params is None:
                params = wf.getparams()
            all_frames.append(wf.readframes(wf.getnframes()))
    out = io.BytesIO()
    with wave.open(out, "wb") as wf:
        wf.setparams(params)  # type: ignore[arg-type]
        for frames in all_frames:
            wf.writeframes(frames)
    return out.getvalue()


def _run_cartesia_tts_chunk(text: str, api_key: str, voice_id: str, language: str = "en") -> bytes:
    """Call Cartesia Sonic TTS for a single text chunk and return WAV bytes."""
    try:
        from cartesia import Cartesia  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "cartesia not installed. Run: pip install cartesia"
        ) from exc

    client = Cartesia(api_key=api_key)
    chunks = client.tts.bytes(  # type: ignore[reportDeprecated]
        model_id=_TTS_MODEL,
        transcript=text,
        voice={"mode": "id", "id": voice_id},
        language=language,
        output_format={
            "container": "wav",
            "encoding": "pcm_s16le",
            "sample_rate": 22050,
        },
    )
    return b"".join(chunks)


def pick_speaker() -> str:
    """Return a random Cartesia voice ID for a TTS session."""
    return random.choice(_VOICES)["id"]


async def synthesize_chunk(
    text: str,
    api_key: str,
    language_code: str = TTS_DEFAULT_LANGUAGE,
    speaker: str | None = None,
) -> bytes:
    """Synthesize a single text chunk asynchronously and return WAV bytes."""
    voice_id = speaker or random.choice(_VOICES)["id"]
    language = _BCP47_TO_CARTESIA_LANG.get(language_code, "en")
    return await asyncio.to_thread(_run_cartesia_tts_chunk, text, api_key, voice_id, language)


async def synthesize_speech(
    text: str,
    api_key: str,
    language_code: str = TTS_DEFAULT_LANGUAGE,
) -> bytes:
    """Convert text to speech — returns a single merged WAV (used for testing/fallback)."""
    if not api_key:
        raise RuntimeError(
            "CARTESIA_API_KEY is not set. Add your Cartesia API key to .env."
        )
    parts: list[bytes] = [chunk async for chunk in synthesize_speech_stream(text, api_key, language_code)]
    return _combine_wav(parts)


async def synthesize_speech_stream(
    text: str,
    api_key: str,
    language_code: str = TTS_DEFAULT_LANGUAGE,
) -> AsyncIterator[bytes]:
    """Synthesize speech and yield WAV chunks as they become ready (in text order).

    All chunks are fired to Cartesia in parallel. They are yielded in the
    original sentence order so the caller can start playing chunk 1 while
    chunks 2, 3, … are still being synthesized in the background.
    """
    if not api_key:
        raise RuntimeError(
            "CARTESIA_API_KEY is not set. Add your Cartesia API key to .env."
        )

    clean_text = _strip_markdown(text)
    chunks = _split_into_chunks(clean_text)
    voice_id = random.choice(_VOICES)["id"]

    tasks = [
        asyncio.ensure_future(
            asyncio.to_thread(_run_cartesia_tts_chunk, chunk, api_key, voice_id)
        )
        for chunk in chunks
    ]

    total_bytes = 0
    for i, task in enumerate(tasks):
        wav_bytes = await task
        total_bytes += len(wav_bytes)
        logger.info("TTS chunk %d/%d ready [bytes=%d]", i + 1, len(tasks), len(wav_bytes))
        yield wav_bytes

    logger.info(
        "TTS stream done [model=%s, voice_id=%s, chunks=%d, text_chars=%d, total_bytes=%d]",
        _TTS_MODEL,
        voice_id,
        len(chunks),
        len(clean_text),
        total_bytes,
    )
