"""Sarvam AI Bulbul v3 text-to-speech service."""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import random
import re
import wave
from typing import AsyncIterator

logger = logging.getLogger(__name__)

TTS_MODEL = "bulbul:v3"
TTS_DEFAULT_LANGUAGE = "en-IN"
_SUPPORTED_LANGUAGES = {"en-IN", "gu-IN", "hi-IN"}
_MAX_CHUNK_CHARS = 500  # Sarvam TTS per-request character limit

# All speakers supported by bulbul:v3
_SPEAKERS = [
    "aditya", "ritu", "ashutosh", "priya", "neha", "rahul", "pooja", "rohan",
    "simran", "kavya", "amit", "dev", "ishita", "shreya", "ratan", "varun",
    "manan", "sumit", "roopa", "kabir", "aayan", "shubh", "advait", "anand",
    "tanya", "tarun", "sunny", "mani", "gokul", "vijay", "shruti", "suhani",
    "mohit", "kavitha", "rehan", "soham", "rupali",
]


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
            # Force-split a sentence that is itself too long
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


def _run_sarvam_tts_chunk(text: str, api_key: str, language_code: str, speaker: str) -> bytes:
    """Call Sarvam TTS API for a single text chunk."""
    try:
        from sarvamai import SarvamAI  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "sarvamai package not installed. Run: pip install sarvamai"
        ) from exc

    client = SarvamAI(api_subscription_key=api_key)
    result = client.text_to_speech.convert(
        text=text,
        target_language_code=language_code,
        speaker=speaker,
        model=TTS_MODEL,
        enable_preprocessing=True,
    )
    if not result.audios:
        raise RuntimeError("No audio returned from Sarvam AI TTS")
    return base64.b64decode(result.audios[0])


def pick_speaker() -> str:
    """Return a random speaker name for a TTS session."""
    return random.choice(_SPEAKERS)


async def synthesize_chunk(
    text: str,
    api_key: str,
    language_code: str = TTS_DEFAULT_LANGUAGE,
    speaker: str | None = None,
) -> bytes:
    """Synthesize a single text chunk asynchronously and return WAV bytes."""
    lang = language_code if language_code in _SUPPORTED_LANGUAGES else TTS_DEFAULT_LANGUAGE
    spk = speaker or random.choice(_SPEAKERS)
    return await asyncio.to_thread(_run_sarvam_tts_chunk, text, api_key, lang, spk)


async def synthesize_speech(text: str, api_key: str, language_code: str = TTS_DEFAULT_LANGUAGE) -> bytes:
    """Convert text to speech — returns a single merged WAV (used for testing/fallback)."""
    if not api_key:
        raise RuntimeError(
            "STT_API_KEY is not set. Add your Sarvam AI API subscription key to .env."
        )
    parts: list[bytes] = [chunk async for chunk in synthesize_speech_stream(text, api_key, language_code)]
    return _combine_wav(parts)


async def synthesize_speech_stream(
    text: str,
    api_key: str,
    language_code: str = TTS_DEFAULT_LANGUAGE,
) -> AsyncIterator[bytes]:
    """Synthesize speech and yield WAV chunks as they become ready (in text order).

    All chunks are fired to Sarvam in parallel. They are yielded in the
    original sentence order so the caller can start playing chunk 1 while
    chunks 2, 3, … are still being synthesized in the background.
    """
    if not api_key:
        raise RuntimeError(
            "STT_API_KEY is not set. Add your Sarvam AI API subscription key to .env."
        )

    lang = language_code if language_code in _SUPPORTED_LANGUAGES else TTS_DEFAULT_LANGUAGE
    clean_text = _strip_markdown(text)
    chunks = _split_into_chunks(clean_text)
    speaker = random.choice(_SPEAKERS)

    # Fire all Sarvam requests at the same time
    tasks = [
        asyncio.ensure_future(
            asyncio.to_thread(_run_sarvam_tts_chunk, chunk, api_key, lang, speaker)
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
        "TTS stream done [model=%s, speaker=%s, lang=%s, chunks=%d, text_chars=%d, total_bytes=%d]",
        TTS_MODEL, speaker, lang, len(chunks), len(clean_text), total_bytes,
    )
