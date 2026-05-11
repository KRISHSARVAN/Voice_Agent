"""Deepgram speech-to-text service (flux-general-multi)."""

from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)

STT_MODEL = "flux-general-multi"

_MIME_MAP: dict[str, str] = {
    "audio/webm": "audio/webm",
    "audio/ogg": "audio/ogg",
    "audio/wav": "audio/wav",
    "audio/mpeg": "audio/mpeg",
    "audio/mp4": "audio/mp4",
    "video/webm": "audio/webm",
}

_LANG_NORMALIZE: dict[str, str] = {
    "en": "en-IN",
    "hi": "hi-IN",
    "gu": "gu-IN",
    "ta": "ta-IN",
    "te": "te-IN",
    "mr": "mr-IN",
    "bn": "bn-IN",
    "kn": "kn-IN",
}


def _resolve_mimetype(content_type: str | None) -> str:
    if content_type:
        ct = content_type.split(";")[0].strip().lower()
        return _MIME_MAP.get(ct, "audio/webm")
    return "audio/webm"


async def transcribe_audio(
    audio_data: bytes,
    api_key: str,
    content_type: str | None = None,
    filename: str | None = None,
) -> tuple[str, str]:
    """Transcribe audio bytes using Deepgram flux-general-multi.

    Args:
        audio_data: Raw audio bytes (WebM/Opus, OGG, WAV, …).
        api_key: Deepgram API key.
        content_type: MIME type of the upload.
        filename: Unused; kept for API compatibility.

    Returns:
        Tuple of (transcribed text, detected language code e.g. "hi-IN").
    """
    if not api_key:
        raise RuntimeError("DEEPGRAM_API_KEY is not set.")

    mimetype = _resolve_mimetype(content_type)

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            "https://api.deepgram.com/v2/listen",
            headers={
                "Authorization": f"Token {api_key}",
                "Content-Type": mimetype,
            },
            content=audio_data,
            params={
                "model": STT_MODEL,
                "smart_format": "true",
                "detect_language": "true",
            },
        )
        response.raise_for_status()
        data = response.json()

    channel = data["results"]["channels"][0]
    alternatives = channel.get("alternatives", [{}])
    transcript = (alternatives[0].get("transcript", "") if alternatives else "").strip()

    raw_lang = channel.get("detected_language", "en")
    language_code = _LANG_NORMALIZE.get(raw_lang, raw_lang) if "-" not in raw_lang else raw_lang

    logger.info(
        "Transcription done [model=%s, chars=%d, lang=%s]",
        STT_MODEL, len(transcript), language_code,
    )
    return transcript, language_code
