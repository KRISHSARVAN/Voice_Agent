"""Sarvam AI Saaras v3 speech-to-text service."""

from __future__ import annotations

import asyncio
import logging
from io import BytesIO

logger = logging.getLogger(__name__)

_MIME_TO_EXT: dict[str, str] = {
    "audio/webm": "webm",
    "audio/ogg": "ogg",
    "audio/wav": "wav",
    "audio/mpeg": "mp3",
    "audio/mp4": "mp4",
    "video/webm": "webm",
}

def _ext_from_content_type(content_type: str | None) -> str:
    if content_type:
        ct = content_type.split(";")[0].strip().lower()
        if ct in _MIME_TO_EXT:
            return _MIME_TO_EXT[ct]
    return "webm"

def _run_sarvam(audio_data: bytes, api_key: str, filename: str) -> tuple[str, str]:
    try:
        from sarvamai import SarvamAI  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "sarvamai package not installed. Run: pip install sarvamai"
        ) from exc

    client = SarvamAI(api_subscription_key=api_key)
    buf = BytesIO(audio_data)
    buf.name = filename
    result = client.speech_to_text.transcribe(
        file=buf,
        model="saaras:v3",
        mode="transcribe",
    )
    transcript = (result.transcript or "").strip()
    language_code = getattr(result, "language_code", None) or "en-IN"
    return transcript, language_code

async def transcribe_audio(
    audio_data: bytes,
    api_key: str,
    content_type: str | None = None,
    filename: str | None = None,
) -> tuple[str, str]:
    """Transcribe audio bytes using Sarvam AI Saaras v3.

    Args:
        audio_data: Raw audio bytes from the browser (WebM/Opus, OGG, WAV …).
        api_key: Sarvam AI API subscription key.
        content_type: MIME type of the upload (used to pick file extension).
        filename: Explicit filename override.

    Returns:
        Tuple of (transcribed text, detected language_code e.g. "hi-IN").
    """
    if not api_key:
        raise RuntimeError(
            "STT_API_KEY is not set. Add your Sarvam AI API subscription key to .env."
        )

    # WAV header is 44 bytes; anything under 1 KB is too short to contain speech
    if len(audio_data) < 1024:
        logger.warning("Audio too short (%d bytes), skipping transcription", len(audio_data))
        return "", "en-IN"
        
    ext = _ext_from_content_type(content_type)
    resolved_name = filename or f"recording.{ext}"

    text, language_code = await asyncio.to_thread(_run_sarvam, audio_data, api_key, resolved_name)
    logger.info("Transcription done [model=saaras:v3, chars=%d, lang=%s]", len(text), language_code)
    return text, language_code
