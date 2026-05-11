"""Cartesia Ink Whisper speech-to-text service."""

from __future__ import annotations

import asyncio
import io
import logging

logger = logging.getLogger(__name__)

# ISO-639-1 codes returned by Cartesia → BCP-47 codes used by the rest of the app.
_CARTESIA_LANG_TO_BCP47: dict[str, str] = {
    "hi": "hi-IN",
    "gu": "gu-IN",
    "en": "en-IN",
    "en-US": "en-IN",
    "en-GB": "en-IN",
    "en-IN": "en-IN",
    "en-AU": "en-IN",
}

_STT_MODEL = "ink-whisper"

_MIME_TO_EXT: dict[str, str] = {
    "audio/webm": "webm",
    "audio/ogg": "ogg",
    "audio/wav": "wav",
    "audio/mpeg": "mp3",
    "audio/mp4": "mp4",
    "video/webm": "webm",
}


def _clean_mime(content_type: str | None) -> str:
    """Return a bare MIME type (strip codec params, normalise video/webm)."""
    if not content_type:
        return "audio/webm"
    base = content_type.split(";")[0].strip().lower()
    if base == "video/webm":
        return "audio/webm"
    return base or "audio/webm"


def _run_cartesia_stt(audio_data: bytes, api_key: str, mimetype: str) -> tuple[str, str]:
    """Call Cartesia Ink Whisper STT and return (transcript, BCP-47 language_code)."""
    try:
        from cartesia import Cartesia  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "cartesia not installed. Run: pip install cartesia"
        ) from exc

    ext = _MIME_TO_EXT.get(mimetype, "webm")

    # Wrap bytes in a BytesIO and attach a name so the SDK can infer the format.
    audio_file = io.BytesIO(audio_data)
    audio_file.name = f"recording.{ext}"  # type: ignore[attr-defined]

    client = Cartesia(api_key=api_key)
    response = client.stt.transcribe(
        file=audio_file,
        model=_STT_MODEL,
        # Language omitted — Cartesia auto-detects it from the audio.
    )

    transcript = (response.text or "").strip()
    raw_lang: str = getattr(response, "language", None) or "en"
    language_code = _CARTESIA_LANG_TO_BCP47.get(raw_lang, "en-IN")

    return transcript, language_code


async def transcribe_audio(
    audio_data: bytes,
    api_key: str,
    content_type: str | None = None,
    filename: str | None = None,
) -> tuple[str, str]:
    """Transcribe audio bytes using Cartesia Ink Whisper.

    Args:
        audio_data: Raw audio bytes from the browser (WebM/Opus, OGG, WAV …).
        api_key: Cartesia API key.
        content_type: MIME type of the upload — used to choose the file extension
                      hint sent to Cartesia (e.g. audio/webm → recording.webm).
        filename: Unused; kept for API compatibility.

    Returns:
        Tuple of (transcribed text, detected BCP-47 language_code e.g. "hi-IN").
    """
    if not api_key:
        raise RuntimeError(
            "CARTESIA_API_KEY is not set. Add your Cartesia API key to .env."
        )

    mimetype = _clean_mime(content_type)
    text, language_code = await asyncio.to_thread(
        _run_cartesia_stt, audio_data, api_key, mimetype
    )
    logger.info(
        "Transcription done [model=%s, chars=%d, lang=%s, mime=%s]",
        _STT_MODEL,
        len(text),
        language_code,
        mimetype,
    )
    return text, language_code
