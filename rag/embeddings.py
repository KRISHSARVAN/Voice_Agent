from __future__ import annotations

import hashlib
import logging
import re
from typing import Any

try:
    import certifi_win32  # type: ignore  # noqa: F401
except Exception:  # pragma: no cover
    pass

try:
    from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
except Exception:  # pragma: no cover
    from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore

from rag.store_config import EMBED_MODEL, FALLBACK_EMBED_DIM

logger = logging.getLogger(__name__)


class LocalHashEmbeddings:
    """Deterministic offline embedding fallback (must match ingest if HF fails)."""

    def __init__(self, dimension: int = FALLBACK_EMBED_DIM) -> None:
        self.dimension = dimension

    def _embed_text(self, text: str) -> list[float]:
        tokens = re.findall(r"\w+", text.lower())
        vec = [0.0] * self.dimension
        if not tokens:
            return vec
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dimension
            vec[idx] += 1.0
        norm = sum(v * v for v in vec) ** 0.5
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_text(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_text(text)


def build_embeddings(embed_model: str | None = None) -> Any:
    model_name = embed_model or EMBED_MODEL
    try:
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception as e:
        logger.warning(
            "HuggingFace embeddings unavailable (%s); using LocalHashEmbeddings",
            e,
        )
        return LocalHashEmbeddings()
