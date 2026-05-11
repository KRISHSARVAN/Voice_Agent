"""
Suvit Help RAG chat API.

Run (from repo root, venv activated):
    uvicorn app.main:app --host 0.0.0.0 --port 8080 --workers 1
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import uuid
from contextlib import asynccontextmanager

from fastapi import Body, FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from langchain_chroma import Chroma
from starlette.middleware.base import BaseHTTPMiddleware

from motor.motor_asyncio import AsyncIOMotorClient

from app.config import Settings, get_settings
from app.db import ensure_chat_indexes, insert_chat_turn, list_turns_for_session
from app.rag_service import astream_rag_chat, run_rag_chat
from app.schemas import ChatRequest, ChatResponse, ConversationTurn, SourceChunk
from app.stt_service import transcribe_audio
from app.tts_service import pick_speaker, synthesize_chunk, synthesize_speech_stream, _strip_markdown
from rag.embeddings import build_embeddings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.basicConfig(
        level=logging.DEBUG if settings.environment == "development" else logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    for noisy in (
        "pymongo",
        "httpcore",
        "httpx",
        "openai._base_client",
        "python_multipart",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logger.info("Loading embeddings and Chroma collection %s", settings.chroma_collection_name)
    embeddings = build_embeddings(settings.embed_model)
    try:
        vectorstore = Chroma(
            collection_name=settings.chroma_collection_name,
            persist_directory=settings.chroma_persist_directory,
            embedding_function=embeddings,
            create_collection_if_not_exists=False,
        )
    except Exception as e:
        logger.exception("Failed to open Chroma at %s", settings.chroma_persist_directory)
        raise RuntimeError(
            f"Chroma init failed. Run ingest first (scrape_and_store_langchain.py). Details: {e}"
        ) from e

    mongo_client = AsyncIOMotorClient(settings.mongodb_uri)
    mongo_db = mongo_client[settings.mongodb_db]
    mongo_coll = mongo_db[settings.mongodb_chat_collection]
    await ensure_chat_indexes(mongo_coll)
    app.state.settings = settings
    app.state.vectorstore = vectorstore
    app.state.mongo_client = mongo_client
    app.state.mongo_collection = mongo_coll
    logger.info("API ready (environment=%s)", settings.environment)
    try:
        yield
    finally:
        mongo_client.close()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version="1.0.0",
        lifespan=lifespan,
    )

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=["X-Session-Id", "X-Request-ID"],
        )

    class RequestIdMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            rid = request.headers.get("X-Request-ID", str(uuid.uuid4()))
            request.state.request_id = rid
            response = await call_next(request)
            response.headers["X-Request-ID"] = rid
            return response

    app.add_middleware(RequestIdMiddleware)

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": exc.errors(), "request_id": getattr(request.state, "request_id", None)},
        )

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):
        req_id = getattr(request.state, "request_id", None)
        logger.exception("Unhandled error request_id=%s", req_id)
        try:
            settings: Settings = request.app.state.settings
        except AttributeError:
            settings = get_settings()
        msg = str(exc) if settings.environment == "development" else "Internal server error"
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": msg, "request_id": req_id},
        )

    @app.get("/health", tags=["ops"])
    async def health():
        return {"status": "ok"}

    @app.get("/ready", tags=["ops"])
    async def ready(request: Request):
        vs = request.app.state.vectorstore
        try:
            n = vs._collection.count()
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Chroma not ready: {e}",
            ) from e
        if n == 0:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Vector store is empty; run ingest first.",
            )
        return {"status": "ready", "chunk_count": n}

    @app.post("/v1/chat", response_model=ChatResponse, tags=["chat"])
    async def chat(request: Request, body: ChatRequest):
        settings: Settings = request.app.state.settings
        vs = request.app.state.vectorstore
        mongo_coll = request.app.state.mongo_collection
        session_id = body.session_id or str(uuid.uuid4())
        user_text = body.messages[-1].content

        try:
            answer, pairs = await asyncio.to_thread(
                run_rag_chat,
                vectorstore=vs,
                settings=settings,
                messages=[(m.role, m.content) for m in body.messages],
                top_k=body.top_k,
                language_code=body.language_code,
            )
        except ValueError as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
        except Exception as e:
            logger.exception("RAG invocation failed")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=str(e) if settings.environment == "development" else "Upstream LLM failed",
            ) from e

        print(f"\n[USER]  {user_text}")
        print(f"[BOT]   {answer}\n", flush=True)

        try:
            await insert_chat_turn(
                mongo_coll,
                session_id=session_id,
                user_text=user_text,
                bot_response=answer,
            )
        except Exception:
            logger.exception("Failed to persist chat turn session_id=%s", session_id)

        sources: list[SourceChunk] = []
        if body.include_sources:
            for doc, score in pairs:
                md = doc.metadata or {}
                excerpt = (doc.page_content or "")[:420].strip()
                if len(doc.page_content or "") > 420:
                    excerpt += "…"
                sources.append(
                    SourceChunk(
                        title=md.get("title"),
                        source=md.get("source"),
                        content_excerpt=excerpt,
                        chroma_distance=float(score),
                    )
                )

        return ChatResponse(answer=answer, sources=sources, session_id=session_id)

    @app.get(
        "/v1/sessions/{session_id}/turns",
        response_model=list[ConversationTurn],
        tags=["chat"],
    )
    async def session_turns(request: Request, session_id: str, limit: int = 200):
        mongo_coll = request.app.state.mongo_collection
        rows = await list_turns_for_session(mongo_coll, session_id, limit=limit)
        return [
            ConversationTurn(
                id=r.id,
                session_id=r.session_id,
                user_text=r.user_text,
                bot_response=r.bot_response,
                created_at=r.created_at,
            )
            for r in rows
        ]

    @app.post("/v1/transcribe", tags=["stt"])
    async def transcribe(request: Request, file: UploadFile = File(...)):
        """Accept an audio file and return its transcription via Sarvam AI Saaras v3."""
        settings: Settings = request.app.state.settings
        audio_data = await file.read()
        if not audio_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded audio file is empty.",
            )
        try:
            text, language_code = await transcribe_audio(
                audio_data,
                api_key=settings.stt_api_key,
                content_type=file.content_type,
                filename=file.filename,
            )
        except Exception as e:
            logger.exception("Transcription failed")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=str(e) if settings.environment == "development" else "Transcription failed",
            ) from e
        return {"text": text, "language_code": language_code}

    @app.post("/v1/synthesize", tags=["tts"])
    async def synthesize(
        request: Request,
        text: str = Body(..., embed=True),
        language_code: str = Body("en-IN", embed=True),
    ):
        """Convert text to speech using Sarvam AI Bulbul v3.

        Returns a newline-delimited stream of base64-encoded WAV chunks.
        Chunks are synthesized in parallel and streamed in sentence order so
        the client can start playing the first chunk while the rest arrive.
        """
        settings: Settings = request.app.state.settings
        if not text or not text.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="text must not be empty.",
            )

        async def audio_stream():
            try:
                async for wav_chunk in synthesize_speech_stream(
                    text.strip(),
                    api_key=settings.tts_api_key,
                    language_code=language_code,
                ):
                    yield base64.b64encode(wav_chunk) + b"\n"
            except Exception:
                logger.exception("TTS synthesis failed mid-stream")

        return StreamingResponse(audio_stream(), media_type="text/plain")

    @app.post("/v1/voice", tags=["voice"])
    async def voice_pipeline(request: Request, body: ChatRequest):
        """Real-time voice pipeline: LLM streams tokens → sentences fire TTS immediately.

        Response: newline-delimited stream where most lines are base64 WAV audio chunks
        and the final line is a JSON metadata object prefixed with ``data:``.
        The metadata line carries ``session_id`` and ``answer`` so the client can
        update its session and detect farewell phrases without a second request.
        """
        settings: Settings = request.app.state.settings
        vs = request.app.state.vectorstore
        mongo_coll = request.app.state.mongo_collection
        session_id = body.session_id or str(uuid.uuid4())
        user_text = body.messages[-1].content

        if not settings.tts_api_key:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="TTS is not configured on this server.",
            )

        pairs_out: list = []
        answer_parts: list[str] = []

        # Sentence boundary pattern — split on punctuation followed by whitespace.
        _sentence_re = re.compile(r'(?<=[.!?।])\s+')

        async def audio_stream():
            speaker = pick_speaker()
            sentence_buf = ""
            tts_tasks: list[asyncio.Task] = []
            yielded_count = 0

            async def _queue_tts(text: str) -> None:
                clean = _strip_markdown(text).strip()
                if clean:
                    task = asyncio.ensure_future(
                        synthesize_chunk(clean, settings.tts_api_key, body.language_code, speaker)
                    )
                    tts_tasks.append(task)

            try:
                async for token in astream_rag_chat(
                    vectorstore=vs,
                    settings=settings,
                    messages=[(m.role, m.content) for m in body.messages],
                    top_k=body.top_k,
                    language_code=body.language_code,
                    pairs_out=pairs_out,
                ):
                    answer_parts.append(token)
                    sentence_buf += token

                    # Fire TTS for every complete sentence detected so far.
                    parts = _sentence_re.split(sentence_buf)
                    if len(parts) > 1:
                        for sentence in parts[:-1]:
                            await _queue_tts(sentence)
                        sentence_buf = parts[-1]

                    # Yield any TTS chunks that finished (in order, no waiting).
                    while yielded_count < len(tts_tasks) and tts_tasks[yielded_count].done():
                        try:
                            wav = tts_tasks[yielded_count].result()
                            yield base64.b64encode(wav) + b"\n"
                        except Exception:
                            logger.exception("TTS chunk %d failed, skipping", yielded_count)
                        yielded_count += 1

            except Exception:
                logger.exception("LLM streaming failed in voice pipeline")

            # Flush the final sentence fragment (no trailing punctuation).
            if sentence_buf.strip():
                await _queue_tts(sentence_buf)

            # Await and yield remaining TTS tasks in order.
            for i in range(yielded_count, len(tts_tasks)):
                try:
                    wav = await tts_tasks[i]
                    yield base64.b64encode(wav) + b"\n"
                except Exception:
                    logger.exception("TTS chunk %d failed, skipping", i)

            # Final metadata line so the client gets session_id + answer text.
            full_answer = "".join(answer_parts)
            meta = json.dumps({"session_id": session_id, "answer": full_answer})
            yield b"data:" + meta.encode() + b"\n"

            # Persist to MongoDB after all audio is sent.
            try:
                await insert_chat_turn(
                    mongo_coll,
                    session_id=session_id,
                    user_text=user_text,
                    bot_response=full_answer,
                )
            except Exception:
                logger.exception("Failed to persist voice turn session_id=%s", session_id)

            print(f"\n[USER]  {user_text}")
            print(f"[BOT]   {full_answer}\n", flush=True)

        return StreamingResponse(
            audio_stream(),
            media_type="text/plain",
            headers={"X-Session-Id": session_id},
        )

    return app


app = create_app()
