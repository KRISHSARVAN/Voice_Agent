"""
Suvit Help RAG chat API.

Run (from repo root, venv activated):
    uvicorn app.main:app --host 0.0.0.0 --port 8080 --workers 1
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, WebSocket, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langchain_chroma import Chroma
from starlette.middleware.base import BaseHTTPMiddleware

from motor.motor_asyncio import AsyncIOMotorClient

from app.config import Settings, get_settings
from app.db import ensure_chat_indexes, insert_chat_turn, list_turns_for_session
from app.rag_service import run_rag_chat
from app.schemas import ChatRequest, ChatResponse, ConversationTurn, SourceChunk
from app.ws_voice import handle_ws_voice
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

    @app.websocket("/ws/voice")
    async def ws_voice(websocket: WebSocket):
        """Persistent WebSocket voice pipeline.

        Replaces per-turn HTTP POST + StreamingResponse with a single long-lived
        connection: binary WAV frames for audio (no base64), JSON frames for LLM
        tokens and metadata, and instant interrupt support.
        """
        await handle_ws_voice(websocket)

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

    return app


app = create_app()
