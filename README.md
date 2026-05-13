# Suvit Voice Agent

A real-time voice-enabled RAG (Retrieval-Augmented Generation) chatbot for [Suvit](https://suvit.io) customer support. Users can speak questions about Suvit's accounting automation platform and receive answers - spoken aloud in Indian languages - backed by content scraped from [help.suvit.io](https://help.suvit.io).

---

## Architecture

```
Browser 
  │
  │  WebSocket  /ws/voice  (persistent; full-duplex during a call)
  │     │
  │     ├─►  JSON: transcribe (base64 WAV) ──► Sarvam Saaras v3 (STT)
  │     ├─►  JSON: synthesize (welcome / short TTS) ──► binary WAV frames
  │     └─►  JSON: query (user text) ──┬──► ChromaDB vector search
  │                                    ├──► OpenAI (streaming tokens → client)
  │                                    └──► Sarvam Bulbul v3 (TTS) → binary WAV
  │
  └── Chat history persisted in MongoDB
```

**Key services**


| Service        | Model                        | Role                                             |
| -------------- | ---------------------------- | ------------------------------------------------ |
| Speech-to-Text | Sarvam Saaras v3             | Transcribes microphone audio; detects language   |
| LLM            | OpenAI GPT-4o-mini (default) | Generates answers from retrieved context         |
| Text-to-Speech | Sarvam Bulbul v3             | Synthesises spoken responses in Indian languages |
| Vector DB      | ChromaDB                     | Stores and retrieves help article chunks         |
| Chat history   | MongoDB                      | Persists user/bot turns per session              |


---

## Project structure

```
Voice_agent/
├── app/
│   ├── main.py           # FastAPI app — all API endpoints
│   ├── config.py         # Settings loaded from .env
│   ├── rag_service.py    # RAG retrieval + LLM chain
│   ├── stt_service.py    # Sarvam Saaras v3 speech-to-text
│   ├── tts_service.py    # Sarvam Bulbul v3 text-to-speech
│   ├── db.py             # MongoDB helpers
│   ├── schemas.py        # Pydantic request/response models
│   └── ws_voice.py       # WebSocket protocol: STT, TTS, RAG + streaming replies
├── rag/
│   ├── embeddings.py     # HuggingFace embedding builder
│   └── store_config.py   # ChromaDB path / collection constants
├── frontend/             # React + TypeScript + Vite UI
│   ├── src/App.tsx       # Main voice chat interface
│   └── src/App.css       # Styles
├── scrape_and_store_langchain.py  # One-time help-site ingestion script
├── chroma_db/            # Persisted ChromaDB (git-ignored)
├── requirements.txt
└── .env                  # Secrets (git-ignored; create locally)
```

---

## Prerequisites

- Python 3.10+
- Node.js 18+ (for the frontend)
- MongoDB (local or Atlas)
- [Sarvam AI](https://sarvam.ai) API subscription key (STT and/or TTS; voice pipeline requires TTS)
- OpenAI API key

---

## Setup

### 1. Clone and create a virtual environment

```bash
git clone <repo-url>
cd Voice_agent
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

On Windows, `requirements.txt` includes `python-certifi-win32` so HTTPS clients resolve certificates reliably.

### 3. Environment variables

Create a `.env` file at the project root. Pydantic loads these names (case-insensitive).

**Required**


| Variable         | Description                       |
| ---------------- | --------------------------------- |
| `OPENAI_API_KEY` | OpenAI API key for the chat model |


**Strongly recommended for full functionality**


| Variable           | Default   | Description                                                                        |
| ------------------ | --------- | ---------------------------------------------------------------------------------- |
| `STT_API_KEY`      | *(empty)* | Sarvam key used by the WebSocket transcribe path (`/ws/voice`)                     |
| `TTS_API_KEY`      | *(empty)* | Sarvam key for WebSocket synthesize and query TTS (voice returns error if missing) |
| `CORS_ORIGINS_RAW` | *(empty)* | Comma-separated browser origins; if empty, CORS middleware is not added            |


**Database and vector store**


| Variable                   | Default                     | Description                                                                              |
| -------------------------- | --------------------------- | ---------------------------------------------------------------------------------------- |
| `MONGODB_URI`              | `mongodb://localhost:27017` | MongoDB connection URI                                                                   |
| `MONGODB_DB`               | `suvit_voice`               | Database name                                                                            |
| `MONGODB_CHAT_COLLECTION`  | `chat_turns`                | Collection for chat turns                                                                |
| `CHROMA_PERSIST_DIRECTORY` | `./chroma_db`               | Chroma persist path (must match ingest)                                                  |
| `CHROMA_COLLECTION_NAME`   | `suvit_help`                | Collection name (must match ingest)                                                      |
| `EMBED_MODEL`              | `BAAI/bge-small-en-v1.5`    | Sentence-transformers model; **must match** what you used when running the ingest script |


**OpenAI and RAG tuning**


| Variable                | Default       | Description                                         |
| ----------------------- | ------------- | --------------------------------------------------- |
| `OPENAI_MODEL`          | `gpt-4o-mini` | Chat completion model                               |
| `OPENAI_TEMPERATURE`    | `0.1`         | Sampling temperature                                |
| `OPENAI_TIMEOUT_S`      | `120`         | Request timeout (seconds)                           |
| `RAG_TOP_K`             | `6`           | Default retrieval `k` when the client omits `top_k` |
| `RAG_MAX_CONTEXT_CHARS` | `12000`       | Cap on context size passed to the model             |


**App**


| Variable      | Default       | Description                                         |
| ------------- | ------------- | --------------------------------------------------- |
| `ENVIRONMENT` | `development` | Use `development` for verbose errors and DEBUG logs |


Minimal example:

```env
OPENAI_API_KEY=sk-...
STT_API_KEY=your-sarvam-key
TTS_API_KEY=your-sarvam-key

MONGODB_URI=
MONGODB_DB=
MONGODB_CHAT_COLLECTION=

CHROMA_PERSIST_DIRECTORY=
CHROMA_COLLECTION_NAME=

ENVIRONMENT=development
CORS_ORIGINS_RAW=
```

### 4. Ingest the help articles (one-time)

This script crawls [help.suvit.io](https://help.suvit.io), chunks the content, and stores embeddings in ChromaDB. Run it once (or again whenever the help docs change):

```bash
python scrape_and_store_langchain.py
```

Expected output: `ChromaDB now has N chunks` in `./chroma_db/`.

### 5. Start the backend

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8080 --workers 1
```

Check readiness:

```bash
curl http://localhost:8080/ready
# {"status":"ready","chunk_count":1234}
```

Interactive API docs: [http://localhost:8080/docs](http://localhost:8080/docs)

### 6. Start the frontend

```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173) in your browser.

---

## API reference

### Cross-cutting behaviour

- **Request ID:** Send optional header `X-Request-ID`; the server echoes it on the response. If omitted, a UUID is generated. Validation and unhandled errors include `request_id` in the JSON body when possible.
- **OpenAPI:** `GET /docs` (Swagger UI), `GET /redoc` (ReDoc).

### Health and readiness


| Method | Path      | Description                                                                                  |
| ------ | --------- | -------------------------------------------------------------------------------------------- |
| `GET`  | `/health` | Returns `{"status":"ok"}`                                                                    |
| `GET`  | `/ready`  | Returns `{"status":"ready","chunk_count":N}`; **503** if Chroma is missing, broken, or empty |


### Chat (text)

`POST /v1/chat`

The **last** message in `messages` must have `role: "user"`. Omitting `top_k` uses `RAG_TOP_K` from settings (default 6).

```json
{
  "messages": [
    {"role": "user", "content": "How do I upload a bank statement?"}
  ],
  "session_id": "optional-uuid",
  "language_code": "hi-IN",
  "top_k": 6,
  "include_sources": true
}
```

`include_sources` defaults to `**true**` in the API schema; set it to `false` if you do not need `sources` in the response.

Response shape: `answer`, `sources` (list of excerpts and metadata when enabled), `session_id`.

---

## Supported languages


| Language        | Code    |
| --------------- | ------- |
| English (India) | `en-IN` |
| Hindi           | `hi-IN` |


The STT model auto-detects the spoken language. For Hindi and Gujarati, the RAG layer translates the query to English for retrieval, then instructs the LLM to answer in the user's language.

---

## Frontend features

- **Push-to-talk** and **auto-silence detection** (stops recording after ~1.8 s of silence)
- **Streaming audio playback** — starts playing the first sentence while the rest is still being synthesised
- **Session persistence** via `sessionStorage` — conversation history survives page refreshes
- **Farewell detection** — detects goodbye phrases in user speech and bot replies to end the call gracefully
- **Language-aware responses** — Hindi/Gujarati questions receive Hindi/Gujarati spoken answers

---

## Development notes

- Use **one** Uvicorn worker (`--workers 1`) when using a local Chroma persist directory; multiple processes are not safe for shared file-backed Chroma.
- TTS sentences are synthesised **in parallel** and yielded in order to reduce perceived latency.
- MongoDB indexes are ensured on startup via `ensure_chat_indexes`.
- Set `ENVIRONMENT=development` to surface full error `detail` in some 5xx paths and to enable DEBUG logging.

