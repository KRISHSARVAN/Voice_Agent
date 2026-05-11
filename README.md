# Suvit Voice Agent

A real-time voice-enabled RAG (Retrieval-Augmented Generation) chatbot for [Suvit](https://suvit.io) customer support. Users can speak or type questions about Suvit's accounting automation platform and receive answers — spoken aloud in Indian languages — backed by content scraped from [help.suvit.io](https://help.suvit.io).

---

## Architecture

```
Browser (React + Vite)
  │
  │  WebM audio  ──►  POST /v1/voice  ──►  Sarvam Saaras v3 (STT)
  │                        │
  │                        ▼
  │               ChromaDB vector search  (BAAI/bge-small-en-v1.5)
  │                        │
  │                        ▼
  │                  OpenAI GPT-4o-mini  (streaming tokens)
  │                        │
  │                        ▼
  │  base64 WAV  ◄──  Sarvam Bulbul v3 (TTS, parallel sentence synthesis)
  │
  └── Chat history persisted in MongoDB
```

**Key services**

| Service | Model | Role |
|---|---|---|
| Speech-to-Text | Sarvam Saaras v3 | Transcribes microphone audio; detects language |
| LLM | OpenAI GPT-4o-mini | Generates answers from retrieved context |
| Text-to-Speech | Sarvam Bulbul v3 | Synthesises spoken responses in Indian languages |
| Vector DB | ChromaDB | Stores and retrieves help article chunks |
| Chat history | MongoDB | Persists user/bot turns per session |

---

## Project Structure

```
Voice_agent/
├── app/
│   ├── main.py           # FastAPI app — all API endpoints
│   ├── config.py         # Settings loaded from .env
│   ├── rag_service.py    # RAG retrieval + LLM chain
│   ├── stt_service.py    # Sarvam Saaras v3 speech-to-text
│   ├── tts_service.py    # Sarvam Bulbul v3 text-to-speech
│   ├── db.py             # MongoDB helpers
│   └── schemas.py        # Pydantic request/response models
├── rag/
│   ├── embeddings.py     # HuggingFace embedding builder
│   └── store_config.py   # ChromaDB path / collection constants
├── frontend/             # React + Vite UI
│   ├── src/App.tsx       # Main voice chat interface
│   └── src/App.css       # Styles
├── scrape_and_store_langchain.py  # One-time help-site ingestion script
├── chroma_db/            # Persisted ChromaDB (git-ignored)
├── requirements.txt
└── .env                  # Secrets (git-ignored)
```

---

## Prerequisites

- Python 3.10+
- Node.js 18+ (for the frontend)
- MongoDB (local or Atlas)
- [Sarvam AI](https://sarvam.ai) API subscription key
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

### 3. Configure environment variables

Create a `.env` file at the project root (copy from `.env.example` if available):

```env
# OpenAI
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini

# Sarvam AI  (same key works for both STT and TTS, or use separate keys)
STT_API_KEY=your-sarvam-key
TTS_API_KEY=your-sarvam-key

# MongoDB
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB=suvit_voice
MONGODB_CHAT_COLLECTION=chat_turns

# ChromaDB
CHROMA_PERSIST_DIRECTORY=./chroma_db
CHROMA_COLLECTION_NAME=suvit_help

# App
ENVIRONMENT=development
CORS_ORIGINS_RAW=http://localhost:5173
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

Verify it is ready:

```bash
curl http://localhost:8080/ready
# {"status":"ready","chunk_count":1234}
```

### 6. Start the frontend

```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173) in your browser.

---

## API Reference

### Health & readiness

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Returns `{"status":"ok"}` always |
| `GET` | `/ready` | Returns chunk count; 503 if ChromaDB is empty |

### Chat (text)

```
POST /v1/chat
```

```json
{
  "messages": [{"role": "user", "content": "How do I upload a bank statement?"}],
  "session_id": "optional-uuid",
  "language_code": "hi-IN",
  "top_k": 6,
  "include_sources": false
}
```

### Speech-to-Text

```
POST /v1/transcribe
Content-Type: multipart/form-data
```

Upload an audio file (`file` field). Returns:

```json
{"text": "स्टेटमेंट कैसे अपलोड करें?", "language_code": "hi-IN"}
```

### Text-to-Speech

```
POST /v1/synthesize
```

```json
{"text": "Hello!", "language_code": "en-IN"}
```

Returns a newline-delimited stream of base64-encoded WAV chunks.

### Voice pipeline (STT + RAG + TTS in one call)

```
POST /v1/voice
```

Accepts the same body as `/v1/chat`. Returns a streaming response where each line is either a base64 WAV audio chunk or a final `data:{...}` JSON line containing `session_id` and the full `answer` text.

### Chat history

```
GET /v1/sessions/{session_id}/turns?limit=200
```

Returns all stored user/bot turns for a session.

---

## Supported Languages

| Language | Code |
|---|---|
| English (India) | `en-IN` |
| Hindi | `hi-IN` |
| Gujarati | `gu-IN` |

The STT model auto-detects the spoken language. For non-English queries, the RAG service translates the query to English before vector search, then instructs the LLM to respond in the user's language.

---

## Frontend Features

- **Push-to-talk** and **auto-silence detection** (stops recording after ~1.8 s of silence)
- **Streaming audio playback** — starts playing the first sentence while the rest is still being synthesised
- **Session persistence** via `sessionStorage` — conversation history survives page refreshes
- **Farewell detection** — detects goodbye phrases in both user speech and bot replies to end the call gracefully
- **Language-aware responses** — Hindi/Gujarati questions receive Hindi/Gujarati spoken answers

---

## Development Notes

- The backend uses a single Uvicorn worker (`--workers 1`) because ChromaDB is not safe for multi-process shared access with a local persist directory.
- TTS chunks are synthesised **in parallel** across all sentences and yielded in order, minimising perceived latency.
- MongoDB indexes are created automatically on startup via `ensure_chat_indexes`.
- Set `ENVIRONMENT=development` to see full error messages in API responses and enable `DEBUG` logging.
