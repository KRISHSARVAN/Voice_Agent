"""RAG retrieval and LLM answer generation."""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator

from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.config import Settings


SYSTEM_PROMPT = """You are Suvit's customer support assistant. Answer using the context below and the conversation. \
If the context does not contain enough information, say what is missing and suggest checking the linked help articles or Suvit support. \
Be concise and practical."""

# SYSTEM_PROMPT = """You are Suvit's friendly voice support assistant. You help users with questions about Suvit — an accounting automation software for Indian CAs and businesses.

# VOICE RULES (Critical):
# - Speak in short, natural sentences. No bullet points, no markdown, no lists.
# - Never say URLs or long links aloud — instead say "check the Suvit help center" or "I'll note the article for you."
# - Spell out abbreviations if needed (e.g., say "G S T" not "GST" if unclear).
# - Use a warm, helpful tone — like a knowledgeable colleague, not a manual.

# ANSWERING RULES:
# - Answer only from the provided context. Do not guess or fabricate steps.
# - If the context partially answers the question, share what you know and say: "For the complete steps, I'd recommend checking the Suvit help center."
# - If the context has no relevant information, say: "I don't have that detail right now. Please reach out to Suvit support directly or visit help.suvit.io for accurate guidance."
# - If the user seems stuck or frustrated, acknowledge it first before answering: "I understand that can be confusing — let me help."

# SCOPE:
# - Only answer questions related to Suvit features, workflows, and account issues.
# - For tax/legal advice, say: "I can help with how Suvit handles this, but for tax advice please consult your CA."

# Keep every response under 3-4 sentences unless the user asks for more detail."""

# Maps BCP-47 language codes to human-readable names used in the system prompt.
_LANGUAGE_NAMES: dict[str, str] = {
    "hi-IN": "Hindi",
    "gu-IN": "Gujarati",
    "en-IN": "English",
    "en-US": "English",
    "en-GB": "English",
}

# Language codes that are NOT English and need query translation for vector search.
_NON_ENGLISH_LANGS = {"hi-IN", "gu-IN"}


def split_messages(
    messages: list[tuple[str, str]],
) -> tuple[str, list[tuple[str, str]]]:
    """Merge free-form system strings into context; keep user/assistant for chat."""
    system_parts: list[str] = []
    chat: list[tuple[str, str]] = []
    for role, content in messages:
        if role == "system":
            system_parts.append(content.strip())
        elif role in ("user", "assistant"):
            chat.append((role, content))
        else:
            chat.append(("user", f"[{role}] {content}"))
    merged = "\n\n".join(s for s in system_parts if s)
    return merged, chat


def lc_messages_from_pairs(messages: list[tuple[str, str]]) -> list[BaseMessage]:
    out: list[BaseMessage] = []
    for role, content in messages:
        if role == "user":
            out.append(HumanMessage(content=content))
        elif role == "assistant":
            out.append(AIMessage(content=content))
        else:
            out.append(HumanMessage(content=f"[{role}] {content}"))
    return out


def build_llm(settings: Settings):
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings.openai_model,
        temperature=settings.openai_temperature,
        api_key=settings.openai_api_key,
        timeout=settings.openai_timeout_s,
    )


def translate_query_to_english(query: str, language_code: str, settings: Settings) -> str:
    """Translate a non-English user query to English for ChromaDB vector search."""
    from langchain_openai import ChatOpenAI

    lang_name = _LANGUAGE_NAMES.get(language_code, language_code)
    llm = ChatOpenAI(
        model=settings.openai_model,
        temperature=0,
        api_key=settings.openai_api_key,
        timeout=settings.openai_timeout_s,
    )
    messages = [
        (
            "system",
            "You are a precise translator. Translate the user's text from "
            f"{lang_name} to English. Output ONLY the translated text, nothing else.",
        ),
        ("user", query),
    ]
    result = (llm | StrOutputParser()).invoke(messages)
    return result.strip() or query


def format_context_from_docs(
    docs: list[tuple[Any, float]],
    max_chars: int,
) -> str:
    parts: list[str] = []
    used = 0
    for i, (doc, score) in enumerate(docs, start=1):
        md = doc.metadata or {}
        title = md.get("title") or "Untitled"
        source = md.get("source") or ""
        body = (doc.page_content or "").strip()
        block = f"### Source {i}: {title}\nURL: {source}\n(relevance-distance: {score:.4f} — lower is closer)\n\n{body}\n"
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)
    return "\n---\n".join(parts)


def retrieve(
    *,
    vectorstore: Chroma,
    query: str,
    top_k: int,
) -> list[tuple[Any, float]]:
    return vectorstore.similarity_search_with_score(query, k=top_k)


def _prepare_rag(
    *,
    vectorstore: Chroma,
    settings: Settings,
    messages: list[tuple[str, str]],
    top_k: int | None,
    language_code: str,
):
    """Run retrieval and build the LangChain chain + inputs. Synchronous (safe to thread)."""
    k = top_k if top_k is not None else settings.rag_top_k
    extra_system, chat_messages = split_messages(messages)
    last_user = chat_messages[-1][1]

    retrieval_query = last_user
    if language_code in _NON_ENGLISH_LANGS:
        retrieval_query = translate_query_to_english(last_user, language_code, settings)

    pairs = retrieve(vectorstore=vectorstore, query=retrieval_query, top_k=k)
    context = format_context_from_docs(pairs, settings.rag_max_context_chars)
    if extra_system:
        context += "\n\n---\nUser / developer instructions:\n" + extra_system

    llm = build_llm(settings)
    lang_name = _LANGUAGE_NAMES.get(language_code, "English")
    lang_instruction = (
        f"IMPORTANT: The user is communicating in {lang_name}. "
        f"You MUST reply exclusively in {lang_name}, regardless of the language "
        "of the context documents."
    )
    system_with_lang = f"{SYSTEM_PROMPT}\n\n{lang_instruction}\n\nContext:\n{{context}}"

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_with_lang),
            MessagesPlaceholder(variable_name="history"),
        ]
    )
    history = lc_messages_from_pairs(chat_messages)
    chain = prompt | llm | StrOutputParser()
    return chain, {"context": context, "history": history}, pairs


def run_rag_chat(
    *,
    vectorstore: Chroma,
    settings: Settings,
    messages: list[tuple[str, str]],
    top_k: int | None = None,
    language_code: str = "en-IN",
) -> tuple[str, list[tuple[Any, float]]]:
    chain, inputs, pairs = _prepare_rag(
        vectorstore=vectorstore,
        settings=settings,
        messages=messages,
        top_k=top_k,
        language_code=language_code,
    )
    answer = chain.invoke(inputs)
    return answer, pairs


async def astream_rag_chat(
    *,
    vectorstore: Chroma,
    settings: Settings,
    messages: list[tuple[str, str]],
    top_k: int | None = None,
    language_code: str = "en-IN",
    pairs_out: list,
) -> AsyncIterator[str]:
    """Stream LLM tokens for a RAG query.

    Runs retrieval synchronously in a thread first, then streams tokens via
    LangChain's astream(). Populates `pairs_out` with the retrieved source docs.
    """
    chain, inputs, pairs = await asyncio.to_thread(
        _prepare_rag,
        vectorstore=vectorstore,
        settings=settings,
        messages=messages,
        top_k=top_k,
        language_code=language_code,
    )
    pairs_out.extend(pairs)
    async for chunk in chain.astream(inputs):
        yield chunk
