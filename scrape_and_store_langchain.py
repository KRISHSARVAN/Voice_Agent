"""
scrape_and_store_langchain.py    Scrape help.suvit.io and store in ChromaDB using LangChain.

Install:
    pip install langchain langchain-community langchain-chroma \
                sentence-transformers beautifulsoup4 requests lxml tqdm

Run:
    python scrape_and_store_langchain.py
"""

import re
import time
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from tqdm import tqdm

from rag.embeddings import build_embeddings
from rag.store_config import CHROMA_DIR, COLLECTION_NAME, EMBED_MODEL

# ──────────────────────────── CONFIG ────────────────────────────────────── #

BASE_URL        = "https://help.suvit.io"
ALLOWED_DOMAIN  = "help.suvit.io"

MAX_PAGES       = 300
REQUEST_DELAY   = 0.6                          # seconds between requests
CHUNK_SIZE      = 1000                         # characters per chunk
CHUNK_OVERLAP   = 150                          # overlap between chunks
BATCH_SIZE      = 64                           # docs per Chroma upsert

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; SuvitHelpScraper/1.0)"}
REMOVE_TAGS = ["nav", "header", "footer", "aside", "script", "style", "noscript"]


def build_embeddings_for_ingest():
    print(f"🔧  Loading embedding model: {EMBED_MODEL} …")
    return build_embeddings(EMBED_MODEL)

# ──────────────────────────── SCRAPER ───────────────────────────────────── #

def clean_text(soup: BeautifulSoup) -> str:
    for tag in REMOVE_TAGS:
        for el in soup.find_all(tag):
            el.decompose()
    root = soup.find("article") or soup.find("main") or soup.body
    if not root:
        return ""
    text = root.get_text(separator="\n")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def get_links(soup: BeautifulSoup, current_url: str) -> list[str]:
    links = []
    for a in soup.find_all("a", href=True):
        full = urljoin(current_url, a["href"])
        p = urlparse(full)
        if p.netloc == ALLOWED_DOMAIN and p.scheme in ("http", "https"):
            links.append(p._replace(fragment="", query="").geturl())
    return links


def scrape() -> list[Document]:
    """Crawl help.suvit.io and return a list of LangChain Documents."""
    visited, queue, documents = set(), [BASE_URL], []
    session = requests.Session()

    print(f"\n📡 Scraping {BASE_URL}  (max {MAX_PAGES} pages)…\n")

    while queue and len(visited) < MAX_PAGES:
        url = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)

        try:
            resp = session.get(url, headers=HEADERS, timeout=12)
            resp.raise_for_status()
            if "text/html" not in resp.headers.get("Content-Type", ""):
                continue
        except requests.RequestException as e:
            print(f"  ⚠  Skip {url}  →  {e}")
            continue

        soup  = BeautifulSoup(resp.text, "lxml")
        title = soup.title.string.strip() if soup.title else url
        text  = clean_text(soup)

        if len(text) < 80:
            continue

        # ── LangChain Document ──────────────────────────────────────────── #
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": url,
                    "title":  title,
                },
            )
        )

        print(f"  ✔  [{len(visited):>3}]  {title[:70]}")

        for link in get_links(soup, url):
            if link not in visited:
                queue.append(link)

        time.sleep(REQUEST_DELAY)

    print(f"\n✅  Scraped {len(documents)} pages.\n")
    return documents


# ──────────────────────────── CHUNK ─────────────────────────────────────── #

def chunk(documents: list[Document]) -> list[Document]:
    """Split documents into smaller chunks using LangChain splitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],   # tries largest first
    )
    chunks = splitter.split_documents(documents)
    print(f"📄  {len(documents)} pages  →  {len(chunks)} chunks\n")
    return chunks


# ──────────────────────────── STORE IN CHROMADB ─────────────────────────── #

def store(chunks: list[Document]) -> None:
    embeddings = build_embeddings_for_ingest()

    print(f"💾  Storing to ChromaDB at: {CHROMA_DIR}\n")

    # Upsert in batches to avoid memory spikes
    db = None
    for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Storing batches"):
        batch = chunks[i : i + BATCH_SIZE]
        if db is None:
            db = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                collection_name=COLLECTION_NAME,
                persist_directory=CHROMA_DIR,
            )
        else:
            db.add_documents(batch)

    total = db._collection.count() if db else 0
    print(f"\n✅  Done! ChromaDB now has {total} chunks.")
    print(f"   Saved to: {CHROMA_DIR}")


# ──────────────────────────── VERIFY (optional) ─────────────────────────── #

def verify():
    """Quick sanity-check: run a similarity search after storing."""
    print("\n🔍  Running test query…")
    embeddings = build_embeddings_for_ingest()
    db = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )
    results = db.similarity_search("How to upload bank statement", k=3)
    print(f"\nTop {len(results)} results:\n")
    for i, doc in enumerate(results, 1):
        print(f"  [{i}] {doc.metadata.get('title', 'N/A')}")
        print(f"       {doc.metadata.get('source', '')}")
        print(f"       {doc.page_content[:120].strip()}…\n")


# ──────────────────────────── MAIN ──────────────────────────────────────── #

if __name__ == "__main__":
    docs   = scrape()
    if not docs:
        print("❌  No pages scraped. Check your internet or BASE_URL.")
    else:
        chunks = chunk(docs)
        store(chunks)
        verify()