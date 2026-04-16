"""
ingest.py — Crawl Adyen docs and store in ChromaDB

Run once (or whenever docs update):
    python ingest.py

Requirements:
    pip install chromadb sentence-transformers requests
"""
import random
import re
import time
import requests
import chromadb
from chromadb.utils import embedding_functions

LLMS_TXT_URL = "https://docs.adyen.com/llms.txt"
CHROMA_PATH = "./adyen_chroma_db"
COLLECTION_NAME = "adyen_docs"
CHUNK_SIZE = 500        # words per chunk
CHUNK_OVERLAP = 50      # words of overlap between chunks
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # fast, free, local — no API key needed


def fetch_doc_urls(llms_txt_url: str) -> list[str]:
    """Parse llms.txt and return all .md URLs."""
    print(f"Fetching index from {llms_txt_url}...")
    response = requests.get(llms_txt_url, timeout=10)
    response.raise_for_status()

    urls = re.findall(r'\(https://[^\)]+\.md\)', response.text)
    urls = [url.strip("()") for url in urls]
    print(f"Found {len(urls)} doc URLs")
    return urls


def fetch_markdown(url: str) -> str | None:
    """Fetch a single markdown file. Returns None on failure."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        return response.text
    except Exception as e:
        print(f"  ⚠ Failed to fetch {url}: {e}")
        return None


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping word-based chunks."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def add_all_from_api_explorer():
    llms_txt_url = "https://docs.adyen.com/api-explorer/llms.txt"
    # Get all doc URLs from the API Explorer index
    urls = fetch_doc_urls(llms_txt_url)
    # Replace all 'api-explorer/api-explorer' with 'api-explorer' to get the correct raw markdown URLs
    urls = [url.replace("api-explorer/api-explorer", "api-explorer") for url in urls]
    for i, url in enumerate(urls):
        print(f"[{i+1}/{len(urls)}] Adding from API Explorer: {url}")
        add_doc(url)

        # Be polite to Adyen's servers
        # Add some random jitter to avoid hammering if there are many docs
        random_time = random.uniform(0.1, 0.5)
        time.sleep(random_time)


def add_doc(url: str):
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    collection = client.get_collection(name=COLLECTION_NAME, embedding_function=embed_fn)

    markdown = requests.get(url).text
    chunks = chunk_text(markdown, CHUNK_SIZE, CHUNK_OVERLAP)

    # Use the URL as a stable base for IDs to avoid clashes
    safe_id = url.replace("https://", "").replace("/", "_").replace(".", "_")
    collection.add(
        documents=chunks,
        metadatas=[{"url": url, "chunk": j} for j in range(len(chunks))],
        ids=[f"{safe_id}_{j}" for j in range(len(chunks))]
    )
    print(f"Added {len(chunks)} chunks from {url}")

def ingest():
    # Set up ChromaDB with a local embedding model (no API key needed)
    print(f"\nInitialising ChromaDB at '{CHROMA_PATH}'...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    # Delete existing collection so we start fresh on re-runs
    try:
        client.delete_collection(COLLECTION_NAME)
        print("Deleted existing collection")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"}
    )

    # Fetch all doc URLs from llms.txt
    urls = fetch_doc_urls(LLMS_TXT_URL)

    total_chunks = 0
    for i, url in enumerate(urls):
        print(f"[{i+1}/{len(urls)}] {url}")
        markdown = fetch_markdown(url)
        if not markdown:
            continue

        chunks = chunk_text(markdown, CHUNK_SIZE, CHUNK_OVERLAP)

        # Store each chunk with metadata
        collection.add(
            documents=chunks,
            metadatas=[{"url": url, "chunk": j} for j in range(len(chunks))],
            ids=[f"{i}_{j}" for j in range(len(chunks))]
        )
        total_chunks += len(chunks)

        # Be polite to Adyen's servers
        # Add some random jitter to avoid hammering if there are many docs
        random_time = random.uniform(0.1, 0.5)
        time.sleep(random_time)

    print(f"\n Ingested {len(urls)} docs as {total_chunks} chunks into ChromaDB")


if __name__ == "__main__":
    # ingest()
    add_all_from_api_explorer()