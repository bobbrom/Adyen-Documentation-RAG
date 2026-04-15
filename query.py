"""
query.py — Ask questions about Adyen docs using ChromaDB + Ollama (local LLM)

Setup:
    1. Install Ollama from https://ollama.com
    2. Run: ollama pull llama3.1:8b
    3. Make sure Ollama is running (it starts automatically on Mac after install)

Usage:
    python query.py "How do I set up payouts?"
    python query.py  # interactive mode

Requirements:
    pip install chromadb sentence-transformers openai
"""

import sys
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions
import re

CHROMA_PATH = "./adyen_chroma_db"
COLLECTION_NAME = "adyen_docs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "mistral:7b"
# OLLAMA_MODEL = "llama3.1:8b"  # smaller, faster, less accurate than Mistral
TOP_K = 5   # number of chunks to retrieve per question

# Ollama runs locally and exposes an OpenAI-compatible API
ollama_client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # required by the client but not used
)


def load_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    return client.get_collection(name=COLLECTION_NAME, embedding_function=embed_fn)

def print_blue(text: str):
    """Helper to print text in blue in the terminal."""
    print(f"\033[94m{text}\033[0m")


def retrieve(collection, question: str) -> list[dict]:
    """Find the most relevant chunks for a question."""
    results = collection.query(
        query_texts=[question],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"]
    )
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        chunks.append({"text": doc, "url": meta["url"], "score": 1 - dist})
    return chunks

def build_retrieval_query(question: str, history: list[dict]) -> str:
    """Combine current question with last user message for better retrieval."""
    if history:
        last_user = next(
            (m["content"] for m in reversed(history) if m["role"] == "user"),
            ""
        )
        # Trim to just the question part, not the full doc context
        last_question = last_user.split("Question:")[-1].strip()
        return f"{last_question} {question}"
    return question

def append_sources(answer: str, chunks: list[dict]) -> str:
    """Append real URLs from ChromaDB metadata — bypasses the model entirely."""
    urls = list(dict.fromkeys(c["url"].replace(".md", "") for c in chunks))
    sources = "\n\n## Sources\n" + "\n".join(f"- {url}" for url in urls)
    return answer + sources


def ask_data(question: str, chunks: list[dict], history: list[dict]):
    context = "\n\n---\n\n".join(f"{c['text']}" for c in chunks)

    system_prompt = """You are a helpful assistant that answers questions about Adyen's products and APIs.
    Answer based only on the provided documentation excerpts.
    If the answer isn't in the excerpts, say so clearly.
    Format your answer in Markdown.
    Do not include any source links or URLs in your answer."""

    user_message = f"""Documentation excerpts:

    {context}

    Question: {question}"""

    print_blue(history)
    messages = [
        {"role": "system", "content": system_prompt},
        *history,
        {"role": "user", "content": user_message}
    ]
    return context, system_prompt, user_message, messages


def ask(question: str, chunks: list[dict], history: list[dict]) -> str:
    context, system_prompt, user_message, messages = ask_data(question, chunks, history)
    response = ollama_client.chat.completions.create(
        model=OLLAMA_MODEL,
        messages=messages
    )

    answer = response.choices[0].message.content

    # Append this turn to history for next time
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": answer})

    return answer


def ask_stream(question: str, chunks: list[dict], history: list[dict]):
    context, system_prompt, user_message, messages = ask_data(question, chunks, history)

    stream = ollama_client.chat.completions.create(
        model=OLLAMA_MODEL,
        messages=messages,
        stream=True
    )

    full_response = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            full_response += delta
            yield delta

    # Update history after streaming completes
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": full_response})



def clean_source_urls(text: str) -> str:
    """Remove .md extension from any Source: URLs in the model's output."""
    return re.sub(r'(Source:\s*https?://\S+)\.md', r'\1', text)



def main():
    print("Loading ChromaDB collection...")
    try:
        collection = load_collection()
    except Exception:
        print("Collection not found. Run ingest.py first.")
        sys.exit(1)

    print(f"Using model: {OLLAMA_MODEL} via Ollama\n")

    # Single question mode
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        retrieval_query = build_retrieval_query(question, [])
        chunks = retrieve(collection, retrieval_query)
        answer = ask(question, chunks, [])
        answer = append_sources(answer, chunks)
        answer = clean_source_urls(answer)
        print(f"\n{answer}\n")
        return

    # Interactive mode
    print("Type your question (or 'quit' to exit)\n")
    history = []
    while True:
        question = input("You: ").strip()
        if not question or question.lower() in ("quit", "exit"):
            break

        print("\nSearching docs...")
        retrieval_query = build_retrieval_query(question, history)
        chunks = retrieve(collection, retrieval_query)

        print(f"Found {len(chunks)} relevant chunks. Thinking...\n")
        answer = ask(question, chunks, history)
        answer = append_sources(answer, chunks)
        answer = clean_source_urls(answer)
        print(f"Assistant: {answer}\n")


if __name__ == "__main__":
    main()
