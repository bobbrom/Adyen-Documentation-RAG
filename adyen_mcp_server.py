"""
adyen_mcp_server.py — Adyen docs RAG as an MCP tool for Junie

Junie handles the LLM — this server just does retrieval from ChromaDB.

Setup:
    pip install mcp chromadb sentence-transformers

Run (Junie starts this automatically via mcp.json):
    python adyen_mcp_server.py
"""

import asyncio
import chromadb
from chromadb.utils import embedding_functions
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types
import os

BASE_PATH = os.path.dirname(__file__)

CHROMA_PATH = os.path.join(BASE_PATH, "adyen_chroma_db")
COLLECTION_NAME = "adyen_docs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5

# Initialise ChromaDB once at startup
embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBEDDING_MODEL
)
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_collection(
    name=COLLECTION_NAME,
    embedding_function=embed_fn
)

server = Server("adyen-docs")


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="search_adyen_docs",
            description=(
                "Search the Adyen documentation for information about APIs, "
                "payment methods, webhooks, payouts, onboarding, and more. "
                "Use this whenever you need to answer questions about Adyen."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The question or topic to search for"
                    }
                },
                "required": ["query"]
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    if name != "search_adyen_docs":
        raise ValueError(f"Unknown tool: {name}")

    query = arguments["query"]

    results = collection.query(
        query_texts=[query],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"]
    )

    chunks = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        url = meta["url"].replace(".md", "")
        chunks.append(f"Source: {url}\n\n{doc}")

    output = "\n\n---\n\n".join(chunks)
    return [types.TextContent(type="text", text=output)]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())