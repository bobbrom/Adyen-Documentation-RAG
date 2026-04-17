"""
codebase_mcp_server.py — Java codebase semantic search as an MCP tool for Junie

Junie handles the LLM — this server just does retrieval from ChromaDB.

Setup:
    pip install mcp chromadb sentence-transformers

Run (Junie starts this automatically via mcp.json):
    python codebase_mcp_server.py
"""

import asyncio
import os
import chromadb
from chromadb.utils import embedding_functions
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "codebase_chroma_db")
COLLECTION_NAME = "codebase"
EMBEDDING_MODEL = "nomic-ai/nomic-embed-code"
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

server = Server("java-codebase")


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="search_codebase",
            description=(
                "Semantically search the codebase for relevant methods, classes, "
                "functions, components, or configuration. Supports Java, TypeScript, "
                "TSX, JavaScript, and YAML. Use this to find where something is "
                "implemented, how a feature works, which classes handle a given "
                "behaviour, or how a Lambda function or React component is structured."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The feature, behaviour, or concept to search for"
                    }
                },
                "required": ["query"]
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    if name != "search_codebase":
        raise ValueError(f"Unknown tool: {name}")

    query = arguments["query"]

    results = collection.query(
        query_texts=[query],
        n_results=TOP_K,
        include=["documents", "metadatas"]
    )

    chunks = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        lang = meta.get("language", "")
        header = f"File: {meta['file']} (lines {meta['start_line']}–{meta['end_line']}) [{meta['type']}]"
        chunks.append(f"{header}\n\n```{lang}\n{doc}\n```")

    output = "\n\n---\n\n".join(chunks)
    return [types.TextContent(type="text", text=output)]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
