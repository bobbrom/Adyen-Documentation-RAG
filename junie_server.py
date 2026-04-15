"""
mcp_server.py — Expose Adyen RAG as an MCP tool for Junie

Setup:
    pip install mcp chromadb sentence-transformers

Run (Junie starts this automatically via mcp.json):
    python mcp_server.py
"""

import asyncio
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

from query import load_collection, retrieve, append_sources, build_retrieval_query

# Initialise collection once at startup
collection = load_collection()

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
    retrieval_query = build_retrieval_query(query, [])
    chunks = retrieve(collection, retrieval_query)
    output = append_sources("", chunks).strip()

    return [types.TextContent(type="text", text=output)]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
