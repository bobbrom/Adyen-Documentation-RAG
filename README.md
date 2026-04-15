## Prerequisites

- Python 3.11+
- [Homebrew](https://brew.sh/)

---

## 1. Install Ollama

```bash
brew install ollama
ollama pull mistral:7b
```

---

## 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

---

## 3. Unzip Vector Database
 
```bash
unzip adyen_chroma_db/chroma.sqlite3.zip
```

---

## 4. Start the app

```bash
python app.py
```

Then open [http://localhost:5000](http://localhost:5000/) in your browser.

---

## Project structure

```
adyen-rag/
├── ingest.py          # Crawl and index Adyen docs into ChromaDB (run once)
├── query.py           # All RAG logic — retrieval, prompting, history
├── app.py             # Flask web server
├── mcp_server.py      # MCP server for Junie integration
├── mcp.json           # Junie MCP config (update path before using)
├── requirements.txt
└── templates/
    └── index.html     # Chat UI
```

---

## Junie integration (optional)

1. Update the path in `mcp.json`:

```json
{
  "mcpServers": {
    "adyen-docs": {
      "command": "python",
      "args": ["/absolute/path/to/adyen-rag/junie_server.py"]
    }
  }
}
```

2. Place `mcp.json` in one of:
    
    - **Project-level**: `.junie/mcp/mcp.json` in your project root
    - **Global**: `~/.junie/mcp/mcp.json`
3. In your JetBrains IDE go to **Settings → Tools → Junie → MCP Settings** and verify `adyen-docs` shows as Active.
    

---

## Tuning

| Setting          | File        | Default      | Notes                                             |
| ---------------- | ----------- | ------------ | ------------------------------------------------- |
| Model            | `query.py`  | `mistral:7b` | Swap for `llama3.1:8b` for a smaller/faster model |
| Chunks retrieved | `query.py`  | `5`          | Lower to `3` for smaller models                   |
| History length   | `query.py`  | Unlimited    | Set `MAX_HISTORY` to cap context window usage     |
| Chunk size       | `ingest.py` | `500` words  | Larger = more context per chunk, fewer results    |
