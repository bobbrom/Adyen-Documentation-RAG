"""
app.py — Flask web frontend for the Adyen RAG system

Usage:
    pip install flask
    python app.py
    Open http://localhost:5000

Requires ingest.py to have been run first.
"""

from flask import Flask, render_template, request, Response, stream_with_context, session
import json
import os

from query import load_collection, retrieve, build_retrieval_query, append_sources, ask_stream

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Initialise collection once at startup
collection = load_collection()

# In-memory history store keyed by session ID
conversation_histories = {}


@app.route("/")
def index():
    session.setdefault("id", os.urandom(16).hex())
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    question = data.get("question", "").strip()
    if not question:
        return {"error": "No question provided"}, 400

    session_id = session.get("id", "default")
    history = conversation_histories.setdefault(session_id, [])

    retrieval_query = build_retrieval_query(question, history)
    chunks = retrieve(collection, retrieval_query)

    def generate():
        for token in ask_stream(question, chunks, history):
            yield f"data: {json.dumps({'token': token})}\n\n"

        # Append sources after streaming finishes
        sources = append_sources("", chunks).strip()
        yield f"data: {json.dumps({'token': sources})}\n\n"
        yield "data: [DONE]\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"X-Accel-Buffering": "no"}
    )


@app.route("/reset", methods=["POST"])
def reset():
    """Clear conversation history for the current session."""
    session_id = session.get("id", "default")
    conversation_histories.pop(session_id, None)
    return {"status": "ok"}


if __name__ == "__main__":
    from query import OLLAMA_MODEL, COLLECTION_NAME
    print(f"Starting Adyen RAG on http://localhost:5000")
    print(f"Model: {OLLAMA_MODEL} | Collection: {COLLECTION_NAME}")
    app.run(debug=True, threaded=True)
