"""
code_ingest.py — Index a Java/TypeScript/TSX/JS/YAML codebase into ChromaDB using tree-sitter

Run once (or when the codebase changes):
    python code_ingest.py /path/to/your/codebase
"""

import os
import sys
import argparse
import chromadb
from chromadb.utils import embedding_functions
from tree_sitter import Language, Parser
import tree_sitter_java as tsjava
import tree_sitter_typescript as tstype
import tree_sitter_javascript as tsjs
import tree_sitter_yaml as tsyaml

CHROMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "codebase_chroma_db")
COLLECTION_NAME = "codebase"
EMBEDDING_MODEL = "nomic-ai/nomic-embed-code"

# Node types to extract per language
CHUNK_NODE_TYPES = {
    ".java": {
        "method_declaration",
        "constructor_declaration",
        "class_declaration",
        "interface_declaration",
        "enum_declaration",
    },
    ".ts": {
        "function_declaration",
        "arrow_function",
        "class_declaration",
        "interface_declaration",
        "type_alias_declaration",
        "method_definition",
    },
    ".tsx": {
        "function_declaration",
        "arrow_function",
        "class_declaration",
        "interface_declaration",
        "type_alias_declaration",
        "method_definition",
        "jsx_element",
    },
    ".js": {
        "function_declaration",
        "arrow_function",
        "class_declaration",
        "method_definition",
    },
    ".mjs": {
        "function_declaration",
        "arrow_function",
        "class_declaration",
        "method_definition",
    },
    # YAML uses top-level block mapping pairs (e.g. each Lambda function/resource)
    ".yaml": {"block_mapping_pair"},
    ".yml":  {"block_mapping_pair"},
}

# File extensions we care about
SUPPORTED_EXTENSIONS = set(CHUNK_NODE_TYPES.keys())

# Set up parsers per language
PARSERS = {
    ".java": Parser(Language(tsjava.language())),
    ".ts":   Parser(Language(tstype.language_typescript())),
    ".tsx":  Parser(Language(tstype.language_tsx())),
    ".js":   Parser(Language(tsjs.language())),
    ".mjs":  Parser(Language(tsjs.language())),
    ".yaml": Parser(Language(tsyaml.language())),
    ".yml":  Parser(Language(tsyaml.language())),
}

# YAML only chunks top-level keys (depth 1) to avoid massive nesting
YAML_EXTENSIONS = {".yaml", ".yml"}


def extract_chunks(source: str, file_path: str, ext: str) -> list[dict]:
    """Parse a source file and extract functions/classes as individual chunks."""
    parser = PARSERS[ext]
    node_types = CHUNK_NODE_TYPES[ext]
    tree = parser.parse(bytes(source, "utf-8"))
    chunks = []

    if ext in YAML_EXTENSIONS:
        # For YAML, only walk top-level block_mapping_pair nodes
        for node in tree.root_node.children:
            for child in node.children:
                if child.type in node_types:
                    text = source[child.start_byte:child.end_byte]
                    if len(text.strip()) > 30:
                        chunks.append({
                            "text": text,
                            "file": file_path,
                            "type": child.type,
                            "start_line": child.start_point[0] + 1,
                            "end_line": child.end_point[0] + 1,
                        })
        return chunks

    def walk(node):
        if node.type in node_types:
            text = source[node.start_byte:node.end_byte]
            if len(text.strip()) > 30:
                chunks.append({
                    "text": text,
                    "file": file_path,
                    "type": node.type,
                    "start_line": node.start_point[0] + 1,
                    "end_line": node.end_point[0] + 1,
                })
        for child in node.children:
            walk(child)

    walk(tree.root_node)
    return chunks


def find_source_files(root: str) -> list[str]:
    """Recursively find all supported source files, skipping non-essential dirs."""
    skip_dirs = {"target", "build", ".git", ".idea", "node_modules", "__pycache__", "dist", ".next"}
    source_files = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        for filename in filenames:
            ext = os.path.splitext(filename)[1]
            if ext in SUPPORTED_EXTENSIONS:
                source_files.append(os.path.join(dirpath, filename))
    return source_files


def ingest(codebase_path: str, include_tests: bool = False):
    print(f"Scanning {codebase_path}...")
    source_files = find_source_files(codebase_path)

    if not include_tests:
        source_files = [
            f for f in source_files
            if "/test/" not in f
            and "Test.java" not in f
            and ".test." not in f
            and ".spec." not in f
        ]

    # Count by language
    by_ext = {}
    for f in source_files:
        ext = os.path.splitext(f)[1]
        by_ext[ext] = by_ext.get(ext, 0) + 1
    for ext, count in sorted(by_ext.items()):
        print(f"  {ext}: {count} files")
    print(f"  Total: {len(source_files)} files")

    # Set up ChromaDB
    print(f"\nInitialising ChromaDB at '{CHROMA_PATH}'...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

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

    total_chunks = 0
    for i, file_path in enumerate(source_files):
        ext = os.path.splitext(file_path)[1]
        rel_path = os.path.relpath(file_path, codebase_path)
        print(f"[{i+1}/{len(source_files)}] {rel_path}")

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                source = f.read()
        except Exception as e:
            print(f"  ⚠ Could not read {file_path}: {e}")
            continue

        chunks = extract_chunks(source, rel_path, ext)
        if not chunks:
            continue

        ids = [f"{rel_path}:{c['start_line']}" for c in chunks]
        documents = [c["text"] for c in chunks]
        metadatas = [{
            "file": c["file"],
            "type": c["type"],
            "start_line": c["start_line"],
            "end_line": c["end_line"],
            "language": ext.lstrip(".")
        } for c in chunks]

        collection.add(documents=documents, metadatas=metadatas, ids=ids)
        total_chunks += len(chunks)

    print(f"\nDone! Indexed {len(source_files)} files as {total_chunks} chunks")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Index a codebase into ChromaDB")
    arg_parser.add_argument("path", help="Path to the root of the codebase")
    arg_parser.add_argument("--include-tests", action="store_true", help="Include test files")
    args = arg_parser.parse_args()

    if not os.path.isdir(args.path):
        print(f"Error: {args.path} is not a directory")
        sys.exit(1)

    ingest(args.path, include_tests=args.include_tests)