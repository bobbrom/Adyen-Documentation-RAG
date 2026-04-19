import os
import sys
import subprocess
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import chromadb
from chromadb.utils import embedding_functions
from tree_sitter import Language, Parser
import tree_sitter_java as tsjava
import tree_sitter_typescript as tstype
import tree_sitter_javascript as tsjs
import tree_sitter_yaml as tsyaml
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COLLECTION_NAME = "codebase"
EMBEDDING_MODEL = "nomic-ai/nomic-embed-code"
BATCH_SIZE = 100
PARALLEL_WORKERS = 2

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
    ".yaml": {"block_mapping_pair"},
    ".yml":  {"block_mapping_pair"},
}

SUPPORTED_EXTENSIONS = set(CHUNK_NODE_TYPES.keys())
YAML_EXTENSIONS = {".yaml", ".yml"}

PARSERS = {
    ".java": Parser(Language(tsjava.language())),
    ".ts":   Parser(Language(tstype.language_typescript())),
    ".tsx":  Parser(Language(tstype.language_tsx())),
    ".js":   Parser(Language(tsjs.language())),
    ".mjs":  Parser(Language(tsjs.language())),
    ".yaml": Parser(Language(tsyaml.language())),
    ".yml":  Parser(Language(tsyaml.language())),
}


def get_project_name(codebase_path: str) -> str:
    return os.path.basename(codebase_path.rstrip("/"))


def get_chroma_path(codebase_path: str) -> str:
    return os.path.join(BASE_DIR, f"codebase_chroma_db_{get_project_name(codebase_path)}")


def get_commit_hash_file(codebase_path: str) -> str:
    return os.path.join(BASE_DIR, f".last_indexed_commit_{get_project_name(codebase_path)}")


def get_current_commit(codebase_path: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=codebase_path,
            capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return None


def get_last_indexed_commit(codebase_path: str) -> str | None:
    try:
        with open(get_commit_hash_file(codebase_path), "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        return None


def save_commit_hash(codebase_path: str, commit_hash: str):
    with open(get_commit_hash_file(codebase_path), "w") as f:
        f.write(commit_hash)


def get_changed_files(codebase_path: str, since_commit: str) -> dict[str, str]:
    try:
        result = subprocess.run(
            ["git", "diff", "--name-status", since_commit, "HEAD"],
            cwd=codebase_path,
            capture_output=True, text=True, check=True
        )
        changed = {}
        for line in result.stdout.strip().splitlines():
            if not line:
                continue
            parts = line.split("\t")
            status = parts[0][0]
            path = parts[2] if status == "R" and len(parts) > 2 else parts[1]
            changed[path] = status
        return changed
    except subprocess.CalledProcessError:
        return {}


def extract_chunks(source: str, file_path: str, ext: str) -> list[dict]:
    parser = PARSERS[ext]
    node_types = CHUNK_NODE_TYPES[ext]
    tree = parser.parse(bytes(source, "utf-8"))
    chunks = []

    if ext in YAML_EXTENSIONS:
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
    skip_dirs = {"target", "build", ".git", ".idea", "node_modules", "__pycache__", "dist", ".next"}
    source_files = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        for filename in filenames:
            ext = os.path.splitext(filename)[1]
            if ext in SUPPORTED_EXTENSIONS:
                source_files.append(os.path.join(dirpath, filename))
    return source_files


def get_or_create_collection(client, embed_fn):
    try:
        return client.get_collection(name=COLLECTION_NAME, embedding_function=embed_fn)
    except Exception:
        return client.create_collection(
            name=COLLECTION_NAME,
            embedding_function=embed_fn,
            metadata={"hnsw:space": "cosine"}
        )


def delete_file_chunks(collection, rel_path: str):
    try:
        results = collection.get(where={"file": rel_path})
        if results["ids"]:
            collection.delete(ids=results["ids"])
            print(f"  Removed {len(results['ids'])} chunks for {rel_path}")
    except Exception as e:
        print(f"  Could not delete chunks for {rel_path}: {e}")


def index_file(collection, file_path: str, rel_path: str):
    ext = os.path.splitext(file_path)[1]
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            source = f.read()
    except Exception as e:
        print(f"  Could not read {file_path}: {e}")
        return 0

    chunks = extract_chunks(source, rel_path, ext)
    if not chunks:
        return 0

    ids = [f"{rel_path}:{c['start_line']}" for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [{
        "file": c["file"],
        "type": c["type"],
        "start_line": c["start_line"],
        "end_line": c["end_line"],
        "language": ext.lstrip(".")
    } for c in chunks]

    collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
    return len(chunks)


def parse_file(file_path: str, rel_path: str) -> tuple[str, list[dict]]:
    ext = os.path.splitext(file_path)[1]
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            source = f.read()
    except Exception as e:
        print(f"  Could not read {file_path}: {e}")
        return rel_path, []
    return rel_path, extract_chunks(source, rel_path, ext)


def get_existing_ids(collection) -> set[str]:
    existing = set()
    limit = 1000
    offset = 0
    while True:
        result = collection.get(limit=limit, offset=offset, include=[])
        if not result["ids"]:
            break
        existing.update(result["ids"])
        if len(result["ids"]) < limit:
            break
        offset += limit
    return existing


def ingest_full(codebase_path: str, include_tests: bool, client, embed_fn, st_model):
    print("Running full index...")
    source_files = find_source_files(codebase_path)

    if not include_tests:
        source_files = [
            f for f in source_files
            if "/test/" not in f
            and "Test.java" not in f
            and ".test." not in f
            and ".spec." not in f
        ]

    by_ext = {}
    for f in source_files:
        ext = os.path.splitext(f)[1]
        by_ext[ext] = by_ext.get(ext, 0) + 1
    for ext, count in sorted(by_ext.items()):
        print(f"  {ext}: {count} files")
    print(f"  Total: {len(source_files)} files\n")

    collection = get_or_create_collection(client, embed_fn)

    print("Checking existing chunks in DB...")
    existing_ids = get_existing_ids(collection)
    print(f"  Found {len(existing_ids)} already indexed\n")

    print(f"Parsing files with {PARALLEL_WORKERS} workers...")
    all_chunks_by_file = {}
    pairs = [(fp, os.path.relpath(fp, codebase_path)) for fp in source_files]

    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
        futures = {executor.submit(parse_file, fp, rp): rp for fp, rp in pairs}
        for i, future in enumerate(as_completed(futures)):
            try:
                rel_path, chunks = future.result()
                all_chunks_by_file[rel_path] = chunks
                if (i + 1) % 10 == 0:
                    print(f"  Parsed {i+1}/{len(source_files)} files...")
            except Exception as e:
                print(f"  Worker failed: {e}")

    print("Collecting chunks...")
    all_ids, all_docs, all_metas = [], [], []
    for rel_path, chunks in all_chunks_by_file.items():
        ext = os.path.splitext(rel_path)[1]
        for c in chunks:
            chunk_id = f"{rel_path}:{c['start_line']}"
            if chunk_id in existing_ids:
                continue
            all_ids.append(chunk_id)
            all_docs.append(c["text"])
            all_metas.append({
                "file": c["file"],
                "type": c["type"],
                "start_line": c["start_line"],
                "end_line": c["end_line"],
                "language": ext.lstrip(".")
            })

    total_chunks = len(all_ids)
    if total_chunks == 0:
        print("All chunks already in DB — nothing to do")
        return collection

    print(f"Embedding and writing {total_chunks} new chunks in batches of {BATCH_SIZE}...")

    written = 0
    for i in range(0, total_chunks, BATCH_SIZE):
        batch_docs  = all_docs[i:i+BATCH_SIZE]
        batch_ids   = all_ids[i:i+BATCH_SIZE]
        batch_metas = all_metas[i:i+BATCH_SIZE]

        embeddings = st_model.encode(
            batch_docs,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        collection.upsert(
            documents=batch_docs,
            embeddings=embeddings.tolist(),
            metadatas=batch_metas,
            ids=batch_ids
        )

        written += len(batch_ids)
        pct = round(written / total_chunks * 100)
        print(f"  Written {written}/{total_chunks} chunks ({pct}%)...")

    print(f"\nFull index complete — {len(source_files)} files, {written} chunks written")
    return collection


def ingest_incremental(codebase_path: str, since_commit: str, include_tests: bool, client, embed_fn):
    print(f"Incremental update since {since_commit[:8]}...")
    changed = get_changed_files(codebase_path, since_commit)

    changed = {
        path: status for path, status in changed.items()
        if os.path.splitext(path)[1] in SUPPORTED_EXTENSIONS
    }

    if not changed:
        print("No supported files changed — nothing to do")
        return

    print(f"  {len(changed)} files changed\n")

    collection = get_or_create_collection(client, embed_fn)
    total_chunks = 0

    for rel_path, status in changed.items():
        print(f"  [{status}] {rel_path}")

        delete_file_chunks(collection, rel_path)

        if status == "D":
            continue

        if not include_tests:
            if "/test/" in rel_path or "Test.java" in rel_path or ".test." in rel_path or ".spec." in rel_path:
                print(f"  Skipping test file: {rel_path}")
                continue

        abs_path = os.path.join(codebase_path, rel_path)
        if not os.path.exists(abs_path):
            continue

        total_chunks += index_file(collection, abs_path, rel_path)

    print(f"\nIncremental update complete — {len(changed)} files, {total_chunks} chunks added/updated")


def main():
    arg_parser = argparse.ArgumentParser(description="Index a codebase into ChromaDB")
    arg_parser.add_argument("path", help="Path to the root of the codebase")
    arg_parser.add_argument("--full", action="store_true", help="Force a full re-index")
    arg_parser.add_argument("--include-tests", action="store_true", help="Include test files")
    args = arg_parser.parse_args()

    if not os.path.isdir(args.path):
        print(f"Error: {args.path} is not a directory")
        sys.exit(1)

    codebase_path = os.path.abspath(args.path)
    project_name = get_project_name(codebase_path)
    chroma_path = get_chroma_path(codebase_path)
    current_commit = get_current_commit(codebase_path)
    last_commit = get_last_indexed_commit(codebase_path)

    print(f"Project:       {project_name}")
    print(f"Database:      {chroma_path}")
    print(f"Current HEAD:  {current_commit[:8] if current_commit else 'N/A'}")
    print(f"Last indexed:  {last_commit[:8] if last_commit else 'none'}\n")

    client = chromadb.PersistentClient(path=chroma_path)
    st_model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL,
        device="cpu"
    )

    if args.full:
        try:
            client.delete_collection(COLLECTION_NAME)
            print("Deleted existing collection")
        except Exception:
            pass
        ingest_full(codebase_path, args.include_tests, client, embed_fn, st_model)
    elif not last_commit or not current_commit:
        ingest_full(codebase_path, args.include_tests, client, embed_fn, st_model)
    else:
        ingest_incremental(codebase_path, last_commit, args.include_tests, client, embed_fn)

    if current_commit:
        save_commit_hash(codebase_path, current_commit)
        print(f"Saved commit hash: {current_commit[:8]}")


if __name__ == "__main__":
    main()