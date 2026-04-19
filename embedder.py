import sys
import json
import torch
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "nomic-ai/nomic-embed-code"

def main():
    data = json.loads(sys.stdin.read())
    docs = data["docs"]

    model = SentenceTransformer(EMBEDDING_MODEL, device="mps")
    embeddings = model.encode(
        docs,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True
    )

    del model
    torch.mps.empty_cache()

    sys.stdout.write(json.dumps({"embeddings": embeddings.tolist()}))
    sys.stdout.flush()

if __name__ == "__main__":
    main()
