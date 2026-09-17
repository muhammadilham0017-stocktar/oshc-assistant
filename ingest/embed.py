"""
Step 2. Precompute vectors for every chunk.

    python ingest/embed.py

Output: data/vectors.npy

Uses fastembed, which runs the model on ONNX rather than PyTorch. About
25MB of libraries instead of 2GB, which is what makes this fit inside
Streamlit Community Cloud's memory allocation.
"""
import json
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding

MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def main():
    chunks = json.loads(Path("data/chunks.json").read_text())
    texts = [c["embed_text"] for c in chunks]
    print(f"embedding {len(texts)} chunks with {MODEL}")

    model = TextEmbedding(MODEL)
    vectors = np.array(list(model.embed(texts)), dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    np.save("data/vectors.npy", vectors)
    print(f"wrote data/vectors.npy  shape={vectors.shape}")


if __name__ == "__main__":
    main()
