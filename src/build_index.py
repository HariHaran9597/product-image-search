"""
build_index.py — Build a FAISS index from pre-computed CLIP embeddings.

The index uses Inner Product (IP) on L2-normalised vectors, which is
equivalent to cosine similarity.

Usage:
    python src/build_index.py
"""

import os
import sys
import numpy as np
import faiss

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import EMBED_DIR


def main():
    emb_path = os.path.join(EMBED_DIR, "image_embeddings.npy")
    ids_path = os.path.join(EMBED_DIR, "image_ids.npy")

    print("Loading embeddings …")
    embeddings = np.load(emb_path).astype(np.float32)
    ids        = np.load(ids_path)
    print(f"  Shape: {embeddings.shape}  (products × dim)")

    # ── L2-normalise (so inner product == cosine similarity) ──
    faiss.normalize_L2(embeddings)
    print("  L2-normalised ✓")

    # ── Build FAISS IndexFlatIP ──
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"  Index built — {index.ntotal} vectors")

    # ── Save ──
    index_path = os.path.join(EMBED_DIR, "faiss_index.bin")
    faiss.write_index(index, index_path)
    print(f"  Saved → {index_path}")

    # ── Quick sanity check ──
    print("\nSanity check: querying index with the first vector …")
    query = embeddings[0:1]          # already normalised
    scores, indices = index.search(query, 5)
    print(f"  Top-5 indices : {indices[0]}")
    print(f"  Top-5 scores  : {np.round(scores[0], 4)}")
    print(f"  Top result should be index 0 with score ≈ 1.0  →  {'✅ PASS' if indices[0][0] == 0 else '❌ FAIL'}")


if __name__ == "__main__":
    main()
