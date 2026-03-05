"""
search_engine.py — Core search logic.

Supports:
  • Image-to-image search using CLIP embeddings.
  • Text-to-image search via CLIP text encoder.
  • Category/gender post-filtering on FAISS results.
  • Maximal Marginal Relevance (MMR) for result diversity.
  • Latency profiling (encoding / search / post-processing).
"""

import os
import sys
import time

import numpy as np
import torch
import faiss
import open_clip
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import load_metadata, get_image_path, EMBED_DIR

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"


class SearchEngine:
    """
    Encapsulates CLIP model, FAISS index, metadata, and exposes
    `search_by_image` / `search_by_text` methods with MMR and filtering.
    """

    def __init__(self):
        # ── Load CLIP ──
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            MODEL_NAME, pretrained=PRETRAINED, device=DEVICE
        )
        self.tokenizer = open_clip.get_tokenizer(MODEL_NAME)
        self.model.eval()

        # ── Load FAISS index ──
        index_path = os.path.join(EMBED_DIR, "faiss_index.bin")
        self.index = faiss.read_index(index_path)

        # ── Load ID mapping & normalised embeddings ──
        self.image_ids  = np.load(os.path.join(EMBED_DIR, "image_ids.npy"))
        embeddings      = np.load(os.path.join(EMBED_DIR, "image_embeddings.npy")).astype(np.float32)
        faiss.normalize_L2(embeddings)
        self.embeddings = embeddings

        # ── Load metadata ──
        self.metadata = load_metadata(clean=True)
        self.metadata.set_index("id", inplace=True)

    # ────────────────────────────── public API ──

    def search_by_image(self, pil_image, top_k=5, category=None, gender=None, diversity=True):
        """
        Search for similar products given a PIL image.
        Returns (results_list, latency_dict).
        """
        t_start = time.perf_counter()

        # Encode
        img_tensor = self.preprocess(pil_image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            query_emb = self.model.encode_image(img_tensor).cpu().numpy().astype(np.float32)
        faiss.normalize_L2(query_emb)
        t_encode = time.perf_counter()

        # Search
        results, latency = self._search_core(query_emb, top_k, category, gender, diversity, t_start, t_encode)
        return results, latency

    def search_by_text(self, text_query, top_k=5, category=None, gender=None, diversity=True):
        """
        Search for products matching a natural-language description.
        Returns (results_list, latency_dict).
        """
        t_start = time.perf_counter()

        # Encode text
        tokens = self.tokenizer([text_query]).to(DEVICE)
        with torch.no_grad():
            query_emb = self.model.encode_text(tokens).cpu().numpy().astype(np.float32)
        faiss.normalize_L2(query_emb)
        t_encode = time.perf_counter()

        results, latency = self._search_core(query_emb, top_k, category, gender, diversity, t_start, t_encode)
        return results, latency

    def search_by_product_id(self, product_id, top_k=5, category=None, gender=None, diversity=True):
        """
        "Find Similar" — search using a product's existing embedding.
        """
        t_start = time.perf_counter()

        idx = np.where(self.image_ids == product_id)[0]
        if len(idx) == 0:
            return [], {"total_ms": 0}

        query_emb = self.embeddings[idx[0]:idx[0]+1].copy()
        t_encode = time.perf_counter()

        results, latency = self._search_core(query_emb, top_k + 1, category, gender, diversity, t_start, t_encode)
        # Remove the query product itself from results
        results = [r for r in results if r["product_id"] != product_id][:top_k]
        return results, latency

    # ──────────────────────────── internals ──

    def _search_core(self, query_emb, top_k, category, gender, diversity, t_start, t_encode):
        """Run FAISS search → post-filter → MMR → build result dicts."""

        # Fetch a larger pool for post-filtering / MMR
        fetch_k = min(max(top_k * 10, 50), self.index.ntotal)
        scores, indices = self.index.search(query_emb, fetch_k)
        t_search = time.perf_counter()

        scores  = scores[0]
        indices = indices[0]

        # Post-filter by category / gender
        candidates = []
        for rank, (idx, score) in enumerate(zip(indices, scores)):
            pid = int(self.image_ids[idx])
            if pid not in self.metadata.index:
                continue
            meta = self.metadata.loc[pid]
            if category and meta.get("masterCategory", "") != category:
                continue
            if gender and meta.get("gender", "") != gender:
                continue
            candidates.append({
                "faiss_idx": int(idx),
                "product_id": pid,
                "score": float(score),
                "meta": meta,
            })
            if len(candidates) >= fetch_k:
                break

        # MMR for diversity (optional)
        if diversity and len(candidates) > top_k:
            selected = self._mmr(candidates, query_emb[0], top_k, lambda_=0.7)
        else:
            selected = candidates[:top_k]

        t_post = time.perf_counter()

        # Build result list
        results = []
        for c in selected:
            meta = c["meta"]
            results.append({
                "product_id":   c["product_id"],
                "score":        round(c["score"] * 100, 1),  # percentage
                "image_path":   get_image_path(c["product_id"]),
                "product_name": meta.get("productDisplayName", "Unknown"),
                "category":     meta.get("masterCategory", ""),
                "sub_category": meta.get("subCategory", ""),
                "article_type": meta.get("articleType", ""),
                "colour":       meta.get("baseColour", ""),
                "gender":       meta.get("gender", ""),
                "season":       meta.get("season", ""),
            })

        latency = {
            "encode_ms":  round((t_encode - t_start) * 1000, 1),
            "search_ms":  round((t_search - t_encode) * 1000, 1),
            "post_ms":    round((t_post   - t_search) * 1000, 1),
            "total_ms":   round((t_post   - t_start)  * 1000, 1),
        }

        return results, latency

    def _mmr(self, candidates, query_vec, top_k, lambda_=0.7):
        """
        Maximal Marginal Relevance.
        Balances relevance (similarity to query) with diversity (dissimilarity to
        already-selected results).
        lambda_ = 1.0 → pure relevance, lambda_ = 0.0 → pure diversity.
        """
        if not candidates:
            return []

        # Gather candidate embeddings
        cand_embs = np.array([self.embeddings[c["faiss_idx"]] for c in candidates])

        selected_indices = []
        remaining        = list(range(len(candidates)))

        for _ in range(min(top_k, len(candidates))):
            if not remaining:
                break

            if not selected_indices:
                # First pick: highest relevance
                best = max(remaining, key=lambda i: candidates[i]["score"])
            else:
                # Subsequent picks: balance relevance vs diversity
                sel_embs = cand_embs[selected_indices]
                best_score = -float("inf")
                best = remaining[0]
                for i in remaining:
                    relevance = candidates[i]["score"]
                    # Max similarity to any already-selected
                    sim_to_selected = float(np.max(cand_embs[i] @ sel_embs.T))
                    mmr_score = lambda_ * relevance - (1 - lambda_) * sim_to_selected
                    if mmr_score > best_score:
                        best_score = mmr_score
                        best = i

            selected_indices.append(best)
            remaining.remove(best)

        return [candidates[i] for i in selected_indices]

    # ──────────────────────────── helpers ──

    def get_categories(self):
        """Return unique master categories for the filter dropdown."""
        return sorted(self.metadata["masterCategory"].dropna().unique().tolist())

    def get_genders(self):
        """Return unique genders for the filter dropdown."""
        return sorted(self.metadata["gender"].dropna().unique().tolist())

    def total_products(self):
        return self.index.ntotal
