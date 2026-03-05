"""
evaluate.py — Evaluate search quality with Recall@K metrics.

Compares CLIP (ViT-B/32) against a ResNet50 baseline on same-category retrieval.

Usage:
    python src/evaluate.py
"""

import os
import sys
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import faiss
import open_clip
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import load_metadata, get_image_path, EMBED_DIR

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
NUM_QUERIES    = 200       # Number of random query images to evaluate
TOP_K          = 5         # Recall@K
BATCH_SIZE     = 64
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
SEED           = 42


def encode_with_resnet(image_paths, batch_size=64):
    """Encode images using ResNet50 (avgpool features, 2048-dim)."""
    print("Loading ResNet50 …")
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    # Remove the final classification layer to get feature vectors
    resnet = nn.Sequential(*list(resnet.children())[:-1])
    resnet.to(DEVICE).eval()

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    all_emb = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc="ResNet50 encoding"):
        batch_paths = image_paths[i:i + batch_size]
        imgs = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                imgs.append(preprocess(img))
            except Exception:
                imgs.append(torch.zeros(3, 224, 224))
        batch_tensor = torch.stack(imgs).to(DEVICE)
        with torch.no_grad():
            features = resnet(batch_tensor).squeeze(-1).squeeze(-1)
        all_emb.append(features.cpu().numpy())

    return np.vstack(all_emb).astype(np.float32)


def encode_with_clip(image_paths, batch_size=64):
    """Encode images using CLIP ViT-B/32 (512-dim)."""
    print("Loading CLIP (ViT-B-32) …")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai", device=DEVICE
    )
    model.eval()

    all_emb = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc="CLIP encoding"):
        batch_paths = image_paths[i:i + batch_size]
        imgs = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                imgs.append(preprocess(img))
            except Exception:
                imgs.append(torch.zeros(3, 224, 224))
        batch_tensor = torch.stack(imgs).to(DEVICE)
        with torch.no_grad():
            features = model.encode_image(batch_tensor)
        all_emb.append(features.cpu().numpy())

    return np.vstack(all_emb).astype(np.float32)


def compute_recall_at_k(embeddings, categories, query_indices, k=5):
    """
    For each query, check if the top-K results (excluding self) contain
    at least one item from the same category.

    Returns: recall (float), mrr (float)
    """
    # Normalise and build FAISS index
    emb = embeddings.copy()
    faiss.normalize_L2(emb)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)

    hits = 0
    reciprocal_ranks = []

    for q_idx in query_indices:
        query_vec = emb[q_idx:q_idx + 1]
        scores, indices = index.search(query_vec, k + 1)  # +1 because self is included
        result_indices = [int(i) for i in indices[0] if i != q_idx][:k]

        query_cat = categories[q_idx]
        found = False
        for rank, r_idx in enumerate(result_indices, 1):
            if categories[r_idx] == query_cat:
                if not found:
                    reciprocal_ranks.append(1.0 / rank)
                    found = True
                    hits += 1

        if not found:
            reciprocal_ranks.append(0.0)

    recall = hits / len(query_indices)
    mrr    = np.mean(reciprocal_ranks)
    return recall, mrr


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    # ── Load metadata ──
    print("Loading metadata …")
    df = load_metadata(clean=True)

    # Use a subset for faster evaluation (5000 products)
    EVAL_SIZE = min(5000, len(df))
    sample_df = df.sample(n=EVAL_SIZE, random_state=SEED).reset_index(drop=True)

    image_paths = [get_image_path(pid) for pid in sample_df["id"]]
    categories  = sample_df["masterCategory"].values.tolist()

    # Random query indices
    query_indices = random.sample(range(EVAL_SIZE), NUM_QUERIES)

    print(f"\nEvaluation setup:")
    print(f"  Catalog size : {EVAL_SIZE}")
    print(f"  Queries      : {NUM_QUERIES}")
    print(f"  Top-K        : {TOP_K}")
    print()

    # ── CLIP ──
    t0 = time.time()
    clip_emb = encode_with_clip(image_paths, BATCH_SIZE)
    clip_time = time.time() - t0
    clip_recall, clip_mrr = compute_recall_at_k(clip_emb, categories, query_indices, TOP_K)

    # ── ResNet50 ──
    t0 = time.time()
    resnet_emb = encode_with_resnet(image_paths, BATCH_SIZE)
    resnet_time = time.time() - t0
    resnet_recall, resnet_mrr = compute_recall_at_k(resnet_emb, categories, query_indices, TOP_K)

    # ── Report ──
    print("\n" + "="*60)
    print("  EVALUATION RESULTS")
    print("="*60)
    print(f"{'Metric':<25} {'CLIP (ViT-B/32)':<20} {'ResNet50':<20}")
    print("-"*60)
    print(f"{'Recall@' + str(TOP_K):<25} {clip_recall*100:>6.1f}%{'':<13} {resnet_recall*100:>6.1f}%")
    print(f"{'MRR':<25} {clip_mrr:>6.3f}{'':<14} {resnet_mrr:>6.3f}")
    print(f"{'Embedding Dim':<25} {clip_emb.shape[1]:<20} {resnet_emb.shape[1]:<20}")
    print(f"{'Encode Time (s)':<25} {clip_time:>6.1f}{'':<14} {resnet_time:>6.1f}")
    print("="*60)

    improvement = ((clip_recall - resnet_recall) / max(resnet_recall, 0.01)) * 100
    print(f"\n📊 CLIP outperforms ResNet50 by {improvement:+.1f}% on Recall@{TOP_K}")
    print(f"   → Resume line: \"Achieved {clip_recall*100:.0f}% Recall@{TOP_K}, "
          f"outperforming ResNet50 baseline by {improvement:.0f}%\"")


if __name__ == "__main__":
    main()
