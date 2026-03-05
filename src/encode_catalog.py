"""
encode_catalog.py — Encode all catalog images into CLIP 512-dim embeddings.

Features:
  • Batch processing with configurable batch size.
  • Checkpoint saving every N batches (resume-safe).
  • Progress bar via tqdm.

Usage:
    python src/encode_catalog.py
"""

import os
import sys
import glob
import time

import numpy as np
import torch
import open_clip
from PIL import Image
from tqdm import tqdm

# ── project imports ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import load_metadata, get_image_path, EMBED_DIR

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
MODEL_NAME      = "ViT-B-32"
PRETRAINED      = "openai"                # weights trained by OpenAI
BATCH_SIZE      = 64
CHECKPOINT_EVERY = 16                     # save a checkpoint every 16 batches (~1024 images)
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    # ── Load metadata ──
    print("Loading metadata …")
    df = load_metadata(clean=True)
    product_ids = df["id"].values
    total = len(product_ids)
    print(f"Total images to encode: {total}")

    # ── Load CLIP model ──
    print(f"Loading CLIP model ({MODEL_NAME}, {PRETRAINED}) on {DEVICE} …")
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED, device=DEVICE
    )
    model.eval()
    print("Model loaded ✓")

    # ── Check for existing checkpoints (resume logic) ──
    checkpoint_pattern = os.path.join(EMBED_DIR, "checkpoint_*.npz")
    existing_checkpoints = sorted(glob.glob(checkpoint_pattern))

    if existing_checkpoints:
        last_ckpt = existing_checkpoints[-1]
        print(f"Resuming from checkpoint: {last_ckpt}")
        data = np.load(last_ckpt)
        all_embeddings = list(data["embeddings"])
        all_ids        = list(data["ids"])
        start_idx      = len(all_ids)
        print(f"  Already encoded: {start_idx}/{total}")
    else:
        all_embeddings = []
        all_ids        = []
        start_idx      = 0

    # ── Encode in batches ──
    remaining_ids = product_ids[start_idx:]
    num_batches   = (len(remaining_ids) + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"Encoding {len(remaining_ids)} remaining images in {num_batches} batches …")
    t0 = time.time()

    for batch_idx in tqdm(range(num_batches), desc="Encoding"):
        batch_start = batch_idx * BATCH_SIZE
        batch_end   = min(batch_start + BATCH_SIZE, len(remaining_ids))
        batch_ids   = remaining_ids[batch_start:batch_end]

        images = []
        valid_ids = []
        for pid in batch_ids:
            try:
                img = Image.open(get_image_path(pid)).convert("RGB")
                images.append(preprocess(img))
                valid_ids.append(pid)
            except Exception as e:
                print(f"\n  ⚠ Skipping image {pid}: {e}")
                continue

        if not images:
            continue

        image_tensor = torch.stack(images).to(DEVICE)

        with torch.no_grad():
            embeddings = model.encode_image(image_tensor)
            embeddings = embeddings.cpu().numpy()

        all_embeddings.append(embeddings)
        all_ids.extend(valid_ids)

        # ── Checkpoint ──
        if (batch_idx + 1) % CHECKPOINT_EVERY == 0:
            _save_checkpoint(all_embeddings, all_ids)

    # ── Final save ──
    all_embeddings_np = np.vstack(all_embeddings).astype(np.float32)
    all_ids_np        = np.array(all_ids)

    os.makedirs(EMBED_DIR, exist_ok=True)
    np.save(os.path.join(EMBED_DIR, "image_embeddings.npy"), all_embeddings_np)
    np.save(os.path.join(EMBED_DIR, "image_ids.npy"),        all_ids_np)

    elapsed = time.time() - t0
    print(f"\n✅ Done! Encoded {len(all_ids_np)} images in {elapsed:.1f}s")
    print(f"   Embeddings shape: {all_embeddings_np.shape}")
    print(f"   Saved to: {EMBED_DIR}")

    # Clean up checkpoint files
    for ckpt in glob.glob(os.path.join(EMBED_DIR, "checkpoint_*.npz")):
        os.remove(ckpt)
    print("   Checkpoint files cleaned up.")


def _save_checkpoint(embeddings_list, ids_list):
    """Save a checkpoint of the current encoding progress."""
    os.makedirs(EMBED_DIR, exist_ok=True)
    emb = np.vstack(embeddings_list).astype(np.float32)
    ids = np.array(ids_list)
    path = os.path.join(EMBED_DIR, f"checkpoint_{len(ids):06d}.npz")
    np.savez_compressed(path, embeddings=emb, ids=ids)
    tqdm.write(f"  💾 Checkpoint saved: {len(ids)} images → {path}")


if __name__ == "__main__":
    main()
