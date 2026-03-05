"""
utils.py — Helper functions for image loading, preprocessing, and path management.
"""

import os
import pandas as pd
from PIL import Image

# ──────────────────────────────────────────────
# Path Constants
# ──────────────────────────────────────────────
PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR      = os.path.join(PROJECT_ROOT, "data")
IMAGES_DIR    = os.path.join(DATA_DIR, "images")
EMBED_DIR     = os.path.join(PROJECT_ROOT, "embeddings")
STYLES_CSV    = os.path.join(DATA_DIR, "styles.csv")
CLEAN_CSV     = os.path.join(DATA_DIR, "styles_clean.csv")


def load_metadata(clean=True):
    """Load the product metadata CSV."""
    path = CLEAN_CSV if clean else STYLES_CSV
    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found at {path}")
    return pd.read_csv(path, on_bad_lines="skip")


def get_image_path(product_id):
    """Return the full path to a product image given its ID."""
    return os.path.join(IMAGES_DIR, f"{product_id}.jpg")


def image_exists(product_id):
    """Check whether the image file for a product actually exists on disk."""
    return os.path.exists(get_image_path(product_id))


def load_image(product_id):
    """Load a product image as a PIL RGB Image."""
    path = get_image_path(product_id)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    return Image.open(path).convert("RGB")


def clean_metadata(df):
    """
    Clean the raw styles.csv:
      - Drop rows whose image files are missing on disk.
      - Fill NaN in colour/season with 'Unknown'.
      - Reset index.
    Returns a new DataFrame.
    """
    print(f"  Raw rows: {len(df)}")

    # Keep only rows that have a corresponding image
    df = df[df["id"].apply(image_exists)].copy()
    print(f"  After removing missing images: {len(df)}")

    # Fill missing values
    fill_cols = {"baseColour": "Unknown", "season": "Unknown", "usage": "Unknown"}
    for col, default in fill_cols.items():
        if col in df.columns:
            df[col] = df[col].fillna(default)

    df.reset_index(drop=True, inplace=True)
    return df
