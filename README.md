# 🔍 Visual Product Search Engine

> Find visually similar products using OpenAI CLIP and FAISS vector search — supports image upload AND natural language text queries.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](YOUR_STREAMLIT_URL)

![Demo GIF](assets/demo.gif)  <!-- Replace with your recorded GIF -->

---

## ✨ Key Features

| Feature | Description |
|---|---|
| **Image Search** | Upload any product photo → get top-K visually similar items |
| **Text Search** | Type "red running shoes" → CLIP finds matching products |
| **Cross-Modal AI** | One model, one index powers both modalities |
| **MMR Diversity** | Maximal Marginal Relevance ensures varied, non-redundant results |
| **"Find Similar" Chain** | Click any result to discover more like it |
| **Category Filters** | Filter by category, gender, or both |
| **Latency Breakdown** | See exactly where time is spent: encoding / search / post-process |
| **44K+ Products** | Searched in < 50ms using FAISS Inner Product index |

---

## 🏗️ Architecture

```text
User uploads image / types query
       │
       ▼
┌─────────────────────┐
│  CLIP Model Encoder │  ← open-clip-torch (ViT-B/32)
│  (PyTorch)          │     Image/Text → 512-dim embedding
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  FAISS Vector Index │  ← IndexFlatIP on L2-normalised vectors
│  (Facebook AI)      │     ≈ cosine similarity, < 50ms
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Post-Filter + MMR   │  ← Category/gender filtering + diversity
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Streamlit Frontend │  ← Cards, scores, latency, "Find Similar"
└─────────────────────┘
```

### How It Works (3 Steps)

1. **Encode** — All 44K catalog images are encoded offline into 512-dimensional CLIP embeddings.
2. **Index** — Embeddings are L2-normalised and stored in a FAISS Inner Product index for sub-linear search.
3. **Search** — At query time, the input (image or text) is encoded by CLIP, searched against the index, and results are diversified using MMR.

---

## 📊 Evaluation Results

| Metric | CLIP (ViT-B/32) | ResNet50 Baseline |
|---|---|---|
| **Recall@5** | ~90%+ | ~60% |
| **MRR** | ~0.85+ | ~0.55 |
| **Embedding Dim** | 512 | 2048 |
| **Avg Encode Time** | ~25ms | ~15ms |

> CLIP outperforms ResNet50 by **~30-50%** on same-category Recall@5, while using 4× fewer dimensions.

---

## 🛠️ Tech Stack

- **Model:** OpenAI CLIP (ViT-B/32) via `open-clip-torch`
- **Vector Search:** Facebook FAISS (`faiss-cpu`)
- **Framework:** PyTorch
- **Frontend:** Streamlit
- **Data:** Kaggle Fashion Product Images (44,446 items)
- **Evaluation:** scikit-learn, custom Recall@K pipeline

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/product-image-search.git
cd product-image-search
pip install -r requirements.txt
```

### 2. Download Dataset

Download [Fashion Product Images (Small)](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small) from Kaggle and place:
- Images → `data/images/`
- Metadata → `data/styles.csv`

### 3. Run the Pipeline

```bash
# Step 1: Clean metadata
python src/utils.py  # or run the notebook

# Step 2: Encode all images (30-60 min on CPU, checkpoint-safe)
python src/encode_catalog.py

# Step 3: Build FAISS index
python src/build_index.py

# Step 4: Evaluate (optional)
python src/evaluate.py

# Step 5: Launch the app
streamlit run app.py
```

---

## 📁 Project Structure

```text
product-image-search/
├── data/
│   ├── images/                ← 44K product images
│   └── styles.csv             ← Product metadata
├── embeddings/
│   ├── image_embeddings.npy   ← CLIP embeddings (44K × 512)
│   ├── image_ids.npy          ← Index → product ID mapping
│   └── faiss_index.bin        ← FAISS index
├── src/
│   ├── encode_catalog.py      ← Encodes images (with checkpointing)
│   ├── build_index.py         ← Builds FAISS index
│   ├── search_engine.py       ← Search logic (MMR, filtering, latency)
│   ├── evaluate.py            ← Recall@K benchmark vs ResNet50
│   └── utils.py               ← Shared helpers
├── app.py                     ← Streamlit frontend
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🎯 Performance

- **Search Latency:** < 50ms for 44K products
- **CLIP Encoding:** ~25ms per image (CPU)
- **Embedding Dimension:** 512
- **Index Type:** FAISS Flat Inner Product (exact search)
- **Scalability:** Can upgrade to IVF-PQ for millions of products

---
<img width="2510" height="1207" alt="image" src="https://github.com/user-attachments/assets/de9f2985-c2b3-4aa4-bec4-29421ac01ee3" />

## 📝 License

MIT License — free to use, modify, and distribute.
