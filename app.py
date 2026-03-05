"""
app.py — Streamlit frontend for the Visual Product Search Engine.

Features:
  • Image Upload search  (snap a photo → find similar products)
  • Text search          (describe what you want in words)
  • "Find Similar" chain  (click any result to discover more like it)
  • Category & gender filters
  • Diversity toggle (MMR)
  • Latency breakdown
"""

import os
import sys
import streamlit as st
from PIL import Image
from download_models import download_files

download_files()

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from search_engine import SearchEngine
from utils import get_image_path

# ──────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Visual Product Search Engine",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global */
    html, body, .stApp {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 40%, #16213e 100%);
    }

    /* Hero Header */
    .hero-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea, #764ba2, #f093fb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .hero-sub {
        text-align: center;
        color: #a0a0c0;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }

    /* Metrics Bar */
    .metrics-bar {
        display: flex;
        justify-content: center;
        gap: 2.5rem;
        margin-bottom: 2rem;
    }
    .metric-item {
        text-align: center;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #667eea;
    }
    .metric-label {
        font-size: 0.78rem;
        color: #808099;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }

    /* Result Card */
    .result-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1rem;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .result-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.15);
    }
    .result-card img {
        border-radius: 12px;
        width: 100%;
    }
    .score-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .product-name {
        font-weight: 600;
        color: #e0e0f0;
        font-size: 0.92rem;
        margin: 0.4rem 0 0.2rem 0;
    }
    .product-meta {
        color: #808099;
        font-size: 0.78rem;
    }

    /* Latency */
    .latency-bar {
        display: flex;
        gap: 1.5rem;
        justify-content: center;
        margin: 1rem 0;
    }
    .latency-item {
        font-size: 0.78rem;
        color: #a0a0c0;
    }
    .latency-item span {
        color: #667eea;
        font-weight: 600;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(15, 12, 41, 0.95);
    }

    /* Search mode tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        justify-content: center;
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.05);
        border-radius: 10px;
        color: #a0a0c0;
        padding: 0.5rem 1.5rem;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
    }

    /* Example chips */
    .example-chip {
        display: inline-block;
        background: rgba(102,126,234,0.12);
        border: 1px solid rgba(102,126,234,0.3);
        color: #a0b4ff;
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 0.82rem;
        margin: 3px;
        cursor: pointer;
        transition: background 0.2s;
    }
    .example-chip:hover {
        background: rgba(102,126,234,0.25);
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #505070;
        font-size: 0.75rem;
        margin-top: 3rem;
        padding: 1rem;
        border-top: 1px solid rgba(255,255,255,0.05);
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# AWS S3 Artifact Streaming
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner="Syncing ML artifacts from AWS S3 (first run only)…")
def pull_models_from_s3():
    """Stream FAISS index and embeddings from S3 if they don't exist locally."""
    if "AWS" not in st.secrets:
        return

    import boto3
    from src.utils import EMBED_DIR
    
    aws_conf = st.secrets["AWS"]
    bucket_name = aws_conf["S3_BUCKET_NAME"]
    
    s3 = boto3.client(
        "s3",
        aws_access_key_id=aws_conf["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=aws_conf["AWS_SECRET_ACCESS_KEY"],
        region_name=aws_conf.get("AWS_REGION", "ap-south-1")
    )
    
    os.makedirs(EMBED_DIR, exist_ok=True)
    files_to_download = ["faiss_index.bin", "image_embeddings.npy", "image_ids.npy"]
    
    for filename in files_to_download:
        local_path = os.path.join(EMBED_DIR, filename)
        if not os.path.exists(local_path):
            st.info(f"Downloading {filename} from S3... (This may take a minute)")
            s3.download_file(bucket_name, filename, local_path)

# ──────────────────────────────────────────────
# Load Engine (cached)
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading CLIP model & FAISS index …")
def load_engine():
    pull_models_from_s3()
    return SearchEngine()

engine = load_engine()


# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────
st.markdown('<h1 class="hero-title">🔍 Visual Product Search</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Upload a product image or describe what you\'re looking for — powered by OpenAI CLIP & FAISS</p>', unsafe_allow_html=True)

# Metrics bar
total = engine.total_products()
cats  = len(engine.get_categories())
st.markdown(f"""
<div class="metrics-bar">
    <div class="metric-item"><div class="metric-value">{total:,}</div><div class="metric-label">Products</div></div>
    <div class="metric-item"><div class="metric-value">{cats}</div><div class="metric-label">Categories</div></div>
    <div class="metric-item"><div class="metric-value">&lt;50ms</div><div class="metric-label">Avg Latency</div></div>
    <div class="metric-item"><div class="metric-value">512</div><div class="metric-label">Embedding Dim</div></div>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Sidebar — filters
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Filters")
    top_k = st.slider("Number of results", 1, 10, 5)
    category_filter = st.selectbox("Category", ["All"] + engine.get_categories())
    gender_filter   = st.selectbox("Gender",   ["All"] + engine.get_genders())
    diversity_on    = st.toggle("Diverse results (MMR)", value=True)

    st.markdown("---")
    st.markdown("### 💡 How It Works")
    with st.expander("Architecture", expanded=False):
        st.markdown("""
        1. **Encode** — Your image/text is converted to a 512-dim vector by CLIP.
        2. **Search** — FAISS finds the nearest vectors among 44K+ products.
        3. **Diversify** — MMR ensures you see varied results, not duplicates.
        """)

cat_val = None if category_filter == "All" else category_filter
gen_val = None if gender_filter   == "All" else gender_filter


# ──────────────────────────────────────────────
# "Find Similar" state
# ──────────────────────────────────────────────
if "find_similar_id" not in st.session_state:
    st.session_state.find_similar_id = None


def render_results(results, latency):
    """Display the result cards and latency breakdown."""

    # Latency breakdown
    st.markdown(f"""
    <div class="latency-bar">
        <div class="latency-item">CLIP Encode: <span>{latency['encode_ms']}ms</span></div>
        <div class="latency-item">FAISS Search: <span>{latency['search_ms']}ms</span></div>
        <div class="latency-item">Post-process: <span>{latency['post_ms']}ms</span></div>
        <div class="latency-item">Total: <span>{latency['total_ms']}ms</span></div>
    </div>
    """, unsafe_allow_html=True)

    if not results:
        st.warning("No results found. Try a different query or adjust filters.")
        return

    cols = st.columns(min(len(results), 5))
    for i, res in enumerate(results):
        with cols[i % len(cols)]:
            # Product image
            if os.path.exists(res["image_path"]):
                st.image(res["image_path"], use_container_width=True)
            else:
                st.info("Image not available")

            # Score badge
            st.markdown(f'<span class="score-badge">{res["score"]}% match</span>', unsafe_allow_html=True)

            # Product info
            st.markdown(f'<div class="product-name">{res["product_name"]}</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="product-meta">{res["category"]} › {res["sub_category"]} › {res["article_type"]}<br>'
                f'🎨 {res["colour"]}  •  {res["gender"]}  •  {res["season"]}</div>',
                unsafe_allow_html=True
            )

            # "Find Similar" button
            if st.button("🔗 Find Similar", key=f"sim_{res['product_id']}"):
                st.session_state.find_similar_id = res["product_id"]
                st.rerun()


# ──────────────────────────────────────────────
# Handle "Find Similar" chain results
# ──────────────────────────────────────────────
if st.session_state.find_similar_id:
    pid = st.session_state.find_similar_id
    st.info(f"🔗 Showing products similar to **Product #{pid}**")

    col_clear, _ = st.columns([1, 4])
    with col_clear:
        if st.button("← Back to search"):
            st.session_state.find_similar_id = None
            st.rerun()

    results, latency = engine.search_by_product_id(
        pid, top_k=top_k, category=cat_val, gender=gen_val, diversity=diversity_on
    )
    render_results(results, latency)

else:
    # ──────────────────────────────────────────────
    # Main search tabs
    # ──────────────────────────────────────────────
    tab_image, tab_text = st.tabs(["📷 Image Search", "✏️ Text Search"])

    with tab_image:
        uploaded = st.file_uploader("Upload a product image", type=["jpg", "jpeg", "png", "webp"])
        if uploaded:
            try:
                query_img = Image.open(uploaded).convert("RGB")
                col_preview, col_results = st.columns([1, 3])
                with col_preview:
                    st.image(query_img, caption="Your query", use_container_width=True)
                with col_results:
                    with st.spinner("🔍 Searching …"):
                        results, latency = engine.search_by_image(
                            query_img, top_k=top_k, category=cat_val,
                            gender=gen_val, diversity=diversity_on
                        )
                    render_results(results, latency)
            except Exception as e:
                st.error(f"Could not process the image: {e}")

    with tab_text:
        st.markdown(
            '<div style="margin-bottom:0.5rem;">'
            '<span class="example-chip">red running shoes</span>'
            '<span class="example-chip">blue formal shirt</span>'
            '<span class="example-chip">black leather bag</span>'
            '<span class="example-chip">summer floral dress</span>'
            '</div>',
            unsafe_allow_html=True
        )
        text_query = st.text_input("Describe what you're looking for …", placeholder="e.g. white sneakers")
        if text_query:
            with st.spinner("🔍 Searching …"):
                results, latency = engine.search_by_text(
                    text_query, top_k=top_k, category=cat_val,
                    gender=gen_val, diversity=diversity_on
                )
            render_results(results, latency)


# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────
st.markdown(f"""
<div class="footer">
    Powered by <strong>OpenAI CLIP</strong> (ViT-B/32) &nbsp;•&nbsp; <strong>FAISS</strong> &nbsp;•&nbsp; <strong>Streamlit</strong><br>
    Searching across <strong>{total:,}</strong> products &nbsp;|&nbsp; 512-dim embeddings &nbsp;|&nbsp; MMR diversity
</div>
""", unsafe_allow_html=True)
