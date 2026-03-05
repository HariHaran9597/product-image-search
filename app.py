"""
Streamlit frontend for the Visual Product Search Engine
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

# S3 image base path
S3_IMAGE_BASE = "https://visual-search-models-hariharan-2026.s3.amazonaws.com/product_images"

# Page config
st.set_page_config(
    page_title="Visual Product Search Engine",
    page_icon="🔍",
    layout="wide"
)

# ---------------------------------------------------------
# Load Engine
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Loading CLIP model & FAISS index …")
def load_engine():
    return SearchEngine()

engine = load_engine()

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.title("🔍 Visual Product Search")
st.caption("Upload an image or describe a product to find visually similar items")

total = engine.total_products()
cats  = len(engine.get_categories())

col1, col2, col3 = st.columns(3)
col1.metric("Products", f"{total:,}")
col2.metric("Categories", cats)
col3.metric("Embedding Dim", "512")

# ---------------------------------------------------------
# Sidebar Filters
# ---------------------------------------------------------
with st.sidebar:

    st.header("Filters")

    top_k = st.slider("Number of results", 1, 10, 5)

    category_filter = st.selectbox(
        "Category",
        ["All"] + engine.get_categories()
    )

    gender_filter = st.selectbox(
        "Gender",
        ["All"] + engine.get_genders()
    )

    diversity_on = st.toggle("Diverse results (MMR)", True)

cat_val = None if category_filter == "All" else category_filter
gen_val = None if gender_filter == "All" else gender_filter

# ---------------------------------------------------------
# Session state
# ---------------------------------------------------------
if "find_similar_id" not in st.session_state:
    st.session_state.find_similar_id = None


# ---------------------------------------------------------
# Result Rendering
# ---------------------------------------------------------
def render_results(results, latency):

    if latency:
        st.caption(
            f"CLIP: {latency['encode_ms']} ms | "
            f"FAISS: {latency['search_ms']} ms | "
            f"Post: {latency['post_ms']} ms | "
            f"Total: {latency['total_ms']} ms"
        )

    if not results:
        st.warning("No results found")
        return

    cols = st.columns(5)

    for i, res in enumerate(results):

        with cols[i % 5]:

            img_url = f"{S3_IMAGE_BASE}/{res['product_id']}.jpg"

            try:
                st.image(img_url)
            except:
                st.info("Image not available")

            st.markdown(f"**{res['product_name']}**")

            st.caption(
                f"{res['category']} • {res['sub_category']} • {res['article_type']}"
            )

            st.caption(
                f"{res['colour']} • {res['gender']} • {res['season']}"
            )

            st.caption(f"Match: {res['score']}%")

            if st.button("Find Similar", key=f"sim_{res['product_id']}"):
                st.session_state.find_similar_id = res["product_id"]
                st.rerun()


# ---------------------------------------------------------
# Similar chain
# ---------------------------------------------------------
if st.session_state.find_similar_id:

    pid = st.session_state.find_similar_id

    st.info(f"Showing products similar to Product #{pid}")

    if st.button("Back to search"):
        st.session_state.find_similar_id = None
        st.rerun()

    results, latency = engine.search_by_product_id(
        pid,
        top_k=top_k,
        category=cat_val,
        gender=gen_val,
        diversity=diversity_on
    )

    render_results(results, latency)

# ---------------------------------------------------------
# Main search tabs
# ---------------------------------------------------------
else:

    tab_image, tab_text = st.tabs(["Image Search", "Text Search"])

    # -----------------------------------------------------
    # Image search
    # -----------------------------------------------------
    with tab_image:

        uploaded = st.file_uploader(
            "Upload product image",
            type=["jpg", "jpeg", "png", "webp"]
        )

        if uploaded:

            try:

                query_img = Image.open(uploaded).convert("RGB")

                col1, col2 = st.columns([1, 3])

                with col1:
                    st.image(query_img, caption="Query Image")

                with col2:

                    with st.spinner("Searching..."):

                        results, latency = engine.search_by_image(
                            query_img,
                            top_k=top_k,
                            category=cat_val,
                            gender=gen_val,
                            diversity=diversity_on
                        )

                    render_results(results, latency)

            except Exception as e:

                st.error(f"Could not process image: {e}")

    # -----------------------------------------------------
    # Text search
    # -----------------------------------------------------
    with tab_text:

        query = st.text_input(
            "Describe the product",
            placeholder="e.g. red running shoes"
        )

        if query:

            with st.spinner("Searching..."):

                results, latency = engine.search_by_text(
                    query,
                    top_k=top_k,
                    category=cat_val,
                    gender=gen_val,
                    diversity=diversity_on
                )

            render_results(results, latency)

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")

st.caption(
    "Powered by CLIP + FAISS | Visual Product Search Engine"
)