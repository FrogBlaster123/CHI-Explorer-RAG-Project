import sys
import os
import streamlit as st

# Add the src directory to path so we can import backend modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from embedder import Embedder
from retriever import BaselineRetriever
from llm_client import SimpleGeminiClient
from main import run_query

# ───────────────────────────────────────────────
# Page Configuration
# ───────────────────────────────────────────────
st.set_page_config(
    page_title="CHI Papers Research Assistant",
    page_icon="📚",
    layout="centered"
)

# ───────────────────────────────────────────────
# Custom Styling
# ───────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #888;
        margin-bottom: 2rem;
    }
    .answer-box {
        background-color: #0e1117;
        border: 1px solid #333;
        border-radius: 10px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }
    .source-label {
        font-size: 0.85rem;
        color: #aaa;
    }
    .metric-row {
        display: flex;
        gap: 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────
# Cached Initialization (runs ONCE, survives reruns)
# ───────────────────────────────────────────────
@st.cache_resource
def init_backend():
    """Initialize the embedder, retriever, and LLM client once."""
    embedder = Embedder()
    retriever = BaselineRetriever(embedder)
    llm_client = SimpleGeminiClient()
    return retriever, llm_client

def get_available_datasets():
    """Scan the data/indexes/ folder for indexed datasets."""
    index_root = os.path.join(os.path.dirname(__file__), "data", "indexes")
    if not os.path.exists(index_root):
        return []
    return [d for d in os.listdir(index_root) if os.path.isdir(os.path.join(index_root, d))]

# ───────────────────────────────────────────────
# Header
# ───────────────────────────────────────────────
st.markdown('<p class="main-header">📚 CHI Papers Research Assistant</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Ask questions across 500+ CHI research papers. Answers are grounded with exact citations.</p>', unsafe_allow_html=True)

# ───────────────────────────────────────────────
# Sidebar Controls
# ───────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    
    datasets = get_available_datasets()
    if not datasets:
        st.error("No indexed datasets found. Run indexing first via CLI.")
        st.stop()
        
    dataset_name = st.selectbox("Dataset", datasets, index=0)
    top_k = st.slider("Top K results", min_value=3, max_value=10, value=3)
    
    st.divider()
    st.caption("Built with FAISS + Gemini 2.5 Flash")

# ───────────────────────────────────────────────
# Initialize Backend
# ───────────────────────────────────────────────
retriever, llm_client = init_backend()

# ───────────────────────────────────────────────
# Query Input
# ───────────────────────────────────────────────
query = st.text_input("🔎 Enter your research query", placeholder="e.g. What are the main interface challenges in AR?")

search_clicked = st.button("Search", type="primary", use_container_width=True)

# ───────────────────────────────────────────────
# Query Execution & Results
# ───────────────────────────────────────────────
if search_clicked:
    if not query.strip():
        st.error("Please enter a non-empty query.")
    else:
        with st.spinner("🔍 Searching papers..."):
            result = run_query(query, dataset_name, retriever, llm_client, top_k=top_k)
        
        if result is None:
            st.error("❌ Could not load the index. Please run `--mode index` first via CLI.")
        else:
            # ── Answer Section ──
            st.markdown("---")
            st.subheader("🧠 Answer")
            st.markdown(result["answer"])
            
            # ── Sources Section ──
            st.markdown("---")
            st.subheader(f"📄 Sources ({len(result['sources'])} retrieved)")
            
            if not result["sources"]:
                st.info("No relevant results found.")
            else:
                for i, src in enumerate(result["sources"]):
                    score_display = f"{src['score']:.4f}"
                    label = f"📎 {src['source']} — Page {src['page']} (L2 Dist: {score_display})"
                    
                    with st.expander(label, expanded=(i == 0)):
                        st.caption(f"Year: {src['year']}")
                        st.markdown(f"```\n{src['text'][:300]}...\n```")
            
            # ── Debug Section ──
            with st.expander("🔍 Debug: Full Retrieved Chunks", expanded=False):
                for i, src in enumerate(result["sources"]):
                    st.markdown(f"**Rank {i+1}** | L2 Distance: `{src['score']:.4f}` | `{src['source']}` (Page {src['page']})")
                    st.text(src["text"])
                    st.divider()
