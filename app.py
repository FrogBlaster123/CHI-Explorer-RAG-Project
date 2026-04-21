import os
import sys
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from retrieval.retriever import HybridRetriever
from generation.llm_client import AssistantLLM

@st.cache_resource
def load_system():
    retriever = HybridRetriever()
    llm = AssistantLLM()
    return retriever, llm

st.set_page_config(page_title="CHI Research Assistant", page_icon="🎓", layout="wide")

st.title("🎓 CHI Research Assistant (2020-2024)")
st.markdown("Ask complex questions about CHI research papers. The system leverages Hybrid Retrieval, Cross-Encoder Reranking, and the Gemma LLM to provide grounded answers.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "chunks" in msg and msg["chunks"]:
            with st.expander("📚 View Retrieved Sources"):
                for i, c in enumerate(msg["chunks"]):
                    st.markdown(f"**[{c['year']}_{c['title']}]** - *{c['section']}*")
                    st.caption(f"{c['text']}")

# Input
if prompt := st.chat_input("E.g., What are the core challenges of eye-tracking interfaces?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    retriever, llm = load_system()
    
    with st.chat_message("assistant"):
        with st.spinner("Retrieving literature... (Hybrid Search + Reranking)"):
            chunks = retriever.retrieve(prompt, top_k=5)
            
        with st.spinner("Synthesizing grounded answer..."):
            answer = llm.generate_answer(prompt, chunks)
            
        st.markdown(answer)
        if chunks:
            with st.expander("📚 View Retrieved Sources"):
                for i, c in enumerate(chunks):
                    st.markdown(f"**[{c['year']}_{c['title']}]** - *{c['section']}*")
                    st.caption(f"{c['text']}")
                    
    st.session_state.messages.append({"role": "assistant", "content": answer, "chunks": chunks})
