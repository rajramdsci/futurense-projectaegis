# app/streamlit_app.py


import sys
from pathlib import Path

# Adds the root directory to the python path
root_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_path))


import warnings
import os

# Suppress transformers deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", message=".*__path__.*", module="transformers")
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
## ======= rest of the code =======

import streamlit as st
# from pathlib import Path
# import sys

# sys.path.append(str(Path(__file__).parent.parent))

from src.ingestion.pipeline import IngestionPipeline
from src.retrieval.pipeline import RetrievalPipeline
from config.settings import settings


st.set_page_config(
    page_title="Project Aegis - Enterprise RAG",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Project Aegis")
st.markdown("**Advanced Corporate Policy RAG System**")

# ====================== SIDEBAR ======================
with st.sidebar:
    st.header("⚙️ Configuration")

    # Ingestion Section
    with st.expander("📥 Ingestion Settings", expanded=False):
        st.markdown("**Chunking Parameters**")
        chunk_size = st.slider("Chunk Size", 500, 2500, settings.CHUNK_SIZE, 100)
        chunk_overlap_percent = st.slider("Chunk Overlap (%)", 0.05, 0.20, 
                                        settings.CHUNK_OVERLAP_PERCENT, 0.01, format="%.2f")
        
        col1, col2 = st.columns(2)
        with col1:
            limit_docs = st.number_input("Limit Documents", min_value=1, value=None)
        with col2:
            skip_embedding = st.checkbox("Skip Embedding", value=False)

        if st.button("🚀 Run Ingestion Pipeline", type="primary", use_container_width=True):
            with st.spinner("Processing..."):
                try:
                    pipe = IngestionPipeline()
                    pipe.chunker.chunk_size = chunk_size
                    pipe.chunker.chunk_overlap = int(chunk_size * chunk_overlap_percent)
                    chunks = pipe.run(limit=limit_docs, skip_embedding=skip_embedding)
                    st.success(f"✅ {len(chunks)} chunks created!")
                except Exception as e:
                    st.error(f"❌ {e}")

    # Retrieval Settings
    with st.expander("🔍 Retrieval Settings", expanded=True):
        st.markdown("**Query Strategy**")
        use_mqe = st.checkbox("Multi-Query Expansion (MQE)", value=True)
        use_hyde = st.checkbox("HyDE (Hypothetical Document)", value=True)
        
        st.divider()
        st.markdown("**Ranking & Filtering**")
        use_reranker = st.checkbox("Cohere Cross-Encoder Reranker", value=True)
        
        policy_category = st.selectbox(
            "Policy Category Filter",
            options=["None", "Security", "Training", "Travel", "Work Policies"],
            index=0
        )
        
        st.divider()
        st.markdown("**Parameters**")
        top_k = st.slider("Top K Candidates", 10, 50, 25, 5)
        final_top_n = st.slider("Final Results", 3, 10, 5, 1)

    st.divider()
    
    # === Reset Memory Button ===
    if st.button("🔄 Reset Memory & Start Fresh", type="secondary", use_container_width=True):
        if "messages" in st.session_state:
            st.session_state.messages = []
        if "pipeline" in st.session_state:
            st.session_state.pipeline.chat_history = []
        st.success("✅ Memory cleared! Starting fresh conversation.")
        st.rerun()

# ====================== MAIN CHAT ======================
st.header("💬 Aegis Policy Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm **Aegis**, your corporate policy assistant.\n\nHow can I help you today?"}
    ]

if "pipeline" not in st.session_state:
    st.session_state.pipeline = RetrievalPipeline()

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat Input
if prompt := st.chat_input("Ask anything about company policies..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            category_map = {
                "None": None,
                "Security": "security",
                "Training": "training",
                "Travel": "travel",
                "Work Policies": "work_policies"
            }

            result = st.session_state.pipeline.retrieve_and_answer(
                question=prompt,
                use_mqe=use_mqe,
                use_hyde=use_hyde,
                use_reranker=use_reranker,
                policy_category=category_map[policy_category],
                top_k=top_k,
                final_top_n=final_top_n
            )

            # Display response type
            if result.get("from_memory", False):
                st.info("💬 **Answering from conversation history**")
            else:
                categories_used = set()
                for chunk in result.get("final_results", []):
                    cat = chunk['metadata'].get('policy_category', 'Unknown')
                    if cat:
                        categories_used.add(cat.title())
                
                cat_str = ", ".join(sorted(categories_used)) if categories_used else "General"
                st.success(f"📄 **Policy Retrieval Used** • Categories: **{cat_str}**")
            
            st.markdown(result["answer"])

            # Show Sources
            if not result.get("from_memory", False) and result.get("final_results"):
                with st.expander("📚 View Retrieved Sources", expanded=False):
                    for i, chunk in enumerate(result["final_results"], 1):
                        header = chunk['metadata'].get('h1_header', 'Policy Document')
                        cat = chunk['metadata'].get('policy_category', 'General').title()
                        st.markdown(f"**{i}. {header}** ({cat})")
                        st.caption(chunk['chunk_text'][:400] + "..." if len(chunk['chunk_text']) > 400 else chunk['chunk_text'])

    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})

# Footer
st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Chat Turns", len(st.session_state.messages)//2)
with col2:
    st.metric("Method", result.get("method", "—") if 'result' in locals() else "—")
with col3:
    st.metric("Memory Size", len(st.session_state.pipeline.chat_history))