"""
app.py
------
Streamlit frontend for the Tourism RAG Assistant.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Ensure `src` is importable when Streamlit runs this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.common.config_loader import get_domain_config, get_settings  # noqa: E402
from src.retrieval.retriever import Retriever  # noqa: E402
from src.services.rag_engine import OllamaClient, RagEngine  # noqa: E402

st.set_page_config(
    page_title="Tourism RAG Assistant",
    page_icon="🧭",
    layout="wide",
)


# -----------------------------------------------------------------------------
# Cached resources
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_engine() -> RagEngine:
    return RagEngine()


def none_if_all(value: str):
    return None if value == "All" else value


# -----------------------------------------------------------------------------
# App state
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": ..., "content": ...}

settings = get_settings()
domain_cfg = get_domain_config()
app_meta = domain_cfg.get("app", {})

engine = load_engine()

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🧭 " + app_meta.get("name", "Tourism RAG Assistant"))

    st.divider()
    st.subheader("System status")

    collection_size = engine.retriever.collection_size()
    if collection_size > 0:
        st.success(f"Knowledge base ready — {collection_size} chunks indexed.")
    else:
        st.error(
            "Knowledge base is empty. Run `make ingest` (or "
            "`python -m src.ingestion.ingest_all`) before asking questions."
        )

    ollama_healthy = engine.ollama.health_check()
    if ollama_healthy:
        st.success(f"Ollama connected — model: `{settings.ollama_model}`")
    else:
        st.warning(
            f"Ollama not reachable at {settings.ollama_base_url}. "
            "Start it with `ollama serve` and pull the model with "
            f"`ollama pull {settings.ollama_model}`."
        )

    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    with st.expander("ℹ️ About this assistant"):
        covered = domain_cfg.get("safety", {}).get("covered_states", [])
        st.write(
            "This assistant answers using **only** the content of official "
            "tourism guidebooks and government reports for: "
            + ", ".join(covered)
            + ". It will not answer questions outside this scope, and every "
            "answer is grounded in retrieved source passages with citations."
        )

# Filters removed from the UI — always search the full knowledge base
# with the default top-k from settings.
selected_state = "All"
selected_region = "All"
selected_city = "All"
selected_doc_type = "All"
top_k = settings.retrieval_top_k


# -----------------------------------------------------------------------------
# Main chat area
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ---- Light theme (white background) ---- */
    .stApp {
        background-color: #FFFFFF;
        color: #1A1A1A;
    }
    section[data-testid="stSidebar"] {
        background-color: #F7F9FC;
    }

    .hero {
        background: linear-gradient(135deg, #2563EB 0%, #7DB9FF 100%);
        padding: 2.2rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .hero h1 {
        font-size: 2.1rem;
        margin-bottom: 0.3rem;
    }
    .hero p {
        font-size: 1rem;
        opacity: 0.95;
        margin: 0;
    }
    div[data-testid="stChatMessage"] {
        border-radius: 14px;
        padding: 0.4rem 0.2rem;
    }
    .footer-credit {
        text-align: left;
        color: #888888;
        font-size: 0.85rem;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #E5E7EB;
    }
    </style>

    <div class="hero">
        <h1>🧭 Ask Your Travel Guide</h1>
        <p>Your guide to exploring India — plan trips, discover places, and get grounded answers with citations.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption(
    "Try: *\"What are the best places to visit in Darjeeling?\"* or "
    "*\"How do I reach Kolkata by train?\"*or "
    "*\"Best Local Foods in Kolkata?\"*"
)

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("retrieved_chunks"):
            with st.expander("🔍 Retrieved context (debug view)"):
                for chunk in msg["retrieved_chunks"]:
                    st.markdown(
                        f"**{chunk['label']}** · score={chunk['score']}\n\n"
                        f"> {chunk['text'][:500]}{'...' if len(chunk['text']) > 500 else ''}"
                    )

# Chat input
user_question = st.chat_input("Ask a question about your trip...")

if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    history_for_prompt = [
        {"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]
    ]

    with st.chat_message("assistant"):
        with st.spinner("Searching the knowledge base and generating an answer..."):
            response = engine.answer(
                question=user_question,
                top_k=top_k,
                state=none_if_all(selected_state),
                region=none_if_all(selected_region),
                city=none_if_all(selected_city),
                doc_type=none_if_all(selected_doc_type),
                conversation_history=history_for_prompt,
            )

        st.markdown(response.answer)

        retrieved_chunks_display = []
        if response.retrieval and not response.retrieval.is_empty:
            with st.expander("🔍 Retrieved context (debug view)"):
                for chunk in response.retrieval.chunks:
                    label = chunk.citation_label()
                    st.markdown(
                        f"**{label}** · score={chunk.score}\n\n"
                        f"> {chunk.text[:500]}{'...' if len(chunk.text) > 500 else ''}"
                    )
                    retrieved_chunks_display.append(
                        {"label": label, "score": chunk.score, "text": chunk.text}
                    )

        st.caption(f"⏱️ {response.latency_seconds}s")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response.answer,
            "sources": response.sources,
            "retrieved_chunks": retrieved_chunks_display,
        }
    )

st.markdown(
    '<div class="footer-credit">Tourism RAG Assistant — designed &amp; built by Samiya Mondal</div>',
    unsafe_allow_html=True,
)