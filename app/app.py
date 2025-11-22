import os
import faiss
import torch
import requests
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import streamlit as st

# ---------------------------------------------------------------
# LOAD CONFIG
# ---------------------------------------------------------------
CONFIG_PATH = Path("config.json")
if not CONFIG_PATH.exists():
    st.error("Missing config.json. Please create it in the project root.")
    st.stop()

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

API_BASE = CONFIG["api_base"]
MODELS = CONFIG["models"]
MODEL_DISPLAY = CONFIG.get("model_display_names", {})
DEFAULT_MODEL = CONFIG["default_model"]
EMBEDDING_MODEL_NAME = CONFIG["embedding_model"]
TOP_K = CONFIG.get("top_k", 5)

INDEX_PATH = Path("app/faiss_index/index.faiss")
CHUNKS_PATH = Path("app/faiss_index/chunks.txt")

# ---------------------------------------------------------------
# CONNECTION STATUS CHECK
# ---------------------------------------------------------------
def check_connection():
    """Ping the LM Studio API base to test connectivity."""
    try:
        r = requests.get(f"{API_BASE}/v1/models", timeout=3)
        if r.status_code == 200:
            return True
    except requests.exceptions.RequestException:
        return False
    return False

# ---------------------------------------------------------------
# LOAD INDEX AND EMBEDDING MODEL
# ---------------------------------------------------------------
@st.cache_resource
def load_retriever():
    if not INDEX_PATH.exists() or not CHUNKS_PATH.exists():
        st.error("Missing FAISS index or chunks.txt. Run ingest_data.py first.")
        st.stop()

    index = faiss.read_index(str(INDEX_PATH))
    chunks, sources = [], []
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                src, text = parts
                chunks.append(text)
                sources.append(src)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
    return index, model, chunks, sources


# ---------------------------------------------------------------
# RETRIEVE CONTEXT
# ---------------------------------------------------------------
def get_relevant_context(query, index, model, chunks, sources, top_k=TOP_K):
    query_vector = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vector, top_k)
    context_chunks = [chunks[i] for i in indices[0] if 0 <= i < len(chunks)]
    return "\n\n".join(context_chunks)


# ---------------------------------------------------------------
# STREAMING LM STUDIO RESPONSE
# ---------------------------------------------------------------
def stream_lmstudio_response(prompt, model_name):
    """Stream tokens from LM Studio in real time using JSON lines."""
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "stream": True
    }

    try:
        with requests.post(f"{API_BASE}/v1/chat/completions", json=payload, stream=True, timeout=10) as response:
            if response.status_code != 200:
                yield f"[ERROR] {response.status_code}: {response.text}"
                return

            for line in response.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[len("data: "):].strip()
                if data_str == "[DONE]":
                    break
                try:
                    data_json = json.loads(data_str)
                    delta = data_json["choices"][0]["delta"]
                    token = delta.get("content", "")
                    if token:
                        yield token
                except Exception:
                    continue

    except requests.exceptions.ConnectionError:
        yield "⚠️ Unable to contact the AI for this query. Please contact the app owner to gain access."
    except requests.exceptions.RequestException as e:
        yield f"⚠️ Network error: {e}. Please contact the app owner to gain access."



# ---------------------------------------------------------------
# RESET LM STUDIO CONTEXT
# ---------------------------------------------------------------
def reset_lmstudio_context():
    """Attempt to reset model context in LM Studio (if supported)."""
    try:
        r = requests.post(f"{API_BASE}/v1/chat/reset", timeout=5)
        if r.status_code == 200:
            st.success("✅ LM Studio context reset successfully.")
        else:
            st.warning("⚠️ Reset endpoint not supported. Memory cleared locally only.")
    except Exception:
        st.warning("⚠️ Could not contact LM Studio reset endpoint. Local memory cleared.")


# ---------------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------------
st.set_page_config(page_title="Local RAG Chatbot", layout="centered")
st.title("Local RAG Chatbot (Multi-Model Streaming)")
st.caption("Live, local RAG chatbot powered by LM Studio + FAISS")

index, model, chunks, sources = load_retriever()

# Sidebar controls
st.sidebar.header("🧠 Session Controls")

# ---------------------------------------------------------------
# CONNECTION STATUS INDICATOR
# ---------------------------------------------------------------
connected = check_connection()
if connected:
    st.sidebar.markdown("**STATUS:** 🟢 Connected")
else:
    st.sidebar.markdown("**STATUS:** 🔴 Offline — contact the app owner to gain access.")

# Model dropdown
display_names = [MODEL_DISPLAY.get(mid, mid) for mid in MODELS]
model_lookup = {MODEL_DISPLAY.get(mid, mid): mid for mid in MODELS}
default_display_name = MODEL_DISPLAY.get(DEFAULT_MODEL, DEFAULT_MODEL)
selected_display = st.sidebar.selectbox("Active Model:", display_names, index=display_names.index(default_display_name))
selected_model = model_lookup[selected_display]

# Clear memory button
if st.sidebar.button("🧹 Clear Memory & Reset Chat"):
    reset_lmstudio_context()
    st.session_state.history = []
    st.rerun()

# ---------------------------------------------------------------
# CHAT INTERFACE
# ---------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

chat_container = st.container()

# Display chat history
with chat_container:
    for role, message in st.session_state.history:
        if role == "User":
            with st.chat_message("user"):
                st.markdown(message)
        else:
            with st.chat_message("assistant"):
                st.markdown(message)

# User input
query = st.chat_input("Ask a question about your local knowledge base...")

if query:
    st.session_state.history.append(("User", query))
    with st.chat_message("user"):
        st.markdown(query)

    with st.spinner("Retrieving context..."):
        context = get_relevant_context(query, index, model, chunks, sources)
        prompt = f"""You are a helpful assistant using local knowledge.

CONTEXT:
{context}

QUESTION:
{query}

If the answer cannot be found in the context, say "I don't have that information locally."
"""

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        partial_response = ""
        for token in stream_lmstudio_response(prompt, selected_model):
            partial_response += token
            response_placeholder.markdown(partial_response + "▌")
        response_placeholder.markdown(partial_response)

    st.session_state.history.append(("Assistant", partial_response))

st.divider()
st.caption("Powered by LM Studio • Streamlit • FAISS • Sentence Transformers")
