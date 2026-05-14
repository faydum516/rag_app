import os
import subprocess
import sys

import streamlit as st
from langchain_classic.memory import ConversationBufferMemory

from rag_core import (
    DATA_PATH,
    build_vector_store_from_documents,
    get_retriever_k,
    get_vector_store_backend,
    load_pdf_documents,
    load_vector_store,
    make_conversation_chain,
    openai_api_key_configured,
)

if __name__ == "__main__":
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except ImportError:
        get_script_run_ctx = lambda: None
    if get_script_run_ctx() is None:
        raise SystemExit(
            subprocess.call(
                [sys.executable, "-m", "streamlit", "run", __file__, *sys.argv[1:]]
            )
        )

st.set_page_config(page_title="Personal Knowledge RAG", layout="wide")

if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH, exist_ok=True)


def _ensure_api_key():
    env_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if env_key:
        os.environ["OPENAI_API_KEY"] = env_key
        return True
    try:
        secret_key = st.secrets.get("OPENAI_API_KEY")
    except (FileNotFoundError, KeyError, AttributeError):
        secret_key = None
    if secret_key and str(secret_key).strip():
        os.environ["OPENAI_API_KEY"] = str(secret_key).strip()
        return True
    api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    if api_key and api_key.strip():
        os.environ["OPENAI_API_KEY"] = api_key.strip()
        return True
    st.warning("Set OPENAI_API_KEY (env or secrets) or enter a key in the sidebar.")
    st.stop()
    return False


def _init_session_state():
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "conversation_memory" not in st.session_state:
        st.session_state.conversation_memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
        )
    if "vector_backend_sel" not in st.session_state:
        st.session_state.vector_backend_sel = None


def _reset_conversation():
    st.session_state.chat_history = []
    st.session_state.conversation_memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )


_ensure_api_key()
_init_session_state()

st.title("Personal Knowledge Assistant")
st.sidebar.header("Configuration")

backend = st.sidebar.selectbox(
    "Vector store",
    options=["faiss", "chroma"],
    index=0 if get_vector_store_backend() == "faiss" else 1,
    help="FAISS (local files) or Chroma (folder DB). Sets VECTOR_STORE for this session.",
)
os.environ["VECTOR_STORE"] = backend
if st.session_state.vector_backend_sel not in (None, backend):
    st.session_state.vectorstore = load_vector_store()
    _reset_conversation()
st.session_state.vector_backend_sel = backend

chat_model_inp = st.sidebar.text_input(
    "Chat model",
    value=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o"),
    help="Applied before each response. Re-index if you change the embedding model.",
)
embed_model_inp = st.sidebar.text_input(
    "Embedding model",
    value=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
)
os.environ["OPENAI_CHAT_MODEL"] = chat_model_inp.strip() or "gpt-4o"
os.environ["OPENAI_EMBEDDING_MODEL"] = embed_model_inp.strip() or "text-embedding-3-small"

k = st.sidebar.number_input("Retriever k", min_value=1, max_value=20, value=get_retriever_k())
os.environ["RAG_RETRIEVER_K"] = str(int(k))

st.sidebar.header("Knowledge base")
st.sidebar.caption(f"PDFs only, loaded from `{DATA_PATH}` (recursive).")

if st.sidebar.button("Index / update from PDFs"):
    with st.spinner("Indexing PDFs..."):
        if not openai_api_key_configured():
            st.sidebar.error("Set your OpenAI API key first (env, secrets, or sidebar).")
        else:
            docs = load_pdf_documents()
            if not docs:
                st.sidebar.error("No PDFs found under the data folder.")
            else:
                st.session_state.vectorstore = build_vector_store_from_documents(docs)
                if st.session_state.vectorstore is None:
                    st.sidebar.error("Indexing failed. Check your API key and try again.")
                else:
                    _reset_conversation()
                    st.sidebar.success("Indexed. Conversation memory was cleared.")

if st.sidebar.button("Load existing index"):
    st.session_state.vectorstore = load_vector_store()
    if st.session_state.vectorstore is None:
        st.sidebar.error("No saved index for this vector store backend.")
    else:
        _reset_conversation()
        st.sidebar.success("Loaded index. Conversation memory was cleared.")

if st.sidebar.button("Clear conversation (memory + UI)"):
    _reset_conversation()
    st.sidebar.success("Conversation cleared.")

if st.session_state.vectorstore is None:
    st.session_state.vectorstore = load_vector_store()

if st.session_state.vectorstore:
    chain = make_conversation_chain(
        st.session_state.vectorstore,
        st.session_state.conversation_memory,
    )

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask something about your PDFs..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            result = chain.invoke({"question": prompt})
            answer = result["answer"]
            sources = result.get("source_documents") or []

            st.markdown(answer)
            with st.expander("Sources"):
                for doc in sources:
                    source_name = os.path.basename(doc.metadata.get("source", "Unknown"))
                    preview = doc.page_content[:200].replace("\n", " ")
                    st.write(f"- **{source_name}**: {preview}...")

        st.session_state.chat_history.append({"role": "assistant", "content": answer})
else:
    st.info(
        "Add PDFs under the `data` folder, then use **Index / update from PDFs** "
        "or **Load existing index** in the sidebar."
    )
