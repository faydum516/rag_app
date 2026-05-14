"""Shared RAG utilities for the Streamlit app and offline evaluation."""

from __future__ import annotations

import os
from typing import Literal, Optional

from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import Chroma, FAISS
from langchain_core.vectorstores import VectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

VectorStoreBackend = Literal["faiss", "chroma"]

DATA_PATH = os.getenv("RAG_DATA_PATH", "./data")
FAISS_PATH = os.getenv("RAG_FAISS_PATH", "./faiss_index")
CHROMA_PATH = os.getenv("RAG_CHROMA_PATH", "./chroma_db")


def openai_api_key_configured() -> bool:
    """OpenAI clients (embeddings / chat) require a non-empty key at construction time."""
    return bool((os.getenv("OPENAI_API_KEY") or "").strip())


def get_vector_store_backend() -> VectorStoreBackend:
    backend = os.getenv("VECTOR_STORE", "faiss").strip().lower()
    if backend not in ("faiss", "chroma"):
        return "faiss"
    return backend  # type: ignore[return-value]


def get_chat_model_name() -> str:
    return os.getenv("OPENAI_CHAT_MODEL", "gpt-4o").strip()


def get_embedding_model_name() -> str:
    return os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small").strip()


def get_retriever_k() -> int:
    return int(os.getenv("RAG_RETRIEVER_K", "3"))


def get_embeddings() -> OpenAIEmbeddings:
    if not openai_api_key_configured():
        raise RuntimeError("OPENAI_API_KEY is not set. Export it or enter it in the app sidebar.")
    return OpenAIEmbeddings(model=get_embedding_model_name())


def get_llm() -> ChatOpenAI:
    return ChatOpenAI(model=get_chat_model_name(), temperature=0)


def load_pdf_documents(data_path: str = DATA_PATH):
    """Load PDFs only from ``data_path`` (recursive)."""
    loader = PyPDFDirectoryLoader(data_path, glob="**/*.pdf")
    return loader.load()


def split_documents(documents, chunk_size: int = 1000, chunk_overlap: int = 200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    return splitter.split_documents(documents)


def _faiss_exists(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "index.faiss")) and os.path.isfile(
        os.path.join(path, "index.pkl")
    )


def build_vector_store_from_documents(documents, backend: Optional[VectorStoreBackend] = None):
    """Chunk documents and persist a vector store."""
    if not documents:
        return None
    if not openai_api_key_configured():
        return None
    backend = backend or get_vector_store_backend()
    chunks = split_documents(documents)
    emb = get_embeddings()
    if backend == "chroma":
        store = Chroma.from_documents(
            documents=chunks,
            embedding=emb,
            persist_directory=CHROMA_PATH,
        )
        return store
    store = FAISS.from_documents(chunks, emb)
    os.makedirs(FAISS_PATH, exist_ok=True)
    store.save_local(FAISS_PATH)
    return store


def load_vector_store(backend: Optional[VectorStoreBackend] = None) -> Optional[VectorStore]:
    """Load a persisted vector store if files exist."""
    if not openai_api_key_configured():
        return None
    backend = backend or get_vector_store_backend()
    emb = get_embeddings()
    if backend == "faiss":
        if not _faiss_exists(FAISS_PATH):
            return None
        return FAISS.load_local(
            FAISS_PATH,
            emb,
            allow_dangerous_deserialization=True,
        )
    if not os.path.isdir(CHROMA_PATH):
        return None
    return Chroma(persist_directory=CHROMA_PATH, embedding=emb)


def make_conversation_chain(
    vectorstore: VectorStore,
    memory: ConversationBufferMemory,
):
    return ConversationalRetrievalChain.from_llm(
        llm=get_llm(),
        retriever=vectorstore.as_retriever(search_kwargs={"k": get_retriever_k()}),
        memory=memory,
        return_source_documents=True,
    )
