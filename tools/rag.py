from __future__ import annotations

import os
import tempfile
from typing import Optional, Dict, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from llm.base import get_embedding_model

# ----------------------------
# Configuration
# ----------------------------
BASE_RAG_DIR = os.path.join(os.getcwd(), "rag_store")
os.makedirs(BASE_RAG_DIR, exist_ok=True)

embeddings = get_embedding_model()


# ----------------------------
# Helpers
# ----------------------------
def _thread_dir(thread_id: str) -> str:
    return os.path.join(BASE_RAG_DIR, str(thread_id))


def _index_exists(thread_id: str) -> bool:
    return os.path.exists(os.path.join(_thread_dir(thread_id), "index.faiss"))


def _load_vectorstore(thread_id: str) -> FAISS:
    return FAISS.load_local(
        folder_path=_thread_dir(thread_id),
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )


# ----------------------------
# PDF ingestion (persistent)
# ----------------------------
def ingest_pdf(
    file_bytes: bytes,
    thread_id: str,
    filename: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Ingest a PDF and persist a FAISS index per thread.
    """
    if not file_bytes:
        raise ValueError("Empty PDF upload")

    os.makedirs(_thread_dir(thread_id), exist_ok=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(file_bytes)
        temp_path = f.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = splitter.split_documents(docs)

        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(_thread_dir(thread_id))

        return {
            "thread_id": thread_id,
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


# ----------------------------
# RAG Tool (persistent)
# ----------------------------
@tool
def rag_tool(query: str, thread_id: Optional[str] = None) -> dict:
    """
    Retrieve relevant context from persisted PDF embeddings.
    """
    if not thread_id or not _index_exists(thread_id):
        return {
            "error": "No document indexed for this thread. Upload a PDF first.",
            "query": query,
        }

    vectorstore = _load_vectorstore(thread_id)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(query)

    return {
        "query": query,
        "context": [d.page_content[:800] for d in docs],
        "metadata": [d.metadata for d in docs],
        "source_file": thread_id,
    }


# ----------------------------
# Public helpers
# ----------------------------
def thread_has_document(thread_id: str) -> bool:
    return _index_exists(thread_id)
