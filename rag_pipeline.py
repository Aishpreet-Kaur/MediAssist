"""
Handles document loading, chunking, embedding, vector store creation
and retrieval for the MediAssist RAG pipeline.
"""

import os
from pathlib import Path
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# ── Constants ─────────────────────────────────────────────────────────────────
DOCS_DIR       = "data/medical_docs"
VECTORSTORE_DIR = "vectorstore/faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # free, no API key needed
CHUNK_SIZE     = 600
CHUNK_OVERLAP  = 80


# ── Embeddings (loaded once) ───────────────────────────────────────────────────
def get_embeddings():
    """Load HuggingFace sentence-transformer embeddings."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# ── Build & Save Vector Store ──────────────────────────────────────────────────


# ── Retrieve Relevant Chunks ───────────────────────────────────────────────────
def retrieve_context(query: str, vectorstore, k: int = 5) -> str:
    """
    Retrieve the top-k most relevant document chunks for a given query.
    Returns a single concatenated string of context passages.
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(query)
    context = "\n\n---\n\n".join([doc.page_content for doc in docs])
    return context


if __name__ == "__main__":
    # Run this directly to test the pipeline
    build_vectorstore()
    vs = load_vectorstore()
    print("\n[TEST] Retrieving context for: 'fever headache body ache'")
    print(retrieve_context("fever headache body ache", vs))