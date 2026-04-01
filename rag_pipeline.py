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
def build_vectorstore():
    """
    Load all .txt files from DOCS_DIR, chunk them,
    embed them, and save a FAISS index to disk.
    """
    print("[INFO] Loading documents...")
    loader = DirectoryLoader(
        DOCS_DIR,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    docs = loader.load()
    print(f"[INFO] Loaded {len(docs)} document(s).")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(docs)
    print(f"[INFO] Split into {len(chunks)} chunks.")

    print("[INFO] Building embeddings (this may take a minute on first run)...")
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    Path(VECTORSTORE_DIR).parent.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(VECTORSTORE_DIR)
    print(f"[INFO] Vector store saved to '{VECTORSTORE_DIR}'.")
    return vectorstore


# ── Load Existing Vector Store ─────────────────────────────────────────────────
def load_vectorstore():
    """Load FAISS index from disk. Build it first if it doesn't exist."""
    embeddings = get_embeddings()
    if not Path(VECTORSTORE_DIR).exists():
        print("[WARN] Vector store not found. Building now...")
        return build_vectorstore()
    return FAISS.load_local(
        VECTORSTORE_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )


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