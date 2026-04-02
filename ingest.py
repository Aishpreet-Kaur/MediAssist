"""
ingest.py
---------
One-time script to build the FAISS vector store from your medical documents.
Run this ONCE before starting the app, or whenever you add new documents.

Usage:
    python ingest.py
"""

from rag_pipeline import build_vectorstore

if __name__ == "__main__":
    print("=" * 60)
    print("  MediAssist — Knowledge Base Ingestion")
    print("=" * 60)
    print()
    vectorstore = build_vectorstore()
    print()
    print("[SUCCESS] Knowledge base built successfully!")
    print("You can now run the app with: streamlit run app.py")
    print("=" * 60)