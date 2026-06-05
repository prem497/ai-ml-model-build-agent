"""
ChromaDB vector store — stores pipeline plan embeddings for similarity search.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")


def _get_client():
    try:
        import sys
        # Streamlit Cloud uses an older sqlite3. Override with pysqlite3-binary
        try:
            __import__('pysqlite3')
            sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
        except ImportError:
            pass

        import chromadb
        from chromadb.config import Settings
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        return client
    except Exception:
        return None


def _get_collection(client):
    try:
        return client.get_or_create_collection(
            name="pipeline_plans",
            metadata={"hnsw:space": "cosine"},
        )
    except Exception:
        return None


def store_pipeline_embedding(
    run_id: str,
    user_input: str,
    plan: Dict[str, Any],
) -> bool:
    """
    Embed the user_input + plan summary and store in ChromaDB.
    Returns True on success.
    """
    client = _get_client()
    if client is None:
        return False
    collection = _get_collection(client)
    if collection is None:
        return False

    try:
        doc = f"{user_input} | intent:{plan.get('intent')} | model:{plan.get('model', {}).get('type')}"
        collection.upsert(
            ids=[run_id],
            documents=[doc],
            metadatas=[{
                "run_id":     run_id,
                "user_input": user_input[:200],
                "intent":     plan.get("intent", ""),
                "model_type": plan.get("model", {}).get("type", ""),
            }],
        )
        return True
    except Exception:
        return False


def search_similar_pipelines(
    query: str,
    n_results: int = 3,
) -> List[Dict[str, Any]]:
    """
    Return the most similar past pipeline plans for a given query string.
    """
    client = _get_client()
    if client is None:
        return []
    collection = _get_collection(client)
    if collection is None:
        return []

    try:
        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        items = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            items.append({
                "document": doc,
                "metadata": meta,
                "similarity": round(1 - dist, 4),
            })
        return items
    except Exception:
        return []
