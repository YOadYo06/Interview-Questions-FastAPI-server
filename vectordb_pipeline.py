import os
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions


CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_dbs")
MODEL_NAME = "all-MiniLM-L6-v2"
DB_NAMES = {
    "hr": "hr_questions",
    "tech": "tech_questions",
    "realworld": "realworld_problems",
}


def _get_client():
    Path(CHROMA_PATH).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=CHROMA_PATH)


def _get_or_create_collection(client, name):
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=MODEL_NAME)
    return client.get_or_create_collection(name=name, embedding_function=ef)


def _query_collection(collection, query_text, category, n_results=2):
    if not query_text or not query_text.strip():
        return []

    try:
        if collection.count() == 0:
            return []
    except Exception:
        return []

    try:
        results = collection.query(query_texts=[query_text], n_results=n_results)
    except Exception as e:
        msg = str(e).lower()
        if "nothing found on disk" in msg or "error creating hnsw segment reader" in msg:
            return []
        raise

    docs = results.get("documents", [[]])
    metas = results.get("metadatas", [[]])
    dists = results.get("distances", [[]])
    if not docs or not docs[0]:
        return []

    out = []
    for i, doc in enumerate(docs[0]):
        md = metas[0][i] if metas and metas[0] and i < len(metas[0]) else {}
        dist = dists[0][i] if dists and dists[0] and i < len(dists[0]) else None
        item_id = md.get("id", f"{category}_{i+1}")
        out.append(
            {
                "id": item_id,
                "question": doc,
                "answer": md.get("answer", ""),
                "distance": dist,
                "category": md.get("category", category),
            }
        )
    return out


def run_vectordb_pipeline(
    cv_profile: dict,
    query_text: str | None = None,
    n_results: int = 2,
    n_results_by_category: dict[str, int] | None = None,
) -> dict:
    """Global VectorDB function: profile -> queried entities from existing Chroma collections."""
    client = _get_client()

    effective_query = query_text if query_text else cv_profile.get("position", "software engineer")
    queried_entities = {}
    errors = {}
    collection_counts = {}
    for cat, db_name in DB_NAMES.items():
        collection = _get_or_create_collection(client, db_name)
        try:
            collection_counts[cat] = collection.count()
        except Exception:
            collection_counts[cat] = 0
        per_cat_n = n_results_by_category.get(cat, n_results) if n_results_by_category else n_results
        if per_cat_n <= 0:
            queried_entities[cat] = []
            continue
        queried_entities[cat] = _query_collection(collection, effective_query, cat, n_results=per_cat_n)

    return {
        "query_text": effective_query,
        "collection_counts": collection_counts,
        "queried_entities": queried_entities,
    }
