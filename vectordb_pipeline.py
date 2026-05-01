import os

from pinecone import Pinecone


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "mock-exam")
PINECONE_MODEL = os.getenv("PINECONE_MODEL", "llama-text-embed-v2")
DB_NAMES = {
    "hr": "hr_questions",
    "tech": "tech_questions",
    "realworld": "realworld_problems",
}


def _get_index():
    if not PINECONE_API_KEY:
        raise RuntimeError("PINECONE_API_KEY is not set")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    return pc.Index(PINECONE_INDEX)


def _embed_query(text: str) -> list[float]:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    result = pc.inference.embed(
        model=PINECONE_MODEL,
        inputs=[text],
        parameters={"input_type": "query"},
    )
    return result.data[0]["values"]


def _query_collection(index, query_text, category, n_results=2):
    if not query_text or not query_text.strip():
        return []

    try:
        query_vec = _embed_query(query_text)
        results = index.query(
            top_k=n_results,
            vector=query_vec,
            filter={"category": {"$eq": category}},
            include_metadata=True,
        )
    except Exception:
        return []

    matches = results.get("matches", []) if isinstance(results, dict) else getattr(results, "matches", [])
    if not matches:
        return []

    out = []
    for i, match in enumerate(matches, start=1):
        metadata = getattr(match, "metadata", None) or match.get("metadata", {})
        score = getattr(match, "score", None) or match.get("score")
        item_id = getattr(match, "id", None) or match.get("id") or f"{category}_{i}"
        question = metadata.get("text", "") or metadata.get("question", "")
        out.append(
            {
                "id": item_id,
                "question": question,
                "answer": metadata.get("answer", ""),
                "distance": score,
                "category": metadata.get("category", category),
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
    index = _get_index()

    effective_query = query_text if query_text else cv_profile.get("position", "software engineer")
    queried_entities = {}
    errors = {}
    collection_counts = {}
    for cat in DB_NAMES.keys():
        collection_counts[cat] = 0
        per_cat_n = n_results_by_category.get(cat, n_results) if n_results_by_category else n_results
        if per_cat_n <= 0:
            queried_entities[cat] = []
            continue
        queried_entities[cat] = _query_collection(index, effective_query, cat, n_results=per_cat_n)

    return {
        "query_text": effective_query,
        "collection_counts": collection_counts,
        "queried_entities": queried_entities,
    }
