from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _load_embedder():
    return SentenceTransformer(MODEL_NAME)


def _feedback_from_score(score_pct: float) -> str:
    if score_pct >= 70:
        return "Good answer. You covered key ideas. Add one concrete project example for stronger impact."
    if score_pct >= 45:
        return "Partially correct answer. Improve structure using context, action, and result."
    return "Weak alignment with expected answer. Clarify concept, explain steps, and include practical details."


def run_evaluation_pipeline(queried_entities: dict, user_answers: dict) -> dict:
    """Global Evaluation function: queried entities + user answers -> scores and summary."""
    embedder = _load_embedder()

    evaluations = {}
    scores = []

    for cat, items in queried_entities.items():
        for item in items:
            qid = item.get("id") or f"{cat}_{abs(hash(item.get('question', 'q')))}"
            user_answer = user_answers.get(qid, "").strip()
            ideal_answer = item.get("answer", "")
            if not user_answer:
                continue

            vecs = embedder.encode([user_answer, ideal_answer])
            cos_sim = float(np.dot(vecs[0], vecs[1]) / (np.linalg.norm(vecs[0]) * np.linalg.norm(vecs[1])))
            score_pct = round(cos_sim * 100, 1)
            scores.append(score_pct)

            evaluations[qid] = {
                "question": item.get("question", ""),
                "category": cat,
                "semantic_similarity": score_pct,
                "feedback": _feedback_from_score(score_pct),
                "label": "Strong" if cos_sim > 0.6 else ("Partial" if cos_sim > 0.35 else "Weak"),
                "ideal_answer": ideal_answer,
            }

    average_score = round(sum(scores) / len(scores), 1) if scores else 0.0
    return {
        "evaluations": evaluations,
        "questions_answered": len(scores),
        "average_score": average_score,
    }
