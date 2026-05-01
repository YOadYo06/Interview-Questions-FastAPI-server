import json
import os
from pathlib import Path
from typing import Any

import pdfplumber
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pinecone import Pinecone
from dotenv import load_dotenv

from evaluation_pipeline import run_evaluation_pipeline
from nlp_pipeline import run_nlp_pipeline
from vectordb_pipeline import run_vectordb_pipeline


APP_ROOT = Path(__file__).resolve().parent
DATA_DIR = APP_ROOT / "data"
load_dotenv(APP_ROOT / ".env")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "mock-exam")
PINECONE_MODEL = os.getenv("PINECONE_MODEL", "llama-text-embed-v2")


def _parse_origins() -> list[str]:
    raw = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000")
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


app = FastAPI(title="Mock Exam FastAPI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_parse_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextNlpRequest(BaseModel):
    cv_text: str = Field(..., min_length=1)


class VectorDbRequest(BaseModel):
    cv_profile: dict[str, Any]
    query_text: str | None = None
    n_results: int = Field(default=2, ge=1, le=10)


class EvaluationRequest(BaseModel):
    queried_entities: dict[str, Any]
    user_answers: dict[str, str]


class SingleEvaluationRequest(BaseModel):
    question: str = Field(..., min_length=1)
    ideal_answer: str = Field(..., min_length=1)
    user_answer: str = Field(..., min_length=1)


class GenerateQuestionsRequest(BaseModel):
    position: str = Field(..., min_length=1)
    description: str | None = None
    experience_years: int | None = None
    tech_stack: list[str] = Field(default_factory=list)
    n_results: int = Field(default=2, ge=1, le=10)
    hr_count: int | None = Field(default=None, ge=0, le=10)
    tech_count: int | None = Field(default=None, ge=0, le=10)
    realworld_count: int | None = Field(default=None, ge=0, le=10)
    query_text: str | None = None


def _get_pinecone_index():
    if not PINECONE_API_KEY:
        raise HTTPException(status_code=500, detail="PINECONE_API_KEY is not set")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    return pc.Index(PINECONE_INDEX)


def _embed_texts(texts: list[str]) -> list[list[float]]:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    all_vectors: list[list[float]] = []
    batch_size = 96
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        result = pc.inference.embed(
            model=PINECONE_MODEL,
            inputs=batch,
            parameters={"input_type": "passage"},
        )
        all_vectors.extend([item["values"] for item in result.data])
    return all_vectors


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/nlp/text")
def nlp_from_text(payload: TextNlpRequest) -> dict[str, Any]:
    return {"profile": run_nlp_pipeline(payload.cv_text)}


@app.post("/nlp/pdf")
def nlp_from_pdf(file: UploadFile = File(...)) -> dict[str, Any]:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF file is required")

    try:
        with pdfplumber.open(file.file) as pdf:
            cv_text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read PDF: {exc}")

    if not cv_text.strip():
        raise HTTPException(status_code=400, detail="No text extracted from PDF")

    return {"profile": run_nlp_pipeline(cv_text)}


@app.post("/vectordb/query")
def vectordb_query(payload: VectorDbRequest) -> dict[str, Any]:
    return run_vectordb_pipeline(
        cv_profile=payload.cv_profile,
        query_text=payload.query_text,
        n_results=payload.n_results,
    )


@app.post("/evaluate")
def evaluate_answers(payload: EvaluationRequest) -> dict[str, Any]:
    return run_evaluation_pipeline(payload.queried_entities, payload.user_answers)


@app.post("/evaluate/single")
def evaluate_single(payload: SingleEvaluationRequest) -> dict[str, Any]:
    queried = {
        "single": [
            {
                "id": "single_1",
                "question": payload.question,
                "answer": payload.ideal_answer,
                "category": "single",
            }
        ]
    }
    user_answers = {"single_1": payload.user_answer}
    result = run_evaluation_pipeline(queried, user_answers)
    evaluation = result.get("evaluations", {}).get("single_1", {})
    score_pct = float(evaluation.get("semantic_similarity", 0.0))
    rating = max(1.0, min(10.0, round(score_pct / 10.0, 1)))
    return {
        "rating": rating,
        "feedback": evaluation.get("feedback", ""),
        "score_pct": score_pct,
        "label": evaluation.get("label", ""),
    }


@app.post("/interviews/generate")
def generate_questions(payload: GenerateQuestionsRequest) -> dict[str, Any]:
    cv_profile = {
        "position": payload.position,
        "description": payload.description or "",
        "experience_years": payload.experience_years or 0,
        "tech_stack": payload.tech_stack,
    }
    n_results_by_category = {
        "hr": payload.hr_count,
        "tech": payload.tech_count,
        "realworld": payload.realworld_count,
    }
    n_results_by_category = {
        key: value for key, value in n_results_by_category.items() if value is not None
    }

    vectordb_result = run_vectordb_pipeline(
        cv_profile=cv_profile,
        query_text=payload.query_text,
        n_results=payload.n_results,
        n_results_by_category=n_results_by_category if n_results_by_category else None,
    )
    questions = []
    for items in vectordb_result.get("queried_entities", {}).values():
        for item in items:
            questions.append(
                {
                    "question": item.get("question", ""),
                    "answer": item.get("answer", ""),
                    "category": item.get("category", ""),
                }
            )

    return {
        "questions": questions,
        "query_text": vectordb_result.get("query_text", ""),
        "collection_counts": vectordb_result.get("collection_counts", {}),
        "returned_counts": {
            "hr": len(vectordb_result.get("queried_entities", {}).get("hr", [])),
            "tech": len(vectordb_result.get("queried_entities", {}).get("tech", [])),
            "realworld": len(vectordb_result.get("queried_entities", {}).get("realworld", [])),
        },
        "errors": vectordb_result.get("errors", {}),
    }


@app.post("/db/reset-load")
def reset_and_load_db() -> dict[str, Any]:
    hr_data = _load_json(DATA_DIR / "hr_data.json")
    tech_data = _load_json(DATA_DIR / "tech_data.json")
    real_data = _load_json(DATA_DIR / "real_data.json")

    try:
        _upsert_pinecone(hr_data, "hr", reset=True)
        _upsert_pinecone(tech_data, "tech", reset=False)
        _upsert_pinecone(real_data, "realworld", reset=False)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pinecone upsert failed: {exc}")

    return {
        "status": "ok",
        "counts": {
            "hr": len(hr_data),
            "tech": len(tech_data),
            "realworld": len(real_data),
        },
    }


def _load_json(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Missing data file: {path.name}")
    return json.loads(path.read_text(encoding="utf-8"))


def _upsert_collection(collection, data: list[dict[str, Any]], category: str) -> None:
    collection.upsert(
        ids=[item["id"] for item in data],
        documents=[item.get("question", "") for item in data],
        metadatas=[
            {
                "answer": item.get("answer", ""),
                "domain": item.get("domain", ""),
                "category": category,
            }
            for item in data
        ],
    )


def _upsert_pinecone(data: list[dict[str, Any]], category: str, reset: bool) -> None:
    index = _get_pinecone_index()
    if reset:
        try:
            index.delete(delete_all=True)
        except Exception:
            pass

    if not data:
        return

    texts = [item.get("question", "") for item in data]
    vectors = _embed_texts(texts)

    payload = []
    for item, vector in zip(data, vectors, strict=False):
        payload.append(
            (
                item["id"],
                vector,
                {
                    "answer": item.get("answer", ""),
                    "domain": item.get("domain", ""),
                    "category": category,
                    "text": item.get("question", ""),
                },
            )
        )

    if not payload:
        return

    batch_size = 100
    for start in range(0, len(payload), batch_size):
        batch = payload[start:start + batch_size]
        index.upsert(vectors=batch)


@app.post("/db/load")
def load_db_data() -> dict[str, Any]:
    hr_data = _load_json(DATA_DIR / "hr_data.json")
    tech_data = _load_json(DATA_DIR / "tech_data.json")
    real_data = _load_json(DATA_DIR / "real_data.json")

    _upsert_pinecone(hr_data, "hr", reset=False)
    _upsert_pinecone(tech_data, "tech", reset=False)
    _upsert_pinecone(real_data, "realworld", reset=False)

    return {
        "status": "ok",
        "counts": {
            "hr": len(hr_data),
            "tech": len(tech_data),
            "realworld": len(real_data),
        },
    }
