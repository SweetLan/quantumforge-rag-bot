from __future__ import annotations

from app.rag_pipeline import ask_rag


def ask_bot(question: str) -> str:
    return ask_rag(question)["answer"]
