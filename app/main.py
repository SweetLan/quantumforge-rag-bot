from fastapi import FastAPI
from pydantic import BaseModel

from app.rag import ask_bot

app = FastAPI(title="QuantumForge RAG Bot")


class QuestionRequest(BaseModel):
    question: str


@app.post("/ask")
def ask(request: QuestionRequest):
    answer = ask_bot(request.question)
    return {"answer": answer}