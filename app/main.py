from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.rag_pipeline import (
    EmptyQuestionError,
    IndexNotFoundError,
    LLMNotConfiguredError,
    LLMRequestError,
    ask_rag,
)

app = FastAPI(title="QuantumForge RAG Bot")


class QuestionRequest(BaseModel):
    question: str


@app.post("/ask")
def ask(request: QuestionRequest):
    try:
        return ask_rag(request.question)
    except EmptyQuestionError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except IndexNotFoundError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    except LLMNotConfiguredError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    except LLMRequestError as error:
        raise HTTPException(status_code=error.status_code, detail=str(error)) from error
