from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag_pipeline import (
    EmptyQuestionError,
    IndexNotFoundError,
    ask_rag,
)


def main() -> None:
    print("RAG-бот. Введите вопрос или exit для выхода.")
    while True:
        question = input("> ").strip()
        if question.lower() in {"exit", "quit", "q"}:
            break
        if not question:
            print("Пустой вопрос")
            continue

        try:
            result = ask_rag(question)
        except (EmptyQuestionError, IndexNotFoundError) as error:
            print(error)
            continue

        print(f"Режим: {result['mode']}")
        print(result["answer"])
        print()


if __name__ == "__main__":
    main()
