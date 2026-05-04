from __future__ import annotations

import warnings
from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core._api import LangChainDeprecationWarning


warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)


INDEX_DIR = Path("data/faiss_index")
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_ENCODE_KWARGS = {"normalize_embeddings": True}
TOP_K = 3

TEST_QUERIES = [
    "Кто такой Арлен Вейр?",
    "Что такое Астрогор?",
    "Как работает Осколок души?",
]


def load_index() -> FAISS:
    if not INDEX_DIR.exists():
        raise SystemExit(
            f"Ошибка: индекс не найден в {INDEX_DIR}. "
            "Сначала запустите: python scripts/build_index.py"
        )

    required_files = [INDEX_DIR / "index.faiss", INDEX_DIR / "index.pkl"]
    missing_files = [path for path in required_files if not path.exists()]
    if missing_files:
        missing = ", ".join(str(path) for path in missing_files)
        raise SystemExit(
            f"Ошибка: индекс в {INDEX_DIR} неполный. Не найдены файлы: {missing}. "
            "Пересоздайте индекс командой: python scripts/build_index.py"
        )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"local_files_only": True},
        encode_kwargs=EMBEDDING_ENCODE_KWARGS,
    )
    return FAISS.load_local(
        str(INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def print_result(query: str, result_number: int, document, score: float) -> None:
    metadata = document.metadata
    print("-" * 80)
    print(f"Запрос: {query}")
    print(f"Результат #{result_number}, score: {score:.4f}")
    print(f"source: {metadata.get('source')}")
    print(f"title: {metadata.get('title')}")
    print(f"chunk_id: {metadata.get('chunk_id')}")
    print()
    print(document.page_content)
    print()


def main() -> None:
    print(f"Загружается FAISS-индекс из: {INDEX_DIR}")
    vector_store = load_index()

    for query in TEST_QUERIES:
        print("=" * 80)
        print(f"Тестовый запрос: {query}")
        results = vector_store.similarity_search_with_score(query, k=TOP_K)

        if not results:
            print("Ничего не найдено.")
            continue

        for index, (document, score) in enumerate(results, start=1):
            print_result(query, index, document, score)


if __name__ == "__main__":
    main()
