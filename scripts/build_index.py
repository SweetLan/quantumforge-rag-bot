from __future__ import annotations

import time
import warnings
from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core._api import LangChainDeprecationWarning
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)


KNOWLEDGE_BASE_DIR = Path("knowledge_base")
INDEX_DIR = Path("data/faiss_index")
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def find_source_files(directory: Path) -> list[Path]:
    if not directory.exists():
        raise SystemExit(f"Ошибка: папка базы знаний не найдена: {directory}")
    if not directory.is_dir():
        raise SystemExit(f"Ошибка: путь базы знаний не является папкой: {directory}")

    files = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in {".md", ".txt"}
    )
    if not files:
        raise SystemExit(f"Ошибка: в папке {directory} нет файлов .md или .txt")

    return files


def load_documents(files: list[Path]) -> list[Document]:
    documents: list[Document] = []

    for file_path in files:
        text = file_path.read_text(encoding="utf-8-sig").strip()
        if not text:
            print(f"Пропущен пустой файл: {file_path}")
            continue

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": str(file_path),
                    "title": file_path.stem,
                },
            )
        )

    if not documents:
        raise SystemExit("Ошибка: все файлы базы знаний пустые.")

    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks: list[Document] = []
    for document in documents:
        document_chunks = splitter.split_documents([document])
        for chunk_id, chunk in enumerate(document_chunks):
            chunk.metadata = {
                **chunk.metadata,
                "chunk_id": chunk_id,
            }
            chunks.append(chunk)

    return chunks


def main() -> None:
    start_time = time.perf_counter()

    source_files = find_source_files(KNOWLEDGE_BASE_DIR)
    print(f"Найдено документов: {len(source_files)}")

    documents = load_documents(source_files)
    chunks = split_documents(documents)
    print(f"Создано чанков: {len(chunks)}")

    print(f"Загружается embedding-модель: {EMBEDDING_MODEL_NAME}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    print("Создаётся FAISS-индекс...")
    vector_store = FAISS.from_documents(chunks, embeddings)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(INDEX_DIR))

    elapsed = time.perf_counter() - start_time
    print(f"Индекс сохранён в: {INDEX_DIR}")
    print(f"Генерация заняла: {elapsed:.2f} сек.")


if __name__ == "__main__":
    main()
