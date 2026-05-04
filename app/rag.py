from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

docs = [
    Document(
        page_content="QuantumForge Software разрабатывает SaaS-платформу Digital Twin для моделирования промышленных объектов.",
        metadata={"source": "company_context"},
    ),
    Document(
        page_content="У QuantumForge Software есть проблемы с разрозненной документацией, дублями и устареванием страниц через 2-3 месяца.",
        metadata={"source": "company_context"},
    ),
    Document(
        page_content="Цель RAG-бота — сократить время поиска информации, повысить продуктивность сотрудников и выявлять пробелы в документации.",
        metadata={"source": "company_context"},
    ),
]

db = FAISS.from_documents(docs, embedding_model)


def ask_bot(question: str) -> str:
    results = db.similarity_search_with_score(question, k=3)

    # берём только достаточно похожие
    relevant_docs = [doc for doc, score in results if score < 0.5]

    if not relevant_docs:
        return "Я не знаю"

    context = "\n".join(doc.page_content for doc in relevant_docs)

    return f"Ответ на основе базы знаний:\n\n{context}\n\nИсточник: company_context"