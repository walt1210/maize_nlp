"""
Retrieval layer: queries the pre-built ChromaDB knowledge base for the
top-K chunks relevant to a given diagnosis. Read-only at runtime — the
DB itself is built offline by rag/ingest.py and baked into the Docker image.
"""
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

import config

_vectorstore = None  # lazy-initialized singleton, one embedding client per process


def _get_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is None:
        embeddings = GoogleGenerativeAIEmbeddings(
            model=config.EMBEDDING_MODEL,
            google_api_key=config.GEMINI_API_KEY,
        )
        _vectorstore = Chroma(
            persist_directory=config.CHROMA_PERSIST_DIR,
            embedding_function=embeddings,
        )
    return _vectorstore


def build_rag_query(classification: str, severity_pct: float, cimmyt_grade: int) -> str:
    return (
        f"{classification} maize disease management "
        f"severity {severity_pct:.0f}% grade {cimmyt_grade} "
        f"Philippines field protocol control prevention"
    )


def retrieve_context(classification: str, severity_pct: float, cimmyt_grade: int):
    """Returns (rag_context: str, rag_sources: list[str])."""
    if classification == "HEALTHY":
        return "No disease detected — no management protocol required.", []

    query = build_rag_query(classification, severity_pct, cimmyt_grade)
    store = _get_vectorstore()
    results = store.similarity_search(query, k=config.RETRIEVAL_K)

    if not results:
        return "", []

    context_parts = []
    sources = []
    for doc in results:
        context_parts.append(doc.page_content)
        source = doc.metadata.get("source", "Unknown source")
        page = doc.metadata.get("page")
        label = f"{source} p.{page}" if page is not None else source
        if label not in sources:
            sources.append(label)

    return "\n\n---\n\n".join(context_parts), sources
