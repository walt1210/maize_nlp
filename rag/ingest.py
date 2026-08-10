"""
Run this LOCALLY before building the Docker image:

    python -m rag.ingest

Chunks every PDF and CSV in rag/knowledge_base/, embeds with Google
text-embedding-004, and persists to rag/chroma_db/. Re-run whenever you
add or update a source file. Do NOT run this inside the Docker build step
(see Dockerfile comment) — the resulting chroma_db/ directory is committed
to the image as plain files, not built at container-build time with the
API key exposed.
"""
import csv
import sys

import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

import config


def load_pdf_as_documents(pdf_path) -> list[Document]:
    docs = []
    with fitz.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf, start=1):
            text = page.get_text().strip()
            if text:
                docs.append(
                    Document(
                        page_content=text,
                        metadata={"source": pdf_path.stem, "page": page_num},
                    )
                )
    return docs


def load_csv_as_documents(csv_path) -> list[Document]:
    """
    Each row becomes its own Document — a table row is a natural retrieval
    unit and shouldn't be split mid-row by the text chunker. Columns are
    rendered as "column: value" pairs so retrieval can match on any field
    (e.g. severity_pct, product_name, dosage), not just the first column.
    """
    docs = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_num, row in enumerate(reader, start=1):
            text = "\n".join(f"{col}: {val}" for col, val in row.items() if val)
            if text.strip():
                docs.append(
                    Document(
                        page_content=text,
                        metadata={"source": csv_path.stem, "row": row_num},
                    )
                )
    return docs


def chunk_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    return splitter.split_documents(docs)


def main():
    if not config.GEMINI_API_KEY:
        print("ERROR: GEMINI_API_KEY not set. Export it before running ingest.", file=sys.stderr)
        sys.exit(1)

    pdf_paths = sorted(config.KNOWLEDGE_BASE_DIR.glob("*.pdf"))
    csv_paths = sorted(config.KNOWLEDGE_BASE_DIR.glob("*.csv"))
    if not pdf_paths and not csv_paths:
        print(f"ERROR: no PDFs or CSVs found in {config.KNOWLEDGE_BASE_DIR}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(pdf_paths)} PDF(s): {[p.name for p in pdf_paths]}")
    print(f"Found {len(csv_paths)} CSV(s): {[p.name for p in csv_paths]}")

    pdf_docs = []
    for pdf_path in pdf_paths:
        page_docs = load_pdf_as_documents(pdf_path)
        print(f"  {pdf_path.name}: {len(page_docs)} pages with text")
        pdf_docs.extend(page_docs)

    csv_docs = []
    for csv_path in csv_paths:
        row_docs = load_csv_as_documents(csv_path)
        print(f"  {csv_path.name}: {len(row_docs)} rows")
        csv_docs.extend(row_docs)

    # Only PDFs go through the text chunker — CSV rows are already
    # appropriately sized retrieval units and shouldn't be split mid-row.
    pdf_chunks = chunk_documents(pdf_docs)
    all_chunks = pdf_chunks + csv_docs
    print(f"Total chunks: {len(pdf_chunks)} from PDFs + {len(csv_docs)} from CSVs = {len(all_chunks)}")

    embeddings = GoogleGenerativeAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        google_api_key=config.GEMINI_API_KEY,
    )

    Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        persist_directory=str(config.CHROMA_PERSIST_DIR),
    )
    print(f"Ingested {len(all_chunks)} chunks into {config.CHROMA_PERSIST_DIR}")


if __name__ == "__main__":
    main()
