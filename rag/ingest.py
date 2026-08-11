"""
Run this LOCALLY before building the Docker image:

    python -m rag.ingest

Chunks every PDF and CSV in rag/knowledge_base/, embeds with Gemini's
embedding model, and persists to rag/chroma_db/. Re-run whenever you add
or update a source file. Do NOT run this inside the Docker build step
(see Dockerfile comment) — the resulting chroma_db/ directory is committed
to the image as plain files, not built at container-build time with the
API key exposed.

Embedding is done in batches with pacing and retry-on-429. The free tier
has TWO separate quotas that matter here:
  - A per-minute burst limit (in practice, tighter than the documented
    100/min — expect occasional rate-limit pauses even with pacing).
  - A per-day cap of 1000 embed_content REQUESTS (not chunks/documents).
    Batch size matters a lot here: 10 chunks/request needs 10x more
    requests than 100 chunks/request for the same knowledge base, so a
    small BATCH_SIZE burns through the daily cap much faster for no
    benefit. If you hit the daily cap, retrying won't help — it only
    resets after ~24 hours. This script detects that case and stops
    immediately with a clear message rather than wasting time retrying.
    Re-running later resumes from where it left off (see embed_and_persist).
"""
import csv
import re
import sys
import time

import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

import config

# Free tier: 100 embed_content requests/minute (documented — actual
# observed limit is often tighter in practice) AND 1000 requests/DAY.
# Batch size directly affects daily quota usage — the free tier's daily
# cap is per REQUEST, not per document, so fewer/bigger batches use less
# of that budget. 50 keeps well under the per-request payload limits
# while cutting request count ~5x versus a batch size of 10.
BATCH_SIZE = 50
BATCH_DELAY_SECONDS = 5.0
MAX_RETRIES = 4
DEFAULT_RETRY_WAIT_SECONDS = 30


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


def _extract_retry_wait_seconds(exc: Exception) -> float:
    """Parses "Please retry in 11.2s" (or similar) out of the error message
    if present, so we wait exactly as long as the server asked, plus a
    small buffer — rather than guessing a fixed backoff."""
    match = re.search(r"retry in (\d+(?:\.\d+)?)s", str(exc))
    if match:
        return float(match.group(1)) + 5.0
    return DEFAULT_RETRY_WAIT_SECONDS


def _add_batch_with_retry(vectorstore: Chroma, batch: list[Document], batch_num: int, total_batches: int):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            vectorstore.add_documents(batch)
            print(f"  batch {batch_num}/{total_batches}: {len(batch)} chunks embedded")
            return
        except Exception as exc:
            # LangChain wraps the underlying google.genai.errors.ClientError
            # in its own GoogleGenerativeAIError, so we can't catch the
            # specific exception type — match on the message instead.
            exc_str = str(exc)
            is_rate_limit = "RESOURCE_EXHAUSTED" in exc_str or "429" in exc_str
            if not is_rate_limit:
                raise

            # A per-DAY quota can't be fixed by waiting a few seconds and
            # retrying — it only resets after ~24 hours. Fail fast instead
            # of burning 4 retries x ~1 minute each for nothing.
            if "PerDay" in exc_str or "RequestsPerDay" in exc_str:
                raise RuntimeError(
                    f"Batch {batch_num}/{total_batches}: hit the FREE TIER DAILY "
                    f"embedding quota (not a per-minute limit — retrying won't help "
                    f"today). {len(already_embedded_count_hint(vectorstore))} chunks "
                    f"are already saved in chroma_db/ and won't need to be "
                    f"re-embedded. Wait ~24 hours for the quota to reset, then "
                    f"re-run `python -m rag.ingest` — it will resume automatically "
                    f"from where it stopped."
                ) from exc

            if attempt == MAX_RETRIES:
                raise RuntimeError(
                    f"Batch {batch_num} still rate-limited after {MAX_RETRIES} retries. "
                    f"Consider lowering BATCH_SIZE further or increasing BATCH_DELAY_SECONDS "
                    f"in rag/ingest.py."
                ) from exc
            wait_seconds = _extract_retry_wait_seconds(exc)
            print(
                f"  batch {batch_num}/{total_batches}: rate limited "
                f"(attempt {attempt}/{MAX_RETRIES}), waiting {wait_seconds:.0f}s..."
            )
            time.sleep(wait_seconds)


def already_embedded_count_hint(vectorstore: Chroma) -> list:
    """Best-effort count for the error message above — not critical if it fails."""
    try:
        existing = vectorstore.get(include=[])
        return existing.get("ids", [])
    except Exception:
        return []


def embed_and_persist(chunks: list[Document], embeddings: GoogleGenerativeAIEmbeddings):
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=str(config.CHROMA_PERSIST_DIR),
    )

    # Resumable: if chroma_db/ already has entries (e.g. this script was
    # interrupted last run), skip chunks already embedded rather than
    # re-embedding them and burning quota again. Matches on page_content,
    # which is stable across runs since the same PDFs/CSVs produce the
    # same chunks.
    existing = vectorstore.get(include=["documents"])
    already_embedded = set(existing["documents"]) if existing and existing.get("documents") else set()
    if already_embedded:
        print(f"Found {len(already_embedded)} chunks already in chroma_db/ — skipping those.")
        chunks = [c for c in chunks if c.page_content not in already_embedded]
        print(f"{len(chunks)} chunks remaining to embed.")

    if not chunks:
        print("Nothing left to embed.")
        return vectorstore

    total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Embedding {len(chunks)} chunks in {total_batches} batches of {BATCH_SIZE}...")

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        _add_batch_with_retry(vectorstore, batch, batch_num, total_batches)
        if batch_num < total_batches:
            time.sleep(BATCH_DELAY_SECONDS)

    return vectorstore


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

    embed_and_persist(all_chunks, embeddings)
    print(f"Ingested {len(all_chunks)} chunks into {config.CHROMA_PERSIST_DIR}")


if __name__ == "__main__":
    main()
