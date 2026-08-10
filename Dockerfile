FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# The knowledge base (rag/chroma_db/) must already exist before you run
# `docker build` — run `python -m rag.ingest` LOCALLY first, then build.
# Cloud Run's filesystem is ephemeral, so ChromaDB can't be built at
# runtime — but it also must NOT be built inside the Docker build step
# with the API key passed as an ARG/ENV, since that bakes the key into
# the image's layer history where anyone with the image can extract it.
# `COPY . .` above already includes rag/chroma_db/ as a plain directory
# of local files — no API key needed at build time.
RUN test -d rag/chroma_db && test -n "$(ls -A rag/chroma_db)" || \
    (echo "ERROR: rag/chroma_db is empty. Run 'python -m rag.ingest' locally before building." && exit 1)

ENV PORT=8080
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", \
     "--timeout", "120", "app:app"]
