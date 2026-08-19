# MAIze — NLP Backend

Flask backend for the MAIze maize disease diagnosis app. Given an on-device
classification result (from the Android app's TFLite Student model), this
service generates server-side XAI overlay images and RAG-grounded diagnostic
guidance via Gemini, with a static offline fallback if Gemini is unreachable.

**Team:** Altes, Davis, Tayer, Ursua — AUF BSCS 3-A thesis.
**Adviser:** Ms. Melissa M. Pantig.

---

## Architecture

```
Android app (on-device TFLite)
        │  classification, confidence, severity_pct, original photo
        ▼
   POST /diagnose ──────────────────────────────────────────┐
        │                                                     │
        ├─► XAI overlay generation (pipeline/xai_engine.py)   │
        │   Full PyTorch checkpoint, server-side —            │
        │   TFLite can't run gradient-based XAI (Grad-CAM++)  │
        │                                                     │
        ├─► RAG retrieval (pipeline/rag_engine.py)            │
        │   ChromaDB knowledge base, top-K relevant passages  │
        │                                                     │
        └─► Guidance generation                                │
            ├─ Try: Gemini (3-image CoT prompt) ───────────────┤
            └─ Fail/offline: static bilingual guidance ────────┘
                (offline/static_guidance.py — always works,
                 no network/API dependency)
```

`/chat` is a separate, session-based endpoint for follow-up questions about
one specific diagnosis — grounded in that diagnosis's context, not a
general-purpose assistant.

---

## Prerequisites

- Python 3.10+ (developed/tested on a recent 3.x)
- A Google AI Studio account with billing enabled (free tier is too limited
  for real development — see **Cost Notes** below)
- An OpenAI account (only needed to run `evaluation/ragas_eval.py` — not
  required for the app itself)
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) — only
  needed if re-running knowledge base ingestion with scanned/image-based
  source PDFs
- (Deployment only) `gcloud` CLI, a GCP project with billing enabled

---

## Setup

### 1. Clone and create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux
pip install -r requirements.txt --break-system-packages
```

### 2. Set up your `.env` file

Copy `.env.example` to `.env` and fill in:

```
GEMINI_API_KEY=your_key_here
MAIZE_API_KEY=your_generated_key_here
```

Generate a `MAIZE_API_KEY` yourself (this is a shared app-level secret you
invent, not issued by anyone):

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

Optional — only needed to run `evaluation/ragas_eval.py`:

```
OPENAI_API_KEY=your_openai_key_here
```

(Not required for the app itself — RAGAS's Gemini-as-judge path was tried
and abandoned due to an unresolved `ragas`/`instructor` compatibility bug;
see **Evaluation** below.)

**Never commit `.env`** — it's gitignored, and should stay that way.

### 3. Place the trained model checkpoint

The Student model checkpoint (`.pth`) is **not** in this repo (too large for
git). Place it at:

```
pipeline/student_model/checkpoints/student_mobilenet_v3_small_mode_b_best.pth
```

Ask a teammate for the file, or regenerate it from the training pipeline.

### 4. Place the knowledge base source documents

PDFs/CSVs in `rag/knowledge_base/` are **not** in this repo — most are
copyrighted material (CIMMYT manuals, journal articles) that isn't ours to
redistribute. See the citations in this README's References section (or ask
a teammate) to source them independently, then place them in
`rag/knowledge_base/`.

### 5. Build the knowledge base

```bash
python -m rag.ingest
```

This is resumable — safe to re-run after adding new documents; it skips
chunks already embedded. If a source PDF is scanned/image-based (no text
layer), this step needs Tesseract OCR installed (see Prerequisites) or that
document will silently contribute 0 chunks.

### 6. Run the server

```bash
python app.py
```

**Use this, not `flask run`** — `flask run` doesn't pick up `threaded=True`
and `debug=True` from the `app.run(...)` call in `app.py`, which matters:
without threading, concurrent requests queue behind each other instead of
running in parallel, which can cause client-side timeouts on requests the
server was still legitimately processing.

The server listens on `0.0.0.0:5000` by default — reachable from an Android
emulator at `10.0.2.2:5000`, or from a physical device on the same network
via your machine's LAN IP (shown in the startup log).

**Windows Firewall:** the first time you run this, Windows may prompt to
allow the connection — allow it on both Private and Public networks, or
requests from the emulator/other devices will silently fail to connect at
all (no error, no log line — just nothing happens).

---

## API Endpoints

### `POST /diagnose`
**Header:** `X-API-Key: <MAIZE_API_KEY>`

```json
{
  "classification": "MSV",
  "confidence": 0.923,
  "severity_pct": 31.4,
  "original_image_b64": "<base64 original photo>",
  "language": "english",
  "include_tagalog": false,
  "offline": false
}
```

Does **not** accept `cimmyt_grade`, `segmentation_overlay_b64`, or
`xai_overlay_b64` — all computed/generated server-side. `include_tagalog`
defaults to `false` (see Cost Notes — this was a deliberate cost fix,
verified end-to-end through the full call chain from the Android client
down to the response schema).

Response includes `diagnosis`, `guidance`, `protocol`, `overlays` (when
`source: "gemini"`), `rag_sources`, `session_id`, and `tagalog` (only when
explicitly requested).

### `POST /chat`
**Header:** `X-API-Key: <MAIZE_API_KEY>`

```json
{
  "session_id": "<from a prior /diagnose response>",
  "message": "What should I do first?",
  "language": "english"
}
```

### `GET /health`
No auth required. Returns `{"model": "...", "status": "ok"}`.

---

## Testing

```bash
pytest tests/ -v
```

21 tests covering offline guidance completeness, severity/grade computation,
input validation, RAG query construction, and response schema validation
(including the English-only default path — no `tagalog` field present).
Does **not** cover live Gemini calls (those require real API credits) —
see `test_api_model.py` (quick model-connectivity check), and
`manual_diagnose_check.py` / `manual_chat_translate_check.py` (fuller
manual, credit-consuming integration tests exercising `/diagnose`,
`/chat`, and `/translate` against a locally running server). These last
two are deliberately named outside pytest's `test_*.py` discovery pattern
— they were originally named `test_*.py` and got accidentally collected
and executed by a bare `pytest`/`python -m pytest` run from the project
root, which is exactly why `pytest tests/ -v` (scoped to `tests/` only)
is the recommended command above.

---

## Evaluation (RAGAS)

`evaluation/ragas_eval.py` scores the RAG+generation pipeline on:
- **Faithfulness** — are `immediate_actions`/`management` recommendations
  actually grounded in retrieved knowledge base content?
- **Answer Relevancy** — does the response address a natural,
  farmer-phrased question about the diagnosis?

```bash
python -m evaluation.ragas_eval
```

Needs `OPENAI_API_KEY` in `.env` (used as the LLM judge — a genuine
Google/`ragas`/`instructor` compatibility bug made a same-billing
Gemini-as-judge approach unworkable; see the file's own inline comments
for the exact contradiction found) and a real Student checkpoint (see
Setup step 3). Generated test cases are cached locally
(`ragas_eval_generation_cache_n{N}_k{RETRIEVAL_K}.json`, gitignored) so
re-running to iterate on scoring logic doesn't re-pay for Gemini
generation — delete the relevant cache file to force fresh generations.
Set `RAGAS_EVAL_N_PER_CLASS=1` before running for a cheap ~10%-size
sanity check before committing to a full paid run.

**Final documented result** (`RETRIEVAL_K=5`, averaged across 2 runs due
to observed run-to-run variance from non-deterministic generation — see
`evaluation/ragas_eval_final_summary_k5.md` for full methodology, known
limitations, and per-run detail):

| Metric | Score |
|---|---|
| Faithfulness | 0.353 |
| Answer Relevancy | 0.490 |

`RETRIEVAL_K` was raised from 3 to 5 based on this evaluation — faithfulness
roughly doubled (more retrieved context to verify claims against) at a
small cost to relevancy (richer context → slightly less narrowly-targeted
answers). A follow-up prompt change asking for more concise action items
was tested to try to recover the relevancy cost, but was reverted after it
collapsed faithfulness instead — documented in the final summary as a
real, informative negative result, not silently discarded.

---

## Known Limitations

- **In-memory chat sessions** — not backed by a shared store (Redis, etc.).
  Cloud Run gives no guarantee that two requests for the same `session_id`
  land on the same container instance. `cloudbuild.yaml` pins
  `--max-instances=1` to work around this for thesis-scale demo use. This
  does **not** protect against session loss on container restart/redeploy.
  Not appropriate at real scale.
- **RAGAS's Gemini/`google.genai` provider path is broken** in the
  currently pinned `ragas==0.4.3` + `instructor` combination — confirmed
  via direct testing: `.ascore()` requires an async-capable client
  (`"Cannot use agenerate() with a synchronous client"`), while
  `instructor.from_genai()` (used internally by `llm_factory` for
  `provider="google"`) explicitly rejects anything that isn't a plain
  sync `google.genai.Client`. Neither client type satisfies both layers.
  `evaluation/ragas_eval.py` judges with OpenAI instead — see
  **Evaluation** above. Revisit if a future `ragas`/`instructor` release
  fixes this, since it would allow judging on the same Gemini billing
  account used for generation.
- **No hard spend cap exists for the Gemini API itself.** Google does not
  offer a true dollar-amount cutoff for a billed Gemini API key (unlike
  the free-tier AI Studio key, which just stops at zero cost). The real
  backstop in place is a **GCP Billing Budget** (alert-only, not
  auto-shutoff) set at $5/$9/$10 monthly thresholds with email alerts —
  see Deployment section below. A true hard cap would require wiring the
  budget to a Cloud Function that disables billing entirely on trip,
  which was deliberately **not** done, since it would take the whole
  Cloud Run service offline (not just throttle Gemini spend) — an
  unacceptable risk immediately before a live thesis defense.

---

## Cost Notes

Real-world spend during heavy development: roughly $10-11+ for one day of
active testing/debugging, using `gemini-3.5-flash` ($1.50/M input tokens,
**$9/M output tokens** — and Gemini 3.x's internal "thinking" tokens are
billed at the output rate too, which the Chain-of-Thought prompt design
triggers a lot of).

For local/dev work, set `GEMINI_DEV_MODE=1` in `.env` to route
`/diagnose` calls through `gemini-3.5-flash-lite` instead (~5x cheaper) —
see `config.py`'s `GEMINI_MODEL_LITE`/`GEMINI_DEV_MODE`. **Never set this
in production** — real farmer-facing diagnoses should stay on the full
model (`generate_guidance()` also accepts an explicit `use_lite_model`
override per call, independent of this env var, for callers like
evaluation scripts that need to choose deliberately).

On Cloud Run specifically, `GEMINI_DEV_MODE` is set/unset via:

```powershell
# Enable Lite model for cheap testing
gcloud run services update maize-nlp --region=asia-southeast1 --set-env-vars="GEMINI_DEV_MODE=1"

# Disable before any real/demo use — back to the full model
gcloud run services update maize-nlp --region=asia-southeast1 --remove-env-vars="GEMINI_DEV_MODE"
```

Each change deploys a new revision (no rebuild needed, ~30-60 seconds).
**Always confirm this is unset before a live defense/demo** — it's easy
to forget after a testing session.

**GCP Billing Budget set** at $5/$9/$10 monthly thresholds with email
alerts (Billing → Budgets & alerts). This is alert-only, not an automatic
spend cap — see Known Limitations above for why no harder cutoff is wired
up.

`include_tagalog` defaults to `false` specifically to reduce cost — a full
parallel Tagalog translation was previously generated on every single call
regardless of whether it was used, roughly doubling relevant output content
for no benefit when unused. This is verified end-to-end (Android client →
Flask → `gemini_engine.py` → `prompt_builder.py` → `response_validator.py`)
as of this README's last update — earlier versions of this fix looked
complete but had gaps at several points in that chain.

---

## Deployment (Google Cloud Run)

`Dockerfile` and `cloudbuild.yaml` are set up for Google Cloud Run (4Gi
memory, 2 CPU — needed for the PyTorch XAI checkpoint). Verify the
checkpoint and `chroma_db/` are present before building; the Dockerfile
checks for both.

**Live service:** `https://maize-nlp-437030900334.asia-southeast1.run.app`
Region: `asia-southeast1` (Singapore — closest to the Philippines, lowest
latency for the app's actual users).

Deploy with:
```bash
gcloud builds submit --config cloudbuild.yaml .
```

### First deployment — issues found and fixed

Getting the first successful deployment working surfaced several real
issues, documented here so they aren't re-discovered from scratch next
time (e.g. after a config change, a new teammate's machine, or a fresh
GCP project):

- **`.gcloudignore` was missing.** With none present, `gcloud builds
  submit` silently falls back to `.gitignore`'s exclusion rules for what
  gets uploaded to Cloud Build. Since `.gitignore` correctly excludes
  `rag/chroma_db/` and the `.pth` checkpoint from *git* (large,
  regeneratable/binary, shouldn't bloat the repo), the same exclusion
  wrongly applied to the *build upload* — causing the Dockerfile's own
  existence checks to fail with `"chroma_db is empty"` even though both
  were genuinely present locally. **Fixed** by adding an explicit
  `.gcloudignore` that excludes the same dev-only files but *keeps*
  `chroma_db/` and the `.pth` checkpoint, since the Docker build actually
  needs both baked into the image.

- **Missing OpenCV system libraries at container runtime.** The
  `python:3.10-slim` base image has no graphics/GUI libraries installed.
  `opencv-python` (pulled in transitively by `grad-cam` /
  `segmentation-models-pytorch`, even though only
  `opencv-python-headless` is directly listed in `requirements.txt`) still
  needs some of these shared libraries at import time. Two surfaced one at
  a time across successive deploys: `libxcb.so.1`, then
  `libgthread-2.0.so.0`. **Fixed** by adding this to the `Dockerfile`
  before `pip install`:
  ```dockerfile
  RUN apt-get update && apt-get install -y --no-install-recommends \
      libxcb1 \
      libsm6 \
      libxext6 \
      libgl1 \
      libglib2.0-0 \
      && rm -rf /var/lib/apt/lists/*
  ```

- **Public access (`allUsers`) IAM binding intermittently fails** as part
  of `cloudbuild.yaml`'s own deploy step (exact cause not fully
  root-caused — possibly timing-related on a fresh project). Deploy still
  succeeds, but the service is left non-public. **Workaround**: grant it
  manually after each deploy if the build log shows this warning:
  ```bash
  gcloud run services add-iam-policy-binding maize-nlp \
    --region=asia-southeast1 --member=allUsers --role=roles/run.invoker
  ```

- **Both secrets were corrupted in Secret Manager during initial setup.**
  An early `echo "value" | gcloud secrets create ...` mistake (pasting the
  wrong value, then not correctly overwriting it) left both
  `gemini-api-key` and `maize-api-key` storing 2-character garbage instead
  of real keys. This was **not** caught by the deploy succeeding, or even
  by the container booting successfully — it only surfaced once `/diagnose`
  and `/chat` were actually exercised, both silently falling back to
  offline/canned responses. Server logs showed the real cause:
  `API key not valid` (Gemini) and `401 unauthorized` (Maize key). **Fixed**
  by re-uploading correct values:
  ```powershell
  gcloud secrets versions add gemini-api-key --data-file="path\to\clean_key.txt"
  gcloud run services update maize-nlp --region=asia-southeast1 \
    --update-secrets=GEMINI_API_KEY=gemini-api-key:latest
  ```
  **Lesson for next time:** verify a secret's actual stored length right
  after creating it —
  ```powershell
  (gcloud secrets versions access latest --secret=SECRET_NAME).Length
  ```
  — rather than assuming the upload worked. Also avoid `echo | gcloud
  secrets create` on Windows PowerShell for anything sensitive; it can
  introduce trailing-newline or truncation issues. Writing to a temp file
  with `[System.IO.File]::WriteAllText(...)` first, then passing that file
  via `--data-file=`, is more reliable.

- **`UnicodeDecodeError` crash on `gcloud builds submit` itself**, before
  any Docker step even ran. Caused by em-dash characters (`—`) in code
  comments within `requirements.txt` being read with a mismatched encoding
  during gcloud's own source-upload step. **Fixed** by replacing em-dashes
  with plain hyphens in `requirements.txt`'s comments (the only file that
  actually caused the crash — `cloudbuild.yaml` and `Dockerfile` also had
  em-dashes in comments and were cleaned up defensively, but were not
  confirmed to be part of the actual crash).

### Cold starts

The first request after a period of no traffic is noticeably slower
(container boot: loading the PyTorch checkpoint, initializing ChromaDB,
etc., on top of normal Gemini/XAI compute time). **Before any live
demo/defense**, send a warm-up request a few minutes ahead of time rather
than trusting the first real request to be fast:
```powershell
Invoke-RestMethod -Uri "https://maize-nlp-437030900334.asia-southeast1.run.app/health"
```

To eliminate cold starts entirely (at extra cost — a container stays
running 24/7), set a minimum instance count temporarily:
```powershell
gcloud run services update maize-nlp --region=asia-southeast1 --min-instances=1
# ...and afterward, to stop paying for an always-on container:
gcloud run services update maize-nlp --region=asia-southeast1 --min-instances=0
```
Cloud Run otherwise scales to zero automatically when idle — no cost while
unused, no manual "turning the server on/off" needed day-to-day.

---

## Project Structure

```
maize_nlp/
├── app.py                  # Flask routes
├── config.py                # Model names, timeouts, token budgets, severity brackets
├── pipeline/
│   ├── gemini_engine.py     # Gemini API call (google.genai) + timeout handling
│   ├── offline_fallback.py  # Try Gemini → fall back to static guidance
│   ├── rag_engine.py        # ChromaDB retrieval
│   ├── prompt_builder.py    # System + Chain-of-Thought prompts
│   ├── xai_engine.py        # Server-side overlay generation (PyTorch)
│   ├── response_validator.py # Pydantic schema for Gemini's JSON output
│   ├── input_processor.py   # Image prep, severity→grade computation
│   └── student_model/       # Vendored model architecture + checkpoint
├── chatbot/
│   └── conversation_manager.py  # /chat session logic (google.genai)
├── rag/
│   ├── ingest.py             # Knowledge base builder
│   └── knowledge_base/       # Source PDFs/CSVs (not in repo — see Setup)
├── offline/
│   └── static_guidance.py    # 12 offline fallback entries (bilingual)
├── evaluation/
│   ├── ragas_eval.py         # RAGAS faithfulness/answer_relevancy scoring
│   └── ragas_eval_final_summary_k5.md  # Documented final results
├── Dockerfile                 # Cloud Run container build (incl. OpenCV system libs)
├── cloudbuild.yaml            # Cloud Build + Cloud Run deploy config
├── .gcloudignore               # Build-upload exclusions (keeps chroma_db/.pth, unlike .gitignore)
└── tests/                    # pytest suite (21 tests)
```
