# MAIze NLP Pipeline

Flask backend that turns MAIze Student model outputs (on-device, TFLite)
into RAG-grounded, bilingual (EN/Tagalog) agricultural guidance via Gemini
2.5 Flash. See `MAIZE_NLP_BLUEPRINT.md` for full architecture background.

## Decisions baked into this codebase

- **Student model runs on-device (Android/TFLite).** Flask never touches
  the vision model — it only receives `classification`, `confidence`,
  `severity_pct`, `cimmyt_grade`, and the two base64 images.
- **No `xai_description` text field.** Gemini reasons directly from the
  two attached images (original + Grad-CAM++ overlay). No masks/heatmap
  arrays are sent to the backend.
- **Low-confidence caveat at `confidence < 0.6`** (`config.LOW_CONFIDENCE_THRESHOLD`).
  Below this, the prompt instructs Gemini to open with an
  extension-officer-consultation caveat; the API response also carries
  a `low_confidence: bool` flag.
- **RAGAS is offline-only** (`evaluation/ragas_eval.py`), never computed
  per live request — it makes its own LLM calls internally, so scoring
  every farmer's scan would double latency/cost for no runtime benefit.
- **`/diagnose`, `/chat`, `/translate` all require an `X-API-Key` header**
  matching `MAIZE_API_KEY`, since Cloud Run is deployed with
  `--allow-unauthenticated`.
- **Chatbot sessions are in-memory**, and Cloud Run is deployed with
  `--max-instances=1` to work around the lack of cross-instance session
  affinity. This is a thesis-scope workaround, not a production pattern —
  see the comment at the top of `chatbot/conversation_manager.py`.

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment variables
cp .env.example .env
# Edit .env: set GEMINI_API_KEY, OPENAI_API_KEY (dev only), MAIZE_API_KEY

# 4. Add PDF knowledge base documents
# Place PDFs in rag/knowledge_base/ (see blueprint Section 9 for the list)

# 5. Ingest the knowledge base (run once, or after adding/updating PDFs)
python -m rag.ingest
# This must be run LOCALLY, before building the Docker image — see the
# comment at the top of Dockerfile for why.

# 6. Run locally
export $(cat .env | xargs)        # or use python-dotenv / your IDE's env loader
flask run --port 5000

# 7. Run tests
pytest tests/ -v

# 8. Smoke-test the /diagnose endpoint with mock data
python -c "
import requests
from sample_data.mock_student_output import get_mock_request_body
body = get_mock_request_body('msv_moderate')
r = requests.post('http://localhost:5000/diagnose', json=body,
                   headers={'X-API-Key': 'your_maize_api_key_here'})
print(r.status_code, r.json())
"
```

## Deploy to Cloud Run

```bash
# Make sure rag/chroma_db/ exists (step 5 above) before building.
docker build -t maize-nlp .

# Push and deploy (or use cloudbuild.yaml with `gcloud builds submit`)
gcloud builds submit --config=cloudbuild.yaml

# Store secrets first, if you haven't:
echo -n "your_gemini_key" | gcloud secrets create gemini-api-key --data-file=-
echo -n "your_maize_key"  | gcloud secrets create maize-api-key  --data-file=-
```

## Offline evaluation (for the thesis results chapter)

```bash
# RAGAS faithfulness + answer relevancy over 30 synthetic cases
python -m evaluation.ragas_eval

# chrF score over an expert-validated Tagalog reference set
python -m evaluation.chrf_eval path/to/tagalog_eval_set.csv
```

## Project structure

```
maize_nlp/
├── app.py                        Flask entry point, all endpoints
├── config.py                     All settings, thresholds, model names
├── pipeline/
│   ├── input_processor.py        Image decode/validate, staging, confidence
│   ├── rag_engine.py             ChromaDB retrieval
│   ├── prompt_builder.py         System prompt + CoT prompt construction
│   ├── gemini_engine.py          Gemini API call, timeout, validation
│   ├── response_validator.py     Pydantic schema for Gemini's JSON output
│   ├── tagalog_handler.py        On-demand translation
│   └── offline_fallback.py       Gemini→offline orchestration
├── rag/
│   ├── ingest.py                 Run locally to build chroma_db/
│   └── knowledge_base/           Put PDFs here
├── offline/
│   └── static_guidance.py        12 pre-generated fallback entries
├── chatbot/
│   └── conversation_manager.py   Session store, out-of-scope detection
├── evaluation/
│   ├── ragas_eval.py             Offline RAGAS batch scoring
│   ├── chrf_eval.py              Offline chrF scoring
│   └── expert_audit_template.csv
├── sample_data/
│   └── mock_student_output.py    Test fixtures, no trained model needed
└── tests/
```

## Still needed before full implementation

See `MAIZE_NLP_BLUEPRINT.md` Section 20 for the original list (API keys,
knowledge base PDFs, expert Tagalog validator, trained Student model,
Android integration). One addition: the **offline guidance content in
`offline/static_guidance.py` is a reasonable draft, not expert-reviewed**
— have a licensed agriculturist check it against `evaluation/expert_audit_template.csv`
criteria before treating it as safe to ship, since it's what farmers see
whenever Gemini is unreachable.
