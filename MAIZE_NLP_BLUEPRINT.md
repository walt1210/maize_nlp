# MAIze NLP Pipeline — Complete Technical Blueprint
## Context document for continuing work in a new chat

---

## 1. Project Identity

**System name:** MAIze Pathological Guidance Module — NLP Pipeline
**Parent thesis:** MAIze: A U-Net with MobileNetV2 and Explainable AI Framework for Maize Streak Virus Identification and Symptom Segmentation in Zea mays L.
**Institution:** Angeles University Foundation, College of Computer Studies, BSCS 3-A
**Team:** Altes, Zylah Klein · Davis, Dominic · Tayer, Catherine P. · Ursua, Walter Vince
**NLP subject:** CSE03 — Natural Language Processing (Professional Elective)
**Adviser:** Ms. Melissa M. Pantig, MCS

**Purpose:** Bridge the gap between the Student model's black-box outputs (classification, severity, masks, Grad-CAM++) and human-interpretable, actionable agricultural guidance in both English and Tagalog.

---

## 2. What the Vision Model Produces (NLP Pipeline Inputs)

The Student model (MobileNetV2-based multi-task UNet) produces these outputs per inference run. These are the exact inputs the NLP pipeline receives:

```python
{
  "classification": "HEALTHY" | "MSV" | "MLN",
  "confidence": float,          # softmax probability of predicted class (0.0–1.0)
  "severity_pct": float,        # continuous severity percentage (0.0–100.0)
  "cimmyt_grade": int,          # discrete grade (see grading section below)
  "monitoring_stage": int,      # 0–3 (derived from severity_pct, see below)
  "sil_mask": np.ndarray,       # [224, 224] binary leaf silhouette mask
  "sym_mask": np.ndarray,       # [224, 224] binary symptom mask
  "gradcam_heatmap": np.ndarray,# [224, 224] float32 Grad-CAM++ heatmap
  "gradcam_overlay": PIL.Image, # Grad-CAM++ heatmap overlaid on original image
  "original_image": PIL.Image,  # EXIF-corrected original leaf image
}
```

### CIMMYT Severity Grading Scales

**MSV — Soto et al. (1982), validated by Sime et al. (2021, Agriculture 11(2):130)**
0-5 leaf-area-based scale operationalized from pixel coverage %:
```
Grade 0 : HEALTHY (no symptoms) — handled separately
Grade 1 : ≤10%   chlorotic leaf area (trace streaks)
Grade 2 : 11-25% chlorotic leaf area (light streaking on older leaves)
Grade 3 : 26-50% chlorotic leaf area (moderate streaking, slight stunting)
Grade 4 : 51-75% chlorotic leaf area (severe streaking)
Grade 5 : ≥75%   chlorotic leaf area (highly susceptible, severely stunted)
```

**MLN — Beyene et al. (2017, Euphytica 213:224); Gowda et al. (2015)**
1-5 symptom-progression scale operationalized as pixel coverage %:
```
Grade 1 : <10%   no/trace symptoms
Grade 2 : 10-25% fine chlorotic streaks/mottling on lower leaves
Grade 3 : 25-50% chlorotic mottling and mosaic throughout plant
Grade 4 : 50-75% excessive mottling, necrosis, dead heart
Grade 5 : >75%   dead plant, complete plant necrosis
```

**HEALTHY:** grade = 0 always.

### Monitoring Stage Mapping (derived in NLP pipeline)
```python
def severity_to_stage(severity_pct, classification):
    if classification == "HEALTHY" or severity_pct < 1.0:
        return 0   # No infection — routine monitoring
    elif severity_pct < 25.0:
        return 1   # Mild — monitor weekly, scout for vectors
    elif severity_pct < 60.0:
        return 2   # Moderate — apply management protocol immediately
    else:
        return 3   # Severe — immediate intervention, consider rouging
```

### Confidence Threshold (derived in NLP pipeline)
```python
LOW_CONFIDENCE_THRESHOLD = 0.6

def needs_confidence_caveat(confidence: float) -> bool:
    # Below this, prepend a caveat to the guidance recommending the
    # farmer confirm with a DA extension officer before acting, since
    # the underlying classification itself is uncertain.
    return confidence < LOW_CONFIDENCE_THRESHOLD
```

> **Note:** No programmatic XAI text description is generated server-side.
> Per Design Decision #2 (Section 17), the pipeline relies entirely on
> passing both images (original + Grad-CAM++ overlay) directly to Gemini —
> Gemini describes what it sees rather than consuming a proxy text string.
> `sym_mask` and `gradcam_heatmap` arrays stay on-device and are never
> sent to the Flask backend.

---

## 3. Pipeline Architecture (5 Layers)

```
┌─────────────────────────────────────────────────────────────┐
│  LAYER 1 — INPUT                                             │
│  Student model runs on-device (Android, TFLite) — Flask      │
│  never touches the vision model. Receives only:               │
│  classification · confidence · severity_pct · cimmyt_grade   │
│  · monitoring_stage · original_image_b64 · gradcam_overlay_b64│
└─────────────────────────┬───────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  LAYER 2 — GROUNDING (RAG)                                  │
│  ChromaDB vector store + Google text-embedding-004          │
│  Query: "{classification} maize {severity}% grade {grade}"  │
│  Returns top-3 chunks from knowledge base                   │
│  Prevents hallucination by anchoring Gemini to verified     │
│  agricultural knowledge                                     │
└─────────────────────────┬───────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  LAYER 3 — ENGINE                                           │
│  Gemini 2.5 Flash API (primary)                             │
│  GPT-4o mini (experimental comparison)                      │
│  Multimodal: receives gradcam_overlay image directly        │
│  Chain-of-Thought reasoning prompt                          │
│  Outputs: justification · protocol · key_fact · tagalog     │
└─────────────────────────┬───────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  LAYER 4 — VALIDATION & SAFETY                              │
│  Runtime: Pydantic schema validation on Gemini's JSON       │
│           output — malformed/incomplete response falls      │
│           back to offline_fallback (same path as timeout)   │
│  Offline eval: RAGAS metrics (Faithfulness + Answer Rel.)   │
│  Expert audit: licensed agriculturists, 5-point Likert      │
│  Tagalog: chrF score + separate expert Likert               │
└─────────────────────────┬───────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  LAYER 5 — OUTPUT                                           │
│  JSON response → Android app                                │
│  Chatbot: follow-up Q&A with conversation memory            │
│  Offline fallback: pre-generated static guidance            │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Guidance Scope

The NLP pipeline covers the following topics per diagnosis. These are NOT generic chatbot topics — all responses are grounded in the RAG knowledge base and anchored to the specific diagnosis:

```
✓ What the disease is (diagnostic justification — the "why")
✓ What the visual evidence shows (XAI-grounded explanation)
✓ How to manage the current infection
✓ Actions needed immediately at each severity stage
✓ How the disease spreads (vector, wind, contact)
✓ Recommended distances for rouging / isolation
✓ How to prevent further spread
✓ Precautions for the farmer and field workers
✓ Prevention strategies (resistant varieties, planting schedule)
✓ Detection methods for monitoring progression
✓ Control measures (chemical, biological, cultural)
✓ Philippine-specific context where available
✓ Tagalog translation of all above (on-demand + automatic)
```

Topics explicitly OUT of scope (chatbot will decline gracefully):
```
✗ Non-maize crops
✗ Human health questions
✗ Market prices
✗ General agriculture beyond maize streak diseases
```

---

## 5. Technology Stack

| Component | Technology | Notes |
|---|---|---|
| Primary LLM | Gemini 2.5 Flash | `gemini-2.5-flash` model string |
| Comparison LLM | GPT-4o mini | `gpt-4o-mini` for experimental comparison |
| Embeddings | Google text-embedding-004 | Same ecosystem as Gemini |
| Vector store | ChromaDB | Local, no server, persistent on disk |
| RAG framework | LangChain | Chunking, retrieval, prompt management |
| RAG evaluation | RAGAS | Faithfulness + Answer Relevance metrics |
| Backend framework | Flask | REST API, Python |
| Deployment | Google Cloud Run | Always Free tier, containerized |
| Containerization | Docker | `Dockerfile` included |
| Environment | Python 3.10+ in Docker | `.env` for secrets |
| Tagalog metric | chrF score | `sacrebleu` library |
| PDF processing | PyMuPDF (fitz) | Chunking knowledge base PDFs |

---

## 6. Project File Structure

```
maize_nlp/
│
├── .env                          ← API keys (never commit to git)
│   GEMINI_API_KEY=...
│   OPENAI_API_KEY=...            ← for GPT-4o mini comparison
│
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── cloudbuild.yaml               ← Google Cloud Run deployment config
│
├── app.py                        ← Flask application entry point
│
├── config.py                     ← All settings (model names, paths, thresholds)
│
├── pipeline/
│   ├── __init__.py
│   ├── input_processor.py        ← Receives Student outputs, derives XAI description
│   │                               and monitoring stage
│   ├── rag_engine.py             ← ChromaDB retrieval + query builder
│   ├── prompt_builder.py         ← Structured CoT prompt construction
│   ├── gemini_engine.py          ← Gemini 2.5 Flash API calls + JSON parsing
│   ├── gpt_engine.py             ← GPT-4o mini API calls (comparison only)
│   ├── response_validator.py     ← RAGAS metrics + safety checks
│   ├── tagalog_handler.py        ← Tagalog translation + chrF scoring
│   └── offline_fallback.py       ← Static pre-generated guidance lookup
│
├── rag/
│   ├── __init__.py
│   ├── ingest.py                 ← PDF chunking + embedding + ChromaDB ingestion
│   ├── knowledge_base/           ← PUT PDF FILES HERE
│   │   ├── cruz_et_al_2024.pdf
│   │   ├── cimmyt_maize_disease_field_guide.pdf
│   │   ├── fao_mln_management_guide.pdf
│   │   ├── da_philippines_corn_bulletin.pdf
│   │   ├── beyene_et_al_2017.pdf
│   │   ├── sime_et_al_2021.pdf
│   │   └── eppo_msv_datasheet.pdf   ← optional
│   └── chroma_db/                ← AUTO-GENERATED by ingest.py (persistent)
│
├── offline/
│   └── static_guidance.py        ← Pre-generated static guidance
│                                   12 combinations: 3 classes × 4 stages
│
├── chatbot/
│   ├── __init__.py
│   └── conversation_manager.py   ← Conversation history + context injection
│
├── evaluation/
│   ├── __init__.py
│   ├── ragas_eval.py             ← Automated RAG metrics
│   ├── chrf_eval.py              ← Tagalog translation quality
│   └── expert_audit_template.csv ← 5-point Likert scale for expert review
│
├── sample_data/
│   └── mock_student_output.py    ← Realistic mock Student outputs for testing
│                                   without the actual trained model
│
└── tests/
    ├── test_pipeline.py
    ├── test_rag.py
    └── test_offline.py
```

---

## 7. API Endpoints

### POST `/diagnose`
Primary endpoint. Receives Student model outputs, returns full guidance.

**Request body:**
```json
{
  "classification": "MSV",
  "confidence": 0.923,
  "severity_pct": 31.4,
  "cimmyt_grade": 3,
  "original_image_b64": "<base64 encoded original leaf PNG>",
  "gradcam_overlay_b64": "<base64 encoded Grad-CAM++ overlay PNG>",
  "language": "english",
  "include_tagalog": true
}
```

> Both images are required. `original_image_b64` lets Gemini see the actual
> leaf symptoms directly. `gradcam_overlay_b64` lets Gemini interpret where
> the model's attention was concentrated. Together they enable genuine
> visually-grounded reasoning — Gemini is looking at the same evidence
> that drove the Student's classification, not just reading a text description.

> `low_confidence` in the response is `true` when `confidence < 0.6`
> (`LOW_CONFIDENCE_THRESHOLD` in `config.py`), computed by
> `input_processor.py` before the prompt is built. When true, the
> `justification` field opens with the extension-officer caveat (see
> Section 8, System Prompt rule 6).

**Response:**
```json
{
  "status": "success",
  "source": "gemini" | "offline",
  "classification": "MSV",
  "confidence": 0.923,
  "low_confidence": false,
  "severity_pct": 31.4,
  "cimmyt_grade": 3,
  "monitoring_stage": 2,
  "diagnosis": {
    "justification": "...",
    "xai_explanation": "...",
    "key_fact": "..."
  },
  "guidance": {
    "immediate_actions": ["...", "..."],
    "management": ["...", "..."],
    "spread": "...",
    "distances": "...",
    "prevention": ["...", "..."],
    "precautions": ["...", "..."],
    "detection": "...",
    "control": {
      "chemical": "...",
      "biological": "...",
      "cultural": "..."
    }
  },
  "protocol": {
    "title": "MSV Rapid Response Protocol",
    "steps": ["...", "...", "...", "..."]
  },
  "tagalog": {
    "justification": "...",
    "protocol_steps": ["...", "...", "...", "..."],
    "key_fact": "..."
  },
  "rag_sources": ["Cruz et al. 2024", "CIMMYT Field Guide p.34"]
}
```

> **Note:** `ragas_scores` intentionally does NOT appear in the live response.
> RAGAS faithfulness/answer-relevancy metrics work by making additional LLM
> calls to judge the output — computing them per-request would roughly
> double LLM API calls and latency on every farmer's scan for no runtime
> benefit. RAGAS stays an **offline evaluation step** (Section 13) run
> against the 30-case synthetic test set for the thesis results chapter,
> via `evaluation/ragas_eval.py` — never inside `/diagnose`.

---

### POST `/chat`
Chatbot follow-up Q&A. Maintains conversation context from the `/diagnose` call.

**Request body:**
```json
{
  "session_id": "abc123",
  "message": "How far should I distance the infected plants?",
  "language": "english" | "tagalog"
}
```

**Response:**
```json
{
  "status": "success",
  "source": "gemini" | "offline",
  "reply": "...",
  "reply_tagalog": "...",
  "session_id": "abc123"
}
```

---

### POST `/translate`
On-demand Tagalog translation of any existing response text.

**Request body:**
```json
{
  "text": "...",
  "context": "MSV Grade 3 guidance"
}
```

**Response:**
```json
{
  "status": "success",
  "tagalog": "...",
  "chrf_score": 0.74
}
```

---

### GET `/health`
Health check for Cloud Run.

**Response:** `{"status": "ok", "model": "gemini-2.5-flash"}`

---

### GET `/offline-check`
Returns whether offline fallback is available for a given diagnosis.

---

## 8. Gemini 2.5 Flash Prompt Design

### System Prompt
```
You are an expert plant pathologist and agricultural extension officer
advising Filipino maize farmers. You have deep knowledge of Maize Streak
Virus (MSV) and Maize Lethal Necrosis (MLN) as they affect Zea mays L.
in the Philippine context.

Your role is to convert computer vision diagnostic outputs into clear,
actionable, culturally appropriate guidance that a smallholder farmer
or agriculture student can immediately act on.

RULES:
1. Base all advice ONLY on the retrieved agricultural knowledge provided.
   Never generate management protocols from memory alone.
2. Always reference specific morphological markers visible in the XAI
   explanation when writing the justification.
3. Severity Stage 3 (>60%) requires immediate rouging instructions.
4. All chemical control recommendations must use only DA-Philippines
   registered pesticides when Philippine-specific sources are available.
5. Tagalog translations must preserve clinical safety — never simplify
   a safety instruction to the point of ambiguity.
6. If a low-confidence notice is present in the diagnostic inputs,
   open the justification by noting the classification is uncertain
   and recommend the farmer confirm with a DA extension officer before
   acting on chemical control recommendations specifically.
7. Respond ONLY in valid JSON. No preamble, no markdown backticks.
```

### Confidence Caveat Injection (prompt_builder.py)
```python
LOW_CONFIDENCE_THRESHOLD = 0.6

def build_low_confidence_notice(confidence: float) -> str:
    if confidence < LOW_CONFIDENCE_THRESHOLD:
        return (
            f"- ⚠ LOW CONFIDENCE ({confidence:.1%}): classification is "
            f"uncertain. Guidance must open with a caveat recommending "
            f"the farmer confirm with a DA extension officer before "
            f"acting, especially on chemical control."
        )
    return ""  # empty string — omitted from the rendered prompt
```

### Chain-of-Thought User Prompt
```
DIAGNOSTIC INPUTS (from MAIze computer vision model):
- Disease classification : {classification}  (confidence: {confidence:.1%})
- Infected tissue area   : {severity_pct:.1f}%
- Severity grade         : {grade_label}  (Grade {cimmyt_grade})
- Monitoring stage       : Stage {monitoring_stage} / 3
{low_confidence_notice}

[ATTACHED IMAGE 1: Original leaf photograph]
[ATTACHED IMAGE 2: Grad-CAM++ diagnostic attention overlay —
                    amber/red = high model attention, blue = low attention]

RETRIEVED EXPERT KNOWLEDGE (from verified agricultural sources):
{rag_context}

CHAIN-OF-THOUGHT REASONING — think step by step before responding:
Step 1: Looking at both attached images, what specific visual markers
         are present on the leaf, and which regions does the Grad-CAM++
         overlay highlight? What pattern do they form?
Step 2: Why does this pattern indicate {classification} at this severity?
         What distinguishes it from other conditions (nutrient deficiency,
         other diseases)?
Step 3: Given Grade {cimmyt_grade} and Stage {monitoring_stage},
         what is the urgency level? What happens if no action is taken?
Step 4: What are the most critical immediate actions for Stage {monitoring_stage}?
Step 5: What spread prevention measures are most important in the
         Philippine field context?

Now generate your response as a JSON object with this exact structure:
{
  "justification": "2-3 sentences explaining WHY this classification
                    was made, referencing the specific visual evidence
                    you see in the attached images.",
  "xai_explanation": "1-2 sentences describing what the amber heatmap
                      shows and what it means diagnostically.",
  "key_fact": "One critical fact the farmer must know.",
  "immediate_actions": ["action1", "action2", "action3"],
  "management": ["step1", "step2", "step3", "step4"],
  "spread": "How this disease spreads (vector, mechanism, distance).",
  "distances": "Specific recommended distances for rouging/isolation.",
  "prevention": ["measure1", "measure2", "measure3"],
  "precautions": ["precaution1", "precaution2"],
  "detection": "How to monitor for progression or new infections.",
  "control": {
    "chemical": "Specific chemical control with DA-registered products.",
    "biological": "Biological control options if available.",
    "cultural": "Cultural practices (crop rotation, planting schedule, etc.)"
  },
  "protocol_title": "MSV/MLN Rapid Response Protocol",
  "protocol_steps": ["step1", "step2", "step3", "step4"],
  "tagalog": {
    "justification": "...",
    "key_fact": "...",
    "protocol_steps": ["...", "...", "...", "..."],
    "immediate_actions": ["...", "...", "..."]
  }
}
```

### Gemini API Call Configuration
```python
model = genai.GenerativeModel("gemini-2.5-flash")
response = model.generate_content(
    [
        prompt_text,
        original_image,      # Image 1: actual leaf — Gemini sees real symptoms
                             # (chlorotic streaks, necrotic patches, colour changes)
        gradcam_image,       # Image 2: Grad-CAM++ overlay — Gemini sees where
                             # the Student model's attention was concentrated
                             # (amber heatmap over the leaf)
    ],
    generation_config=genai.GenerationConfig(
        temperature=0.2,            # low = factual, consistent
        max_output_tokens=2000,
        response_mime_type="application/json",
    )
)
# Both images are passed so Gemini can:
#   1. Describe actual visual symptoms from original_image
#      (what the disease looks like on THIS specific leaf)
#   2. Explain the XAI attention from gradcam_image
#      (what regions drove the classification decision)
# This is what "visually-grounded reasoning" means in the NLP objectives —
# the LLM is literally looking at the same visual evidence as the model.
```

### Image Preparation (pipeline/input_processor.py)
```python
import base64
from PIL import Image
import io
import google.generativeai as genai

def prepare_images(original_image_b64: str,
                   gradcam_overlay_b64: str):
    """
    Decode base64 images from Android app and convert to
    Gemini-compatible PIL Image objects.
    """
    def b64_to_pil(b64_str: str) -> Image.Image:
        img_bytes = base64.b64decode(b64_str)
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")

    original_pil  = b64_to_pil(original_image_b64)
    gradcam_pil   = b64_to_pil(gradcam_overlay_b64)

    # Resize to 512×512 max — Gemini vision works well at this size
    # and keeps token count manageable
    max_size = 512
    original_pil.thumbnail((max_size, max_size), Image.LANCZOS)
    gradcam_pil.thumbnail((max_size, max_size), Image.LANCZOS)

    return original_pil, gradcam_pil
```

### Prompt Image Labels
Add these lines to the prompt text so Gemini knows which image is which:
```
[IMAGE 1 — Original leaf photograph]
[IMAGE 2 — Grad-CAM++ diagnostic attention overlay:
           amber/red regions = high model attention,
           blue regions = low model attention]
```

---

## 9. RAG Knowledge Base

### Documents (upload PDFs to `rag/knowledge_base/`)

| # | Document | Priority | Expected chunks |
|---|---|---|---|
| 1 | Cruz et al. (2024) — First report of MSV in Philippines | Must | 4–6 |
| 2 | CIMMYT Maize Disease Field Guide | Must | 20–30 |
| 3 | FAO MLN Management Guide | Must | 15–20 |
| 4 | DA Philippines corn disease protection bulletin | Strong | 10–15 |
| 5 | Beyene et al. (2017) — MLN resistance in tropical maize | Strong | 8–10 |
| 6 | Sime et al. (2021) — MSV diagnostic markers validation | Strong | 6–8 |
| 7 | EPPO MSV datasheet | Optional | 5–8 |

### Chunking Strategy
```python
CHUNK_SIZE    = 600    # tokens per chunk
CHUNK_OVERLAP = 100    # overlap between chunks
RETRIEVAL_K   = 3      # top-k chunks retrieved per query
```

### Query Strategy
```python
def build_rag_query(classification, severity_pct, cimmyt_grade):
    return (
        f"{classification} maize disease management "
        f"severity {severity_pct:.0f}% grade {cimmyt_grade} "
        f"Philippines field protocol control prevention"
    )
```

### Ingestion Command
```bash
python -m rag.ingest
# Chunks all PDFs in rag/knowledge_base/
# Embeds with Google text-embedding-004
# Stores in rag/chroma_db/ (persistent)
# Run once, or re-run when adding new PDFs
```

---

## 10. Offline Fallback

Pre-generated static guidance for 12 combinations (3 classes × 4 stages). Used when:
- No internet connection
- Gemini API call fails or times out (> 10 seconds)
- Android app reports offline status

### Static Guidance Structure
```python
OFFLINE_GUIDANCE = {
    ("HEALTHY", 0): { ... },
    ("MSV", 0): { ... },    # trace — grade 1
    ("MSV", 1): { ... },    # mild — grade 2
    ("MSV", 2): { ... },    # moderate — grade 3
    ("MSV", 3): { ... },    # severe — grade 4-5
    ("MLN", 0): { ... },
    ("MLN", 1): { ... },
    ("MLN", 2): { ... },
    ("MLN", 3): { ... },
}
# Each entry has: justification, protocol_steps (EN + TL),
#                 key_fact, immediate_actions, spread, prevention
```

### Fallback Trigger Logic
```python
def get_guidance(inputs, timeout=10):
    if inputs.get("offline"):
        return offline_fallback.get(inputs)
    try:
        return gemini_engine.generate(inputs, timeout=timeout)
    except (TimeoutError, APIError):
        return offline_fallback.get(inputs)
```

---

## 11. Conversation Manager (Chatbot)

### Session Structure
```python
sessions = {}   # in-memory, keyed by session_id
# Each session:
{
  "session_id": "abc123",
  "diagnosis_context": { ... },    # original /diagnose response
  "history": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
  ],
  "language": "english" | "tagalog",
  "created_at": datetime,
  "last_active": datetime,
}
```

### Every /chat call passes:
```python
messages = [
    {"role": "system", "content": system_prompt + diagnosis_context_json},
    *session["history"],
    {"role": "user", "content": user_message},
]
```

### Out-of-scope detection
```python
OUT_OF_SCOPE_KEYWORDS = [
    "rice", "tomato", "banana", "human", "person", "price",
    "market", "weather", "soil", "fertilizer for other crops"
]
# If detected → polite decline in English and Tagalog
# "I can only provide guidance specific to maize streak diseases
#  (MSV and MLN). For other concerns, please consult your local
#  DA agricultural extension officer."
```

---

## 12. Tagalog Handling

### Strategy
- **Automatic:** `include_tagalog: true` in `/diagnose` → Gemini generates both in one call
- **On-demand:** `/translate` endpoint → separate Gemini call for translation only
- **Chatbot:** `language: "tagalog"` in `/chat` → Gemini responds in Tagalog

### Tagalog Quality Metric
```python
# chrF score (character n-gram F-score) via sacrebleu
# Works well for Filipino/Tagalog (morphologically rich language)
# Reference: expert-validated Tagalog translation
# Target: chrF ≥ 0.60

from sacrebleu.metrics import CHRF
chrf = CHRF()
score = chrf.sentence_score(hypothesis, [reference])
```

### Expert Tagalog Audit
5-point Likert scale, scored by licensed agriculturist:
```
1 — Completely incomprehensible / clinically dangerous mistranslation
2 — Major errors, advice unclear
3 — Understandable but awkward phrasing
4 — Natural and accurate, minor phrasing issues
5 — Native-level accuracy, clinically safe
```
Target: mean score ≥ 4.0.

---

## 13. Validation and Evaluation

### Automated RAGAS Metrics
```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

# faithfulness     : does the answer contradict the RAG context?
#                    Target > 0.85
# answer_relevancy : does the answer address the actual diagnosis?
#                    Target > 0.80

# Test set: 30 synthetic diagnosis cases
# (10 HEALTHY, 10 MSV across grades, 10 MLN across grades)
```

### Expert Audit
- Evaluators: licensed agriculturists (DA Philippines or AUF agri faculty)
- Sample: 30 generated guidance outputs
- Scale: 5-point Likert per criterion

| Criterion | What it measures |
|---|---|
| Technical accuracy | Correct disease facts |
| Protocol safety | Management steps are safe and practical |
| Philippine relevance | Applicable to local conditions |
| Tagalog naturalness | Translation reads naturally |
| Tagalog clinical safety | Safety instructions not lost in translation |

### GPT-4o mini Comparison
Run same 30 test cases through GPT-4o mini. Compare:
- RAGAS faithfulness score
- RAGAS answer relevance score
- Expert audit mean score
- Response latency (ms)
- Cost per 1000 requests

---

## 14. Deployment — Google Cloud Run

### Always Free Limits (permanent, never expires)
```
Requests     : 2 million/month
vCPU time    : 180,000 vCPU-seconds/month
Memory       : 4 GB-seconds/month
Egress       : 1 GB outbound/month
```
For a thesis demo (50–200 scans/day): well within free limits permanently.

### Dockerfile
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PORT=8080
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1",
     "--timeout", "120", "app:app"]
```

### Cloud Run Config
```yaml
# cloudbuild.yaml
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/maize-nlp', '.']
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/maize-nlp']
  - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
    args:
      - 'run'
      - 'deploy'
      - 'maize-nlp'
      - '--image=gcr.io/$PROJECT_ID/maize-nlp'
      - '--region=asia-southeast1'       ← Singapore — closest to Philippines
      - '--platform=managed'
      - '--allow-unauthenticated'
      - '--memory=512Mi'
      - '--cpu=1'
      - '--timeout=120'
```

**Important:** `--allow-unauthenticated` makes `/diagnose` a public, unmetered
endpoint — anyone with the URL can call it and consume your Gemini quota.
For a thesis deployment, add a shared API key check in Flask (Android app
sends a header, e.g. `X-API-Key`, validated against an env var/Cloud Run
secret before any Gemini call is made). This is a few lines in `app.py`
and is enough scope for a thesis demo — no need for full OAuth.

### Environment Variables in Cloud Run
```
GEMINI_API_KEY      → set via Cloud Run secret (needed at runtime)
OPENAI_API_KEY      → NOT set in Cloud Run (dev/eval only — see note below)
CHROMA_PERSIST_DIR  → /app/rag/chroma_db  (baked into the image, NOT /tmp)
```

**Important:** ChromaDB is ephemeral on Cloud Run (container restarts wipe `/tmp`). The knowledge base must be bundled into the Docker image at build time — not loaded from `/tmp` at runtime. The `ingest.py` step runs during `docker build`, and the resulting `chroma_db/` folder is copied into the image at `/app/rag/chroma_db`. `CHROMA_PERSIST_DIR` must point there, not at `/tmp`, or retrieval will silently return nothing after every cold start.

**Note on `OPENAI_API_KEY`:** GPT-4o mini is comparison-only (Design Decision, Section 17) and is never called by the deployed app — so it does not need to be a Cloud Run secret. Keep it in your local `.env` for running the evaluation script only.

---

## 15. Requirements

```
# requirements.txt
flask>=3.0.0
gunicorn>=21.0.0
google-generativeai>=0.7.0
langchain>=0.2.0
langchain-google-genai>=1.0.0
langchain-chroma>=0.1.0
chromadb>=0.5.0
ragas>=0.1.0
openai>=1.30.0
pymupdf>=1.24.0          # PDF chunking (import as fitz)
sacrebleu>=2.4.0         # chrF score for Tagalog
pillow>=10.0.0
numpy>=1.24.0
python-dotenv>=1.0.0
```

---

## 16. Mock Student Output (for testing without trained model)

```python
# sample_data/mock_student_output.py

import numpy as np
from PIL import Image

MOCK_OUTPUTS = {
    "msv_moderate": {
        "classification": "MSV",
        "confidence": 0.923,
        "severity_pct": 31.4,
        "cimmyt_grade": 3,
        "monitoring_stage": 2,
        "original_image_b64": None,    # replaced with real base64 image in production
        "gradcam_overlay_b64": None,   # replaced with real base64 Grad-CAM++ overlay
    },
    "mln_severe": {
        "classification": "MLN",
        "confidence": 0.887,
        "severity_pct": 58.2,
        "cimmyt_grade": 4,
        "monitoring_stage": 3,
        "original_image_b64": None,
        "gradcam_overlay_b64": None,
    },
    "healthy": {
        "classification": "HEALTHY",
        "confidence": 0.961,
        "severity_pct": 0.0,
        "cimmyt_grade": 0,
        "monitoring_stage": 0,
        "original_image_b64": None,
        "gradcam_overlay_b64": None,
    },
    "msv_early": {
        "classification": "MSV",
        "confidence": 0.812,
        "severity_pct": 8.3,
        "cimmyt_grade": 1,
        "monitoring_stage": 1,
        "original_image_b64": None,
        "gradcam_overlay_b64": None,
    },
}

# When testing without Android app, generate placeholder images:
def make_placeholder_images():
    """Generate solid-colour placeholder base64 images for pipeline testing."""
    import base64, io
    from PIL import Image as PILImage

    def solid_b64(color, size=(224, 224)):
        img = PILImage.new("RGB", size, color)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    return {
        "original_image_b64": solid_b64((34, 139, 34)),   # forest green leaf
        "gradcam_overlay_b64": solid_b64((255, 140, 0)),  # amber heatmap
    }
```

---

## 17. Key Design Decisions

| Decision | Rationale |
|---|---|
| Gemini 2.5 Flash over 2.0 Flash | Newer model, better reasoning, same free tier, same API |
| Two images passed to Gemini (original + Grad-CAM++) | Original image: Gemini sees actual symptoms (chlorotic streaks, necrotic patches) and can describe them specifically. Grad-CAM++ overlay: Gemini sees where the Student model's attention was concentrated and can explain the XAI decision. Passing only the Grad-CAM overlay without the original limits Gemini's ability to describe real visual symptoms. |
| Multimodal over text-only XAI description | A programmatic text description of the heatmap ("attention concentrated along veins") is a proxy. Gemini seeing the actual image produces more specific, accurate, and varied justifications. Directly fulfills the "visually-grounded reasoning" thesis objective. |
| ChromaDB over FAISS | Persistent, no server required, simpler for Cloud Run bundling |
| Google text-embedding-004 over OpenAI embeddings | Same ecosystem as Gemini — single API key, single SDK |
| Chain-of-Thought explicit steps | Forces Gemini to reason before outputting — reduces hallucination on edge cases (early-stage MSV, HEALTHY with stress symptoms) |
| response_mime_type="application/json" | Forces Gemini to output valid JSON — no parsing hacks needed |
| temperature=0.2 | Low temperature = factual, consistent, reproducible outputs. Higher temperature increases creative hallucination risk in agricultural safety context |
| ChromaDB bundled in Docker image | Cloud Run containers are ephemeral — `/tmp` is wiped on restart. Knowledge base must be baked into image at build time via `ingest.py` during `docker build` |
| GPT-4o mini as comparison only | Never deployed — used only for RAGAS metric comparison in evaluation chapter |
| Session in-memory (not Redis) | Thesis demo scale — no persistence needed between app restarts |
| chrF for Tagalog evaluation | Character n-gram F-score works better than BLEU for morphologically rich languages like Filipino |
| Offline fallback pre-generated | Network unreliable in Bukidnon/South Cotabato farming areas — farmer should always get actionable guidance |
| Out-of-scope detection | Prevents chatbot from hallucinating about non-maize topics |

---

## 18. Execution Order for Setup

```bash
# 1. Clone / create project directory
mkdir maize_nlp && cd maize_nlp

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env: add GEMINI_API_KEY and OPENAI_API_KEY

# 5. Add PDF knowledge base documents
# Place PDFs in rag/knowledge_base/

# 6. Ingest knowledge base (run once, or after adding new PDFs)
python -m rag.ingest

# 7. Run locally for testing
flask run --port 5000

# 8. Test with mock data
python -m tests.test_pipeline

# 9. Build Docker image (for Cloud Run)
docker build -t maize-nlp .

# 10. Deploy to Cloud Run
gcloud run deploy maize-nlp \
  --image gcr.io/$PROJECT_ID/maize-nlp \
  --region asia-southeast1 \
  --allow-unauthenticated
```

---

## 19. References for NLP Pipeline

| Reference | Used for |
|---|---|
| Cruz et al. (2024). New Disease Reports. DOI:10.1002/ndr2.12302 | Philippine MSV context — RAG source + thesis citation |
| Beyene et al. (2017). Euphytica 213:224 | MLN 1-5 severity scale — RAG source + grading citation |
| Sime et al. (2021). Agriculture 11(2):130 | MSV 0-5 leaf-area scale — RAG source + grading citation |
| Mahmood et al. (2026). AgriChain. arXiv:2604.07814 | Visually-grounded agricultural VLM — related work |
| Sanyal et al. (2026). Scientific Reports 16(1) | RAGMail — RAG hallucination reduction — related work |
| Mondal et al. (2026). Comp. & Elec. in Agric. 248:111735 | Concept-guided neural network for crop diagnosis |
| Joseph et al. (2026). arXiv:2603.06676 | XAI + few-shot plant disease classification |
| Lewis et al. (2020). NeurIPS | RAG original paper |
| Wei et al. (2022). NeurIPS | Chain-of-Thought prompting |

---

## 20. What Is Still Needed Before Full Implementation

| Item | Status | Action |
|---|---|---|
| Gemini API key | Confirm available from Google AI Studio | Add to `.env` |
| OpenAI API key | For GPT-4o mini comparison only | Add to `.env` |
| Cruz et al. 2024 PDF | Uploaded at thesis chat start | Move to `rag/knowledge_base/` |
| CIMMYT Maize Disease Field Guide PDF | Not yet | Download from cimmyt.org |
| FAO MLN Management Guide PDF | Not yet | Download from fao.org |
| DA Philippines corn bulletin PDF | Not yet | Download from da.gov.ph |
| Beyene et al. 2017 PDF | Not yet | Obtain via institutional access |
| Sime et al. 2021 PDF | Not yet | Free — doi.org/10.3390/agriculture11020130 |
| EPPO MSV datasheet PDF | Optional | eppo.int |
| Expert Tagalog validator | Not yet | Licensed agriculturist from DA or AUF |
| Trained Student model | In progress | Needed for real Grad-CAM++ overlay input |
| Android app integration | Pending | Send app code after pipeline is complete |
