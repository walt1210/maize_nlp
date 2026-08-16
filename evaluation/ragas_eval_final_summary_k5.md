# RAGAS Evaluation — Final Summary (RETRIEVAL_K=5, averaged across 2 runs)

## Final Reported Result

| Metric | Average | Range (2 runs) | N (per run) |
|---|---|---|---|
| Faithfulness | **0.353** | 0.292 – 0.414 | 18–19 (HEALTHY + failed generations excluded) |
| Answer Relevancy | **0.490** | 0.485 – 0.494 | 25–29 (failed generations excluded) |

## Why Two Runs Were Averaged

Gemini generation uses `temperature=0.2` (not zero) — regenerating the same
30 synthetic cases with an unchanged pipeline produces genuinely different
answer text each time, not identical output. A single run's score reflects
one sample of generated text, not a stable property of the system. Two
independent full runs under identical configuration (`RETRIEVAL_K=5`, same
prompt, same scoring methodology) were averaged to give a more honest
estimate than reporting either run in isolation.

## Run-Level Detail

**Run A:**
- Faithfulness: 0.4139 (n=18)
- Answer Relevancy: 0.4852 (n=25)
- Generation failures: 5/30 (16.7%)

**Run B:**
- Faithfulness: 0.2922 (n=19)
- Answer Relevancy: 0.4943 (n=29)
- Generation failures: 1/30 (3.3%)

## What the Variance Itself Shows

Answer Relevancy was stable across both runs (0.4852 vs 0.4943, a 0.009
spread) — the embedding-similarity-based judge produced consistent scores
despite different generated text. Faithfulness varied substantially more
(0.4139 vs 0.2922, a 0.122 spread) — the claim-decomposition-based judge is
evidently more sensitive to exactly which facts and phrasing Gemini happens
to generate on a given run. This asymmetry is itself a legitimate
methodological finding: single-run faithfulness scores from LLM-as-judge
evaluation should be treated with more caution than single-run relevancy
scores, at least for this system and judge combination.

## Full Methodology Chain (for reference)

1. Baseline (RETRIEVAL_K=3): Faithfulness 0.193, Answer Relevancy 0.536
2. Switched to RETRIEVAL_K=5: Faithfulness improved substantially (more
   retrieved context gives the judge more material to verify claims
   against); Answer Relevancy dropped slightly (richer context led to
   longer, less narrowly-targeted answers). Net trade-off judged favorable
   for an agricultural safety-guidance system — kept RETRIEVAL_K=5.
3. Tested a prompt change (added a rule requiring concise, single-sentence
   action items in immediate_actions/management, moving explanation into
   justification): Answer Relevancy improved further (0.582) but
   Faithfulness collapsed (0.149) — the concise-actions instruction
   apparently stripped out the specific, RAG-traceable phrasing that made
   claims verifiable. Reverted this change.
4. Final configuration: RETRIEVAL_K=5, original prompt (no concise-actions
   rule) — the averaged result reported above.

## Known Methodology Limitations (documented throughout evaluation)

- Diagnostic inputs (classification, confidence, severity) are synthetic,
  not produced by the real Student model from an actual leaf photo — this
  evaluates RAG+generation groundedness given an assumed-correct diagnosis,
  not end-to-end diagnostic accuracy.
- "Answer" scored is `immediate_actions + management` only, not
  `justification` — justification references visual symptoms from
  attached leaf images, which a text-only judge cannot verify.
- HEALTHY cases are excluded from Faithfulness scoring: their retrieved
  context is a single generic line, structurally insufficient to ground a
  multi-item answer regardless of quality.
- Generation-failed cases (malformed/truncated Gemini JSON output) are
  excluded from both metrics — early testing confirmed leaving them in
  measurably distorted both scores.
- The question used for Answer Relevancy is a natural, farmer-phrased
  question generated per case, not an internal classification/grade label.
- Judge model is OpenAI gpt-4o-mini (via `ragas.metrics.collections`),
  separate from the Gemini model being evaluated — a general-purpose model
  making textual-entailment judgments, not a domain agricultural expert.
