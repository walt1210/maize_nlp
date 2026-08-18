"""
Offline RAGAS evaluation — run manually, NOT called by the live API
(see Section 7 note in the blueprint on why ragas_scores was removed
from the /diagnose response: each RAGAS metric call is itself an LLM
call, so scoring on every live request would roughly double latency
and cost for no runtime benefit).

Usage:
    python -m evaluation.ragas_eval

Builds a synthetic test set (10 HEALTHY, 10 MSV across grades, 10 MLN
across grades), runs each through the real pipeline (RAG + Gemini), and
scores the batch with RAGAS faithfulness + answer_relevancy.

JUDGE LLM: RAGAS's faithfulness/answer_relevancy metrics are LLM-as-judge
— they need their own model to grade the answers generate_guidance()
produces, separate from the Gemini call that produces those answers in
the first place. Judging uses OpenAI (gpt-4o-mini) via AsyncOpenAI +
ragas.metrics.collections, needs OPENAI_API_KEY in .env. A same-billing
approach using Gemini as judge (via google.genai + ragas's provider="google"
path) was tried first and abandoned after hitting a genuine, unresolved
contradiction in this installed ragas 0.4.3 + instructor version — see
the comment above _judge_client below for details. RAGAS's API has had
several breaking changes across versions during this debugging session;
if anything here fails on your installed version, check `pip show ragas`
first.
"""
import asyncio
import csv
import json
import os
import random
import sys
import types
from datetime import datetime
from pathlib import Path

# WORKAROUND for a confirmed upstream bug in ragas 0.4.3 (see
# github.com/vibrantlabsai/ragas issues #2741/#2745/#2753): ragas/llms/
# base.py unconditionally imports ChatVertexAI from a langchain_community
# path that was removed when ChatVertexAI moved to the separate
# langchain-google-vertexai package — this crashes `import ragas` for
# EVERY user on modern langchain-community, not just people using Vertex
# AI. This is unrelated to which judge provider is actually used below
# (currently OpenAI) — it fires purely from `import ragas` itself, so
# it's needed regardless. The stub class is never actually instantiated
# or called. Remove this once ragas ships a real fix (check `pip show
# ragas` for a version newer than 0.4.3 before assuming this workaround
# is still needed).
if "langchain_community.chat_models.vertexai" not in sys.modules:
    try:
        import langchain_community.chat_models.vertexai  # noqa: F401 — just probing
    except ModuleNotFoundError:
        _vertexai_stub = types.ModuleType("langchain_community.chat_models.vertexai")

        class _UnusedChatVertexAIStub:
            """Never instantiated — exists only to satisfy ragas's broken
            import. See workaround comment above."""

        _vertexai_stub.ChatVertexAI = _UnusedChatVertexAIStub
        sys.modules["langchain_community.chat_models.vertexai"] = _vertexai_stub


from openai import AsyncOpenAI
from ragas.embeddings import embedding_factory
from ragas.llms import llm_factory

import config
from pipeline import input_processor, rag_engine, xai_engine
from pipeline.gemini_engine import GeminiCallError, generate_guidance
from sample_data.mock_student_output import make_placeholder_images

# The legacy `from ragas.metrics import answer_relevancy, faithfulness`
# singletons (what this file used to import — hence the deprecation
# warning on every run) expect an older embeddings interface that isn't
# consistently implemented across providers in ragas 0.4.3. The current
# collections API instantiates metrics as classes with the LLM/embeddings
# passed directly at construction.
# Class name uncertainty: RAGAS's own migration docs call this
# "AnswerRelevancy"; at least one other current doc page calls the same
# concept "ResponseRelevancy". Trying both defensively.
from ragas.metrics.collections import Faithfulness
try:
    from ragas.metrics.collections import AnswerRelevancy
except ImportError:
    from ragas.metrics.collections import ResponseRelevancy as AnswerRelevancy

# JUDGE MODEL: gemini-3.5-flash via google.genai/instructor was tried
# first to keep everything on one billing account, but hit a genuine,
# unresolved contradiction in this installed ragas 0.4.3 + instructor
# version: .ascore() demands an async-capable client ("Cannot use
# agenerate() with a synchronous client"), while llm_factory's internal
# instructor.from_genai() adapter demands a plain sync google.genai.Client
# and rejects the async one outright. Neither client type works. Every
# real, working example of ragas.metrics.collections found while
# debugging this used AsyncOpenAI — switched to that instead of
# continuing to chase an apparently-unsupported provider combination.
# Needs OPENAI_API_KEY in .env (config.py already has a slot for it).
# Separate, small cost from the Gemini billing used for generation.
_judge_client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
_judge_llm = llm_factory("gpt-4o-mini", client=_judge_client)

# answer_relevancy needs its own embeddings model — using OpenAI's here
# too for consistency with the judge LLM, rather than mixing providers
# and risking another compatibility gap.
_judge_embeddings = embedding_factory("openai", model="text-embedding-3-small", client=_judge_client)

# LLM/embeddings baked into the metric objects themselves (collections
# API), rather than passed to evaluate() — evaluate() was confirmed at
# runtime to reject collections-style metric objects entirely.
_faithfulness_metric = Faithfulness(llm=_judge_llm)
_answer_relevancy_metric = AnswerRelevancy(llm=_judge_llm, embeddings=_judge_embeddings)


def _build_synthetic_cases(n_per_class: int = 10) -> list[dict]:
    cases = []
    random.seed(42)
    for classification, grades in (("MSV", range(1, 6)), ("MLN", range(1, 6))):
        for i in range(n_per_class):
            grade = list(grades)[i % 5]
            severity = random.uniform((grade - 1) * 20, grade * 20)
            cases.append(
                {
                    "classification": classification,
                    "confidence": random.uniform(0.7, 0.98),
                    "severity_pct": min(severity, 99.9),
                    "cimmyt_grade": grade,
                }
            )
    for _ in range(n_per_class):
        cases.append(
            {"classification": "HEALTHY", "confidence": random.uniform(0.85, 0.99), "severity_pct": 0.0, "cimmyt_grade": 0}
        )
    return cases


def run_pipeline_for_case(case: dict) -> dict:
    # Requires a real Student checkpoint at pipeline/student_model/checkpoints/
    # — this evaluation exercises the full deployed pipeline, including
    # server-side XAI generation (see pipeline/xai_engine.py), not a
    # lightweight mock. If you're iterating on RAG/prompt quality only and
    # don't have a checkpoint bundled yet, this will fail at generate_overlays().
    images = make_placeholder_images()
    original_image = input_processor.prepare_original_image(images["original_image_b64"])

    # Was: only generate_guidance()'s GeminiCallError was caught — the
    # XAI overlay generation and RAG retrieval calls just below were
    # completely unguarded. A single failure in either one crashed this
    # entire function uncaught, which crashed the whole list
    # comprehension building all 30 records in _load_or_generate_records
    # — losing every already-generated (and already-paid-for) case that
    # came before the failure too, since caching only happens after the
    # full list finishes. Broadened to catch any exception for the same
    # reason app.py's /diagnose route now degrades gracefully on XAI/RAG
    # failure instead of hard-failing.
    rag_context = ""
    try:
        segmentation_image, xai_image = xai_engine.generate_overlays(original_image, case["classification"])
        segmentation_image = input_processor.resize_for_gemini(segmentation_image)
        xai_image = input_processor.resize_for_gemini(xai_image)

        monitoring_stage = input_processor.severity_to_stage(case["severity_pct"], case["classification"])
        label = input_processor.grade_label(case["classification"], case["cimmyt_grade"])
        rag_context, _sources = rag_engine.retrieve_context(
            case["classification"], case["severity_pct"], case["cimmyt_grade"]
        )

        result = generate_guidance(
            classification=case["classification"],
            confidence=case["confidence"],
            severity_pct=case["severity_pct"],
            cimmyt_grade=case["cimmyt_grade"],
            monitoring_stage=monitoring_stage,
            grade_label=label,
            rag_context=rag_context,
            original_image=original_image,
            segmentation_image=segmentation_image,
            xai_image=xai_image,
        )
        justification = result.justification
        immediate_actions = result.immediate_actions
        management = result.management
    except Exception as exc:
        failure_msg = f"[GENERATION FAILED: {exc}]"
        justification = failure_msg
        immediate_actions = [failure_msg]
        management = [failure_msg]

    # "answer" (used for RAGAS scoring) is deliberately immediate_actions +
    # management, NOT justification. The eval question asks "what should
    # I do" — immediate_actions + management directly answers that and is
    # groundable in rag_context; justification explains WHY (grounded in
    # the attached images, which a text-only judge can't see at all), so
    # including it was diluting both answer_relevancy (wrong content for
    # the question asked) and faithfulness (unverifiable visual claims
    # mixed in with verifiable RAG-grounded claims). justification is
    # still cached separately below in case you want to score/inspect it
    # on its own later — this doesn't remove it from the real /diagnose
    # response, only from what this eval script treats as "the answer".
    answer = " ".join(immediate_actions) + " " + " ".join(management)

    return {
        "question": f"{case['classification']} grade {case['cimmyt_grade']} management guidance",
        "justification": justification,
        "answer": answer,
        "contexts": [rag_context] if rag_context else [""],
    }


# Override for cheap sanity-checking before committing to the real,
# full-cost run — e.g. in PowerShell:
#   $env:RAGAS_EVAL_N_PER_CLASS = "1"
#   python -m evaluation.ragas_eval
# ...runs 3 cases (1 per class) instead of 30, ~1/10th the cost, to
# confirm a code change actually works before paying for the real run.
# Unset it (or just open a new terminal) to go back to the real n=10
# (30 total cases) used for actual thesis results.
N_PER_CLASS = int(os.environ.get("RAGAS_EVAL_N_PER_CLASS", "10"))

# Filename includes N_PER_CLASS AND config.RETRIEVAL_K so a small test
# run, the real run, and different RETRIEVAL_K experiments never collide
# or get mixed up — no risk of accidentally scoring against generations
# made with a different RAG retrieval setting.
_CACHE_PATH = (
    Path(__file__).resolve().parent
    / f"ragas_eval_generation_cache_n{N_PER_CLASS}_k{config.RETRIEVAL_K}.json"
)

# Explicit override to re-score ANY existing cache file directly,
# regardless of the currently-configured N_PER_CLASS/RETRIEVAL_K — e.g.
# to cleanly re-score an older K=3 dataset with the current scoring
# logic while config.py is set to RETRIEVAL_K=5. Usage:
#   $env:RAGAS_EVAL_CACHE_FILE = "ragas_eval_generation_cache_n10.json"
#   python -m evaluation.ragas_eval
# When set, generation is NEVER triggered even if the file is somehow
# missing — this mode is purely for re-scoring something that already
# exists, not for generating anything new.
_CACHE_OVERRIDE = os.environ.get("RAGAS_EVAL_CACHE_FILE")
if _CACHE_OVERRIDE:
    _CACHE_PATH = Path(__file__).resolve().parent / _CACHE_OVERRIDE


def _load_or_generate_records(cases: list[dict]) -> list[dict]:
    # Generation (run_pipeline_for_case) makes real, billed Gemini calls —
    # 30 of them, each with 3 images + RAG context. Getting the RAGAS
    # SCORING step right (judge LLM, embeddings, provider quirks) can take
    # several attempts, and there's no reason to re-pay for generation on
    # every one of those attempts when nothing about the generation logic
    # changed. Delete the relevant cache file manually whenever you
    # actually want fresh generations (e.g. after a real prompt or
    # pipeline change) — this does NOT auto-invalidate on code changes.
    if _CACHE_OVERRIDE and not os.path.exists(_CACHE_PATH):
        raise FileNotFoundError(
            f"RAGAS_EVAL_CACHE_FILE={_CACHE_OVERRIDE} was set but "
            f"{_CACHE_PATH} doesn't exist. Override mode never generates "
            f"new data — check the filename, or unset the env var to "
            f"generate/use the normal N_PER_CLASS+RETRIEVAL_K cache."
        )

    if os.path.exists(_CACHE_PATH):
        print(f"Found cached generations at {_CACHE_PATH} — reusing them "
              f"(delete this file if you want fresh generations).")
        with open(_CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    print(f"No cache found — running {len(cases)} synthetic cases through "
          f"the pipeline (this makes real, billed Gemini calls)...")
    records = [run_pipeline_for_case(case) for case in cases]

    with open(_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"Cached generations to {_CACHE_PATH} for reuse on future runs.")

    return records


async def _score_all_records(records: list[dict]) -> tuple[dict, list[dict]]:
    """
    Scores every cached record with both metrics directly via .ascore(),
    NOT evaluate(dataset, metrics=[...]) — confirmed at runtime that this
    installed ragas version's evaluate() explicitly rejects
    collections-style metric objects (TypeError: "All metrics must be
    initialised metric objects, e.g: metrics=[BleuScore(), AspectCritic()]"
    — those are legacy-API examples, not collections-API ones). The
    collections API's own docs show per-sample async .ascore() as the
    real intended usage pattern for these classes, not evaluate().

    answer_relevancy classically doesn't use retrieved_contexts at all
    (it only compares the response against reverse-generated questions
    via embeddings) — deliberately NOT passing it here, unlike
    faithfulness which needs it to check grounding.

    HEALTHY records are excluded from faithfulness specifically (still
    included in answer_relevancy). Confirmed from a real inspected
    sample: offline/static_guidance.py's HEALTHY entry retrieves a
    single one-line context ("No disease detected — no management
    protocol required.") while the generated answer still lists 7
    distinct recommendations — none traceable to that one line. This
    isn't a generation quality problem, it's that HEALTHY genuinely has
    almost no RAG content to check claims against, so including it would
    understate how well-grounded the disease-case (MSV/MLN) guidance
    actually is by averaging it against cases that structurally can't
    score well regardless of answer quality.

    GENERATION-FAILED records are excluded from BOTH metrics. Confirmed
    from a real run: run_pipeline_for_case's except GeminiCallError
    branch builds a "[GENERATION FAILED: ...]" placeholder string when
    Gemini's response fails validation (e.g. truncated/malformed JSON),
    and — before this fix — that placeholder was being silently scored
    as if it were real generated content. It's not an answer to
    anything, so answer_relevancy scored it near 0 every time, and
    faithfulness scored it essentially at random depending on how the
    judge happened to parse the error text as "claims". This measurably
    distorted aggregate scores in a real run (confirmed: excluding 5
    failed-out-of-30 cases moved answer_relevancy from 0.406 to 0.487).
    failure_count is reported in the aggregate so the failure RATE
    itself is visible — a real signal worth reporting, since generation
    failures are informative (e.g. about whether GEMINI_MAX_OUTPUT_TOKENS
    needs to scale with RETRIEVAL_K, not just noise to discard silently).

    Returns (aggregate_dict, per_case_results_list) — the per-case list
    is needed for CSV/Markdown export, not just the summary printed to
    the console.
    """
    faithfulness_scores = []
    answer_relevancy_scores = []
    faithfulness_skipped_healthy = 0
    generation_failed_count = 0
    per_case_results = []

    for i, r in enumerate(records):
        user_input = r["question"]
        response = r["answer"]
        retrieved_contexts = r["contexts"]
        classification = r.get("classification")
        cimmyt_grade = r.get("cimmyt_grade")
        generation_failed = "GENERATION FAILED" in response

        case_result = {
            "index": i,
            "classification": classification,
            "cimmyt_grade": cimmyt_grade,
            "generation_failed": generation_failed,
            "faithfulness": None,
            "answer_relevancy": None,
        }

        if generation_failed:
            generation_failed_count += 1
            per_case_results.append(case_result)
            continue

        if classification == "HEALTHY":
            faithfulness_skipped_healthy += 1
        else:
            try:
                f_result = await _faithfulness_metric.ascore(
                    user_input=user_input,
                    response=response,
                    retrieved_contexts=retrieved_contexts,
                )
                faithfulness_scores.append(f_result.value)
                case_result["faithfulness"] = f_result.value
            except Exception as exc:
                print(f"[{i}] faithfulness scoring failed: {exc}")
                case_result["faithfulness"] = f"ERROR: {exc}"

        try:
            ar_result = await _answer_relevancy_metric.ascore(
                user_input=user_input,
                response=response,
            )
            answer_relevancy_scores.append(ar_result.value)
            case_result["answer_relevancy"] = ar_result.value
        except Exception as exc:
            print(f"[{i}] answer_relevancy scoring failed: {exc}")
            case_result["answer_relevancy"] = f"ERROR: {exc}"

        per_case_results.append(case_result)

    def _avg(values: list[float]) -> float:
        return sum(values) / len(values) if values else float("nan")

    aggregate = {
        "faithfulness": _avg(faithfulness_scores),
        "faithfulness_n": len(faithfulness_scores),
        "faithfulness_skipped_healthy": faithfulness_skipped_healthy,
        "answer_relevancy": _avg(answer_relevancy_scores),
        "answer_relevancy_n": len(answer_relevancy_scores),
        "generation_failed_count": generation_failed_count,
        "generation_failed_rate": generation_failed_count / len(records) if records else 0.0,
    }
    return aggregate, per_case_results


def _natural_question(case: dict) -> str:
    """
    A farmer-phrased question, used ONLY for scoring (never sent to
    Gemini during generation — see run_pipeline_for_case, which still
    uses its own templated "question" for the cached record).
    answer_relevancy generates candidate questions from the answer and
    compares them to this field via embeddings — the original templated
    label ("MSV grade 3 management guidance") doesn't resemble a real
    question, which likely depressed answer_relevancy independent of
    actual answer quality. This doesn't touch faithfulness at all.
    """
    if case["classification"] == "HEALTHY":
        return "My maize leaf looks healthy — is there anything I should do?"
    return (
        f"My maize is showing {case['classification']} symptoms at "
        f"about {case['severity_pct']:.0f}% severity (grade "
        f"{case['cimmyt_grade']}). What should I do?"
    )


def _export_csv(per_case_results: list[dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index", "classification", "cimmyt_grade",
                "generation_failed", "faithfulness", "answer_relevancy",
            ],
        )
        writer.writeheader()
        writer.writerows(per_case_results)
    print(f"Wrote per-case CSV to {path}")


def _fmt_score(value) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if value is None:
        return "—"
    return str(value)  # error strings


def _export_markdown(aggregate: dict, per_case_results: list[dict], path: Path) -> None:
    lines = [
        "# RAGAS Evaluation Results — MAIze",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Judge: OpenAI gpt-4o-mini (via ragas.metrics.collections)",
        f"config.RETRIEVAL_K: {config.RETRIEVAL_K}",
        f"Cases: {len(per_case_results)} (n_per_class={N_PER_CLASS})",
        f"Generation failures: {aggregate['generation_failed_count']} "
        f"({aggregate['generation_failed_rate']:.1%}) — excluded from both metrics below",
        "",
        "## Aggregate Scores",
        "",
        "| Metric | Score | N | Notes |",
        "|---|---|---|---|",
        f"| Faithfulness | {_fmt_score(aggregate['faithfulness'])} | "
        f"{aggregate['faithfulness_n']} | HEALTHY excluded "
        f"({aggregate['faithfulness_skipped_healthy']} cases), generation "
        f"failures excluded ({aggregate['generation_failed_count']} cases) |",
        f"| Answer Relevancy | {_fmt_score(aggregate['answer_relevancy'])} | "
        f"{aggregate['answer_relevancy_n']} | Generation failures excluded "
        f"({aggregate['generation_failed_count']} cases), otherwise all cases |",
        "",
        "## Methodology Notes",
        "",
        "- Generation-failed cases (Gemini response failed JSON validation, "
        "e.g. truncated output) are excluded from BOTH metrics — a failure "
        "placeholder string is not real content and was confirmed, in a "
        "real run, to distort both scores when left in (5/30 failed cases "
        "moved answer_relevancy from 0.406 to 0.487 once excluded).",
        "- HEALTHY cases excluded from faithfulness scoring: their retrieved "
        "context is a single generic line, structurally insufficient to "
        "ground a multi-item generated answer regardless of answer quality.",
        "- \"answer\" used for scoring is `immediate_actions + management` "
        "only, NOT `justification` — justification references visual "
        "symptoms from the attached leaf images, which a text-only judge "
        "has no way to verify against retrieved text context.",
        "- The question used for `answer_relevancy` scoring is a natural, "
        "farmer-phrased question generated per case, not the internal "
        "classification/grade label used elsewhere in this pipeline.",
        "- Diagnostic inputs (classification, confidence, severity) are "
        "synthetic, not produced by the real Student model from an actual "
        "leaf photo — this evaluates RAG+generation groundedness given an "
        "assumed-correct diagnosis, not end-to-end diagnostic accuracy.",
        "",
        "## Per-Case Scores",
        "",
        "| # | Classification | Grade | Gen. Failed | Faithfulness | Answer Relevancy |",
        "|---|---|---|---|---|---|",
    ]
    for r in per_case_results:
        lines.append(
            f"| {r['index']} | {r['classification']} | {r['cimmyt_grade']} | "
            f"{'YES' if r.get('generation_failed') else ''} | "
            f"{_fmt_score(r['faithfulness'])} | {_fmt_score(r['answer_relevancy'])} |"
        )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote markdown report to {path}")


def main():
    cases = _build_synthetic_cases(n_per_class=N_PER_CLASS)
    print(f"Running with n_per_class={N_PER_CLASS} ({len(cases)} total cases) — "
          f"{'TEST RUN, small subset' if N_PER_CLASS < 10 else 'full run'}.")
    records = _load_or_generate_records(cases)

    # cases and records are guaranteed to be in the same order: both come
    # from the same deterministic, fixed-seed _build_synthetic_cases()
    # call, and records were generated by iterating that exact list (see
    # _load_or_generate_records). Safe to zip without re-matching by
    # content. This patches "question" and adds "classification"/
    # "cimmyt_grade" in memory only — the cache file on disk is left as
    # pure raw generation output, not mixed with scoring-only metadata.
    for case, record in zip(cases, records):
        record["question"] = _natural_question(case)
        record["classification"] = case["classification"]
        record["cimmyt_grade"] = case["cimmyt_grade"]

    print("Scoring with RAGAS (faithfulness, answer_relevancy) using OpenAI as judge...")
    aggregate, per_case_results = asyncio.run(_score_all_records(records))
    print(aggregate)

    output_dir = Path(__file__).resolve().parent
    if _CACHE_OVERRIDE:
        # Derive suffix from the override filename itself (e.g.
        # "ragas_eval_generation_cache_n10.json" -> "_n10_rescored"),
        # not from the current N_PER_CLASS/RETRIEVAL_K config — those
        # don't necessarily describe what's actually being scored when
        # re-scoring an old cache under a different current config.
        stem = Path(_CACHE_OVERRIDE).stem.replace("ragas_eval_generation_cache", "")
        suffix = f"{stem}_rescored"
    else:
        suffix = f"_n{N_PER_CLASS}_k{config.RETRIEVAL_K}"
    _export_csv(per_case_results, output_dir / f"ragas_eval_results{suffix}.csv")
    _export_markdown(aggregate, per_case_results, output_dir / f"ragas_eval_results{suffix}.md")

    return aggregate


if __name__ == "__main__":
    main()
