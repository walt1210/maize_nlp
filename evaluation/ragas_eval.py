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
"""
import random

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness

from pipeline import input_processor, rag_engine
from pipeline.gemini_engine import GeminiCallError, generate_guidance
from sample_data.mock_student_output import make_placeholder_images


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
    images = make_placeholder_images()
    original_image, segmentation_image, xai_image = input_processor.prepare_images(
        images["original_image_b64"], images["segmentation_overlay_b64"], images["xai_overlay_b64"]
    )
    monitoring_stage = input_processor.severity_to_stage(case["severity_pct"], case["classification"])
    label = input_processor.grade_label(case["classification"], case["cimmyt_grade"])
    rag_context, _sources = rag_engine.retrieve_context(
        case["classification"], case["severity_pct"], case["cimmyt_grade"]
    )

    try:
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
        answer = result.justification + " " + " ".join(result.management)
    except GeminiCallError as exc:
        answer = f"[GENERATION FAILED: {exc}]"

    return {
        "question": f"{case['classification']} grade {case['cimmyt_grade']} management guidance",
        "answer": answer,
        "contexts": [rag_context] if rag_context else [""],
    }


def main():
    cases = _build_synthetic_cases()
    print(f"Running {len(cases)} synthetic cases through the pipeline...")

    records = [run_pipeline_for_case(case) for case in cases]
    dataset = Dataset.from_list(records)

    print("Scoring with RAGAS (faithfulness, answer_relevancy)...")
    results = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
    print(results)
    return results


if __name__ == "__main__":
    main()
