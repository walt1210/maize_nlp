"""
Offline chrF evaluation for Tagalog output quality — run manually against
a CSV of {source_text, generated_tagalog, expert_reference_tagalog} rows.

Usage:
    python -m evaluation.chrf_eval path/to/tagalog_eval_set.csv

Expected CSV columns: source_text, generated_tagalog, expert_reference_tagalog
"""
import csv
import sys

from pipeline.tagalog_handler import chrf_score
import config


def evaluate_csv(csv_path: str):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            score = chrf_score(row["generated_tagalog"], row["expert_reference_tagalog"])
            rows.append({**row, "chrf_score": score})

    scores = [r["chrf_score"] for r in rows]
    mean_score = sum(scores) / len(scores) if scores else 0.0

    print(f"Evaluated {len(rows)} pairs")
    print(f"Mean chrF: {mean_score:.3f}  (target: {config.CHRF_TARGET})")
    below_target = [r for r in rows if r["chrf_score"] < config.CHRF_TARGET]
    if below_target:
        print(f"\n{len(below_target)} pairs below target:")
        for r in below_target:
            print(f"  chrF={r['chrf_score']:.3f}  source={r['source_text'][:60]!r}")

    return rows, mean_score


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m evaluation.chrf_eval path/to/tagalog_eval_set.csv")
        sys.exit(1)
    evaluate_csv(sys.argv[1])
