"""Evaluate DocQuery AI against a small held-out question set.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/evaluate.py
    PYTHONPATH=src .venv/bin/python scripts/evaluate.py --output reports/eval_report.md
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from query import answer_question, get_llm, load_retriever

ROOT_DIR = Path(__file__).resolve().parent.parent
QUESTIONS_FILE = ROOT_DIR / "sample_data" / "held_out_questions.json"


@dataclass
class EvalItem:
    question: str
    expected_phrases: list[str]
    note: str


def load_items(path: Path) -> list[EvalItem]:
    payload = json.loads(path.read_text())
    return [
        EvalItem(
            question=item["question"],
            expected_phrases=item["expected_phrases"],
            note=item.get("note", ""),
        )
        for item in payload
    ]


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def score_answer(answer: str, expected_phrases: Iterable[str]) -> tuple[bool, list[str]]:
    normalized_answer = normalize(answer)
    answer_tokens = set(re.findall(r"[a-z0-9]+", normalized_answer))
    missing = []
    for phrase in expected_phrases:
        phrase_tokens = [token for token in re.findall(r"[a-z0-9]+", normalize(phrase)) if token not in {"the", "a", "an", "to", "of", "and", "or"}]
        if not phrase_tokens:
            continue
        covered = sum(1 for token in phrase_tokens if token in answer_tokens)
        coverage = covered / len(phrase_tokens)
        if coverage < 0.75:
            missing.append(phrase)
    return len(missing) == 0, missing


def format_report(rows: list[dict], total: int, passed: int) -> str:
    lines = [
        "# DocQuery AI Evaluation Report",
        "",
        f"- Questions: {total}",
        f"- Passed: {passed}",
        f"- Accuracy: {passed / total:.0%}",
        "",
        "| # | Question | Passed | Expected | Answer |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        expected = "<br>".join(row["expected_phrases"])
        answer = row["answer"].replace("|", "\\|")
        lines.append(
            f"| {row['index']} | {row['question']} | {'Yes' if row['passed'] else 'No'} | {expected} | {answer} |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, default=QUESTIONS_FILE)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    retriever = load_retriever()
    llm = get_llm()
    items = load_items(args.questions)

    rows = []
    passed = 0
    for index, item in enumerate(items, start=1):
        chunks = retriever.invoke(item.question)
        answer = answer_question(item.question, retriever, llm, chunks=chunks)
        is_passed, missing = score_answer(answer, item.expected_phrases)
        if is_passed:
            passed += 1
        rows.append(
            {
                "index": index,
                "question": item.question,
                "expected_phrases": item.expected_phrases,
                "answer": answer,
                "passed": is_passed,
                "missing": missing,
                "note": item.note,
            }
        )
        status = "PASS" if is_passed else "FAIL"
        print(f"[{status}] {item.question}")
        print(f"Answer: {answer}")
        if missing:
            print(f"Missing: {', '.join(missing)}")
        print()

    report = format_report(rows, len(items), passed)
    print(report)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n")
        print(f"\nWrote report to {args.output}")

    return 0 if passed == len(items) else 1


if __name__ == "__main__":
    raise SystemExit(main())