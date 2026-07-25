"""
Concept: THE GOLDEN DATASET

A "golden dataset" is a curated list of test cases you trust. 
Each case is a prompt plus (optionally) a reference answer, source context, and a scoring rubric. 
This is the single most important artifact in an eval harness — garbage cases in means garbage signal out.

Design choices explained:

- `reference` is optional because not every case has one "correct" answer.
  Without it, similarity scoring is skipped for that case and you rely on the LLM judge instead.

- `context` is the source-of-truth text (what a RAG pipeline retrieved).
  It's what the hallucination scorer checks the output against.
  Without it, hallucination checking is skipped for that case.

- `judge_rubric` lets you write a per-case grading instruction.
  If omitted,a generic rubric is used.

- `min_similarity` / `min_judge_score` let you set pass/fail thresholds per
  case, because not all cases deserve the same bar.
"""
from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EvalCase:
    id: str
    prompt: str
    reference: str | None = None
    context: str | None = None
    judge_rubric: str | None = None
    tags: list[str] = field(default_factory=list)
    min_similarity: float = 0.5
    min_judge_score: float = 0.6

    @staticmethod
    def from_dict(d: dict) -> "EvalCase":
        if "id" not in d or "prompt" not in d:
            raise ValueError(f"Every case needs at least 'id' and 'prompt'. Got: {d}")
        return EvalCase(
            id=d["id"],
            prompt=d["prompt"],
            reference=d.get("reference"),
            context=d.get("context"),
            judge_rubric=d.get("judge_rubric"),
            tags=d.get("tags", []),
            min_similarity=float(d.get("min_similarity", 0.5)),
            min_judge_score=float(d.get("min_judge_score", 0.6)),
        )


def load_dataset(path: str | Path) -> list[EvalCase]:
    """Load and validate a golden dataset YAML file into EvalCase objects."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    raw = yaml.safe_load(path.read_text())
    if not raw or "cases" not in raw:
        raise ValueError("Dataset YAML must have a top-level 'cases' list.")

    cases = [EvalCase.from_dict(c) for c in raw["cases"]]

    # Fail loudly on duplicate IDs, baseline diffing relies on IDs being unique.
    ids = [c.id for c in cases]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"Duplicate case IDs found (must be unique): {dupes}")

    return cases
