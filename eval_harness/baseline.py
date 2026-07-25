"""
baseline.py
-----------
Concept: BASELINES AND REGRESSION DETECTION

A baseline is just a saved copy of a previous "known good" run's results
JSON, committed to your repo (like a snapshot test). Every future `run`
compares fresh scores against it, case by case, by matching on `id`.

Why commit it to git instead of a database? Same reason snapshot tests live
in the repo: the baseline should version alongside the code/prompts that
produced it, be reviewable in a PR diff, and require no external service
to run CI. `eval-harness baseline update` is the equivalent of running
`jest --updateSnapshot` — you only do it deliberately, after reviewing that
a score change is an intentional improvement, not a silent regression.
"""
from __future__ import annotations

import json
from pathlib import Path


def load_baseline(path: str | Path) -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def save_baseline(run_summary: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(run_summary, indent=2))


def save_results(run_summary: dict, path: str | Path) -> None:
    """Same as save_baseline but semantically for a regular (non-baseline) run."""
    save_baseline(run_summary, path)
