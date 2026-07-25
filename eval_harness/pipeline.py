"""
pipeline.py
-----------
Concept: THE SCORING PIPELINE

This is the glue step: for every case, take the raw output (from runner.py)
and run it through whichever scorers apply to that case. Not every scorer
runs on every case:
  - similarity only runs if `reference` is set
  - hallucination only runs if `context` is set
  - the LLM judge always runs (it's the most general-purpose check)

Each case ends up with a single dict record. This is the atomic unit that
gets saved to a results JSON file, diffed against a baseline, and printed
in the report — one consistent shape flowing through the whole system.
"""
from __future__ import annotations

import datetime
from .dataset import EvalCase
from .runner import RunOutput
from .scorers.similarity import score_similarity
from .scorers.llm_judge import score_with_judge
from .scorers.hallucination import score_hallucination


def score_case(case: EvalCase, run_output: RunOutput, use_judge: bool, use_hallucination: bool) -> dict:
    record = {
        "id": case.id,
        "prompt": case.prompt,
        "output": run_output.output,
        "tags": case.tags,
        "latency_ms": round(run_output.latency_ms, 1),
        "app_error": run_output.error,
    }

    # An app-level crash means every other score is meaningless for this case —
    # record it as an automatic fail rather than sending broken output to judges.
    if run_output.error:
        record.update({
            "similarity_score": 0.0,
            "judge_score": 0.0,
            "judge_reasoning": "skipped: app raised an error",
            "hallucinated": False,
            "unsupported_claims": [],
            "passed": False,
        })
        return record

    checks_passed = []

    if case.reference:
        sim = score_similarity(run_output.output, case.reference)
        record["similarity_score"] = sim
        checks_passed.append(sim >= case.min_similarity)
    else:
        record["similarity_score"] = None

    if use_judge:
        judge = score_with_judge(case.prompt, run_output.output, case.judge_rubric, case.min_judge_score)
        record["judge_score"] = judge["score"]
        record["judge_reasoning"] = judge["reasoning"]
        checks_passed.append(judge["score"] >= case.min_judge_score)
    else:
        record["judge_score"] = None
        record["judge_reasoning"] = None

    if use_hallucination and case.context:
        hall = score_hallucination(run_output.output, case.context)
        record["hallucinated"] = hall["hallucinated"]
        record["unsupported_claims"] = hall["unsupported_claims"]
        checks_passed.append(not hall["hallucinated"])
    else:
        record["hallucinated"] = None
        record["unsupported_claims"] = []

    # A case passes only if every check that ran for it passed.
    record["passed"] = all(checks_passed) if checks_passed else True
    return record


def build_run_summary(records: list[dict], target: str, dataset_path: str) -> dict:
    return {
        "meta": {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "target": target,
            "dataset": dataset_path,
            "num_cases": len(records),
        },
        "results": records,
    }
