"""
report.py
---------
Concept: DIFFING AND REPORTING (the "regression" part of regression testing)

This is where the harness earns the "CI/CD for prompts" description. Given
the current run's records and an (optional) baseline run's records, we:

1. Match cases by `id` between the two runs.
2. Compute score deltas per case (judge_score, similarity_score).
3. Classify each case as: NEW (wasn't in baseline), REGRESSED (score dropped
   more than a tolerance), IMPROVED, REMOVED (was in baseline, missing now),
   or UNCHANGED.
4. A regression in ANY case, or any FAIL in the current run, makes the whole
   report exit non-zero — this is the hook that fails a CI pipeline.

REGRESSION_TOLERANCE exists because judge/similarity scores have natural
noise between runs (LLM judges aren't perfectly deterministic even at
temperature 0). A tiny wobble shouldn't fail your build; a real drop should.
"""
from __future__ import annotations

from tabulate import tabulate

REGRESSION_TOLERANCE = 0.05  # score drops smaller than this are noise, not signal


def diff_against_baseline(current_records: list[dict], baseline_records: list[dict] | None) -> dict:
    baseline_by_id = {r["id"]: r for r in baseline_records} if baseline_records else {}
    current_by_id = {r["id"]: r for r in current_records}

    diffs = []
    for case_id, cur in current_by_id.items():
        base = baseline_by_id.get(case_id)
        if base is None:
            diffs.append({"id": case_id, "status": "NEW", "delta": None})
            continue

        cur_score = cur.get("judge_score")
        base_score = base.get("judge_score")
        delta = None
        status = "UNCHANGED"
        if cur_score is not None and base_score is not None:
            delta = round(cur_score - base_score, 4)
            if delta <= -REGRESSION_TOLERANCE:
                status = "REGRESSED"
            elif delta >= REGRESSION_TOLERANCE:
                status = "IMPROVED"
        if cur.get("hallucinated") and not base.get("hallucinated"):
            status = "REGRESSED"  # a new hallucination is always a regression, score aside
        diffs.append({"id": case_id, "status": status, "delta": delta})

    for case_id in baseline_by_id:
        if case_id not in current_by_id:
            diffs.append({"id": case_id, "status": "REMOVED", "delta": None})

    return {
        "diffs": diffs,
        "has_regression": any(d["status"] == "REGRESSED" for d in diffs),
    }


def format_report(run_summary: dict, diff: dict | None) -> str:
    records = run_summary["results"]
    status_by_id = {d["id"]: d for d in diff["diffs"]} if diff else {}

    rows = []
    for r in records:
        drift = ""
        if r["id"] in status_by_id:
            d = status_by_id[r["id"]]
            delta_str = f" ({d['delta']:+.2f})" if d["delta"] is not None else ""
            drift = f"{d['status']}{delta_str}"
        rows.append([
            r["id"],
            "PASS" if r["passed"] else "FAIL",
            r.get("similarity_score"),
            r.get("judge_score"),
            "YES" if r.get("hallucinated") else ("-" if r.get("hallucinated") is None else "no"),
            f"{r['latency_ms']:.0f}ms",
            drift,
        ])

    table = tabulate(
        rows,
        headers=["case", "result", "similarity", "judge", "hallucination", "latency", "vs baseline"],
        tablefmt="github",
    )

    total = len(records)
    passed = sum(1 for r in records if r["passed"])
    lines = [
        f"Eval run: {run_summary['meta']['timestamp']}  |  target: {run_summary['meta']['target']}",
        f"Cases: {total}   Passed: {passed}   Failed: {total - passed}",
        "",
        table,
    ]
    if diff and diff["has_regression"]:
        lines.append("\n⚠️  REGRESSION DETECTED vs baseline — see 'REGRESSED' rows above.")
    return "\n".join(lines)


def build_slack_payload(run_summary: dict, diff: dict | None) -> dict:
    """
    Slack incoming webhooks accept a simple {"text": "..."} JSON body for
    plain messages. This is intentionally minimal — you can swap this for
    Slack "blocks" for richer formatting once the basic pipe works.
    """
    total = len(run_summary["results"])
    passed = sum(1 for r in run_summary["results"] if r["passed"])
    header = f"*LLM Eval Run* — {passed}/{total} passed, target `{run_summary['meta']['target']}`"
    if diff and diff["has_regression"]:
        regressed_ids = [d["id"] for d in diff["diffs"] if d["status"] == "REGRESSED"]
        header += f"\n:rotating_light: Regressions: {', '.join(regressed_ids)}"
    return {"text": header}


def post_to_slack(webhook_url: str, payload: dict) -> None:
    import requests
    resp = requests.post(webhook_url, json=payload, timeout=10)
    resp.raise_for_status()
