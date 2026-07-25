"""
hallucination.py
-----------------
Concept: HALLUCINATION DETECTION

A "hallucination" here specifically means: the app's output contains a
factual claim that is NOT supported by the source context it was given
(e.g. retrieved documents in a RAG pipeline). This is different from the
judge score above — an answer can be well-written and still hallucinate.

We only run this check when `case.context` exists, because without a
source-of-truth there's nothing to check the output against — you'd just be
asking "is this true in general," which is a much harder and less reliable
question for a judge model to answer.

The technique: ask the judge model to extract every factual claim from the
output and label EACH claim as "supported" or "unsupported" by the context.
This is more reliable than asking a single "is anything hallucinated?"
yes/no question, because forcing the model to enumerate claims first makes
it actually check them one at a time (similar to why chain-of-thought
improves accuracy) instead of pattern-matching to a quick "looks fine".
"""
from __future__ import annotations

from ._judge_client import call_judge_json

HALLUCINATION_SYSTEM_PROMPT = """You are a fact-checking grader. You will be given
a SOURCE CONTEXT (the only ground truth available) and an AI RESPONSE that was
supposed to be grounded in that context.

List every distinct factual claim in the AI RESPONSE. For each claim, decide
if it is directly supported by the SOURCE CONTEXT. A claim is "unsupported"
if the context doesn't mention it, contradicts it, or if the response adds
specific details (numbers, names, dates) not present in the context.

Respond with ONLY a JSON object, no other text, no markdown fences, in this
exact shape:
{"unsupported_claims": ["<claim text>", ...], "hallucinated": <bool>}
Set "hallucinated" to true if unsupported_claims is non-empty.
"""


def score_hallucination(output: str, context: str) -> dict:
    """
    Returns {"hallucinated": bool, "unsupported_claims": list[str]}.
    Returns hallucinated=False with an error note on failure — we fail
    OPEN here deliberately (see README for the tradeoff) so a broken judge
    call doesn't itself look like a hallucination in your report.
    """
    if not context or not context.strip():
        return {"hallucinated": False, "unsupported_claims": [], "note": "no context provided, check skipped"}

    user_prompt = f"SOURCE CONTEXT:\n{context}\n\nAI RESPONSE:\n{output}\n"
    try:
        result = call_judge_json(HALLUCINATION_SYSTEM_PROMPT, user_prompt)
        claims = result.get("unsupported_claims", [])
        return {
            "hallucinated": bool(result.get("hallucinated", len(claims) > 0)),
            "unsupported_claims": claims,
        }
    except Exception as e:  # noqa: BLE001
        return {"hallucinated": False, "unsupported_claims": [], "note": f"[hallucination check error] {e}"}
