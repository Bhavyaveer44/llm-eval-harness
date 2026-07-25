"""
Concept: LLM-AS-JUDGE

Similarity scoring catches "does this say roughly the same thing as the
reference" but misses things like: is it polite, does it follow formatting
instructions, does it avoid giving financial advice, etc. For that, you use
a rubric and ask a model to grade against it.

The rubric can be per-case (case.judge_rubric) or a sensible default. We
always ask for the SAME JSON shape back: {score, pass, reasoning}. Keeping
the schema constant means report.py never needs to know which rubric was
used, this is the same principle as having every pytest test return
pass/fail regardless of what it's internally testing.
"""
from __future__ import annotations

from ._judge_client import call_judge_json

DEFAULT_RUBRIC = (
    "Grade the AI's response for correctness, relevance to the prompt, and "
    "clarity. A response is a strong pass if it directly and accurately "
    "answers the prompt with no major omissions."
)

JUDGE_SYSTEM_PROMPT = """You are a strict, consistent QA grader for an AI application.
You will be given a user prompt, the AI's response, and a grading rubric.
Score the response from 0.0 (completely fails the rubric) to 1.0 (fully
satisfies it). Be skeptical — do not give high scores to vague, evasive, or
partially correct answers.

Respond with ONLY a JSON object, no other text, no markdown fences, in this
exact shape:
{"score": <float 0-1>, "pass": <bool>, "reasoning": "<one or two sentence justification>"}
"""


def score_with_judge(prompt: str, output: str, rubric: str | None, min_score: float) -> dict:
    """
    Returns {"score": float, "pass": bool, "reasoning": str}.
    On any error (bad API key, network issue, malformed JSON reply), returns
    a failed result with the error in `reasoning` rather than crashing the
    whole eval run o,ne bad judge call shouldn't kill your CI job's output.
    """
    rubric = rubric or DEFAULT_RUBRIC
    user_prompt = (
        f"RUBRIC:\n{rubric}\n\n"
        f"USER PROMPT GIVEN TO THE AI:\n{prompt}\n\n"
        f"AI'S RESPONSE TO GRADE:\n{output}\n"
    )
    try:
        result = call_judge_json(JUDGE_SYSTEM_PROMPT, user_prompt)
        score = float(result.get("score", 0.0))
        return {
            "score": round(score, 4),
            "pass": bool(result.get("pass", score >= min_score)),
            "reasoning": result.get("reasoning", ""),
        }
    except Exception as e:  # noqa: BLE001
        return {"score": 0.0, "pass": False, "reasoning": f"[judge error] {e}"}
