"""
Concept: WHY A SEPARATE 'JUDGE' MODEL CALL

LLM-as-judge: use an LLM call to grade another LLM's output. It's not magic,
it's just a second prompt whose whole job is grading. The trick is forcing it
to return STRUCTURED output (JSON) so your code can read a score out of it
programmatically, instead of a wall of prose you'd have to parse by hand.

Two things make judge calls trustworthy enough to use in CI:
1. A clear rubric (told exactly what to check, not "is this good?")
2. Forced JSON output, parsed defensively (models sometimes wrap JSON in
   ```json fences even when told not to — we strip that here).

This file centralizes the raw API call so both llm_judge.py and
hallucination.py share one code path (one place to change models, retry
logic, temperature, etc).
"""
import json
import os
import re

DEFAULT_MODEL = os.environ.get("EVAL_JUDGE_MODEL", "openai/gpt-oss-120b")

_client = None


def _get_client():
    global _client
    if _client is None:
        import groq
        _client = groq.Groq()  # reads GROQ_API_KEY from env
    return _client


def call_judge_json(system_prompt: str, user_prompt: str, model: str = DEFAULT_MODEL) -> dict:
    client = _get_client()
    resp = client.chat.completions.create(
        model=model,
        max_tokens=1024,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},  # Groq supports forced JSON mode — use it
    )
    text = resp.choices[0].message.content
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Judge did not return valid JSON. Raw reply:\n{text}") from e