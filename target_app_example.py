"""
target_app_example.py
----------------------
This stands in for YOUR real app. The harness only cares that it exposes a
function matching the contract: answer(prompt: str, context: str | None) -> str

Swap this file's internals for your actual pipeline (LangChain chain,
FastAPI call, whatever) — the CLI target stays "target_app_example:answer",
or you point it at your own module.

This example calls Claude with the given context as a system prompt (a tiny
RAG-style app), so you can eval-test it out of the box. If ANTHROPIC_API_KEY
isn't set, it falls back to a dumb rule-based responder so you can still
exercise the harness end-to-end with zero API cost/setup.
"""
import os

def _real_answer(prompt: str, context: str | None) -> str:
    from groq import Groq
    client = Groq()
    system = (
        f"Answer the user's question using ONLY the following context. "
        f"Do not invent details not present in the context.\n\nCONTEXT:\n{context}"
        if context else
        "Answer the user's question helpfully and concisely."
    )
    resp = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        max_tokens=300,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content

def _fallback_answer(prompt: str, context: str | None) -> str:
    """No API key set — deterministic stand-in so you can still test the harness plumbing."""
    if context:
        return f"[fallback, no API key] Based on the info I have: {context[:120]}..."
    return f"[fallback, no API key] I don't have enough information to answer: '{prompt}'"

def answer(prompt: str, context: str | None = None) -> str:
    if os.environ.get("GROQ_API_KEY"):
        return _real_answer(prompt, context)
    return _fallback_answer(prompt, context)




# --- Try this: introduce a deliberate regression ---------------------------
# To see the harness catch a real regression, after your first successful
# `run` + `baseline update`, come back and edit the system prompt above to
# something like:
#   "Answer confidently even if the context doesn't fully cover it. Feel
#    free to add helpful, plausible-sounding extra details."
# Then run again — the hallucination scorer should flag the new unsupported
# claims, and the judge score should drop on the policy-accuracy cases.
