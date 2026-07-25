"""
similarity.py
-------------
Concept: SEMANTIC SIMILARITY SCORING

Why not just check output == reference? Because LLM output is never exactly
reproducible — different wording, different order, same meaning. So instead
we turn text into vectors and measure the angle between them (cosine
similarity). 1.0 = identical direction (very similar meaning), 0 = unrelated.

TWO WAYS TO GET VECTORS (this is the important thing to understand):

1. TF-IDF (what this file uses): a classic, dependency-light technique.
   Each word gets a weight based on how often it appears in this text vs
   how rare it is overall. It only catches WORD OVERLAP, not true meaning
   — "the cat sat" and "a feline rested" would score low even though they
   mean similar things. It's fast, free, and needs no API key, which is
   why it's the default here.

2. Embedding models (production-grade): a neural network (e.g. OpenAI's
   text-embedding-3, Voyage, or a local sentence-transformers model) maps
   text into a vector space trained so that semantically similar text ends
   up close together, even with zero word overlap. This is what you'd
   swap in for a real production eval suite. The interface below
   (`embed(text) -> vector`, `cosine(v1, v2) -> float`) is written so you
   can swap TF-IDF for a real embedding call without touching any other
   file — that's the point of isolating scorers behind one function.
"""
from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def score_similarity(output: str, reference: str) -> float:
    """
    Returns a similarity score in [0, 1] between generated output and a
    reference answer. Returns 0.0 for degenerate cases (empty strings) so
    a crashed/empty app output always fails rather than raising.
    """
    if not output.strip() or not reference.strip():
        return 0.0

    # Fit the vectorizer on JUST these two texts. This is a simplification
    # (production systems fit/embed against a fixed vocabulary or use a
    # pretrained embedding model instead), but it works fine pairwise and
    # needs zero external dependencies or API calls.
    vectorizer = TfidfVectorizer().fit([output, reference])
    vectors = vectorizer.transform([output, reference])
    sim = cosine_similarity(vectors[0], vectors[1])[0][0]
    return round(float(sim), 4)


# --- Drop-in replacement example (commented out, for you to try) ----------
# To swap in real embeddings via the Anthropic-compatible OpenAI-style API
# or Voyage AI, replace score_similarity's body with something like:
#
# import numpy as np
# def embed(text: str) -> np.ndarray:
#     resp = voyage_client.embed([text], model="voyage-3")
#     return np.array(resp.embeddings[0])
#
# def score_similarity(output: str, reference: str) -> float:
#     v1, v2 = embed(output), embed(reference)
#     return float(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2)))
