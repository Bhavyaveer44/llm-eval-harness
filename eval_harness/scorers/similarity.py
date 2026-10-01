"""
similarity.py
-------------
Concept: SEMANTIC SIMILARITY SCORING

Why not just check output == reference? Because LLM output is never exactly
reproducible — different wording, different order, same meaning. So instead
we turn text into vectors and measure the angle between them (cosine
similarity). 1.0 = identical direction (very similar meaning), 0 = unrelated.

We use sentence-transformers (BAAI/bge-small-en-v1.5) for production-grade semantic
similarity. This model is:
- Fast and lightweight (runs locally, no API calls)
- Trained on semantic similarity (catches meaning, not just word overlap)
- Better separation between similar and dissimilar text than MiniLM
- Free to use (no API costs)

This is a significant upgrade over TF-IDF, which only catches word overlap.
"""
from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer


_model = None


def _get_model():
    """Lazy-load the embedding model to avoid import-time overhead."""
    global _model
    if _model is None:
        _model = SentenceTransformer('BAAI/bge-small-en-v1.5')
    return _model


def score_similarity(output: str, reference: str) -> float:
    """
    Returns a similarity score in [0, 1] between generated output and a
    reference answer using semantic embeddings. Returns 0.0 for degenerate
    cases (empty strings) so a crashed/empty app output always fails rather
    than raising.
    """
    if not output.strip() or not reference.strip():
        return 0.0

    model = _get_model()
    embeddings = model.encode([output, reference])
    v1, v2 = embeddings[0], embeddings[1]
    sim = float(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    return round(sim, 4)
