# Milestone 2 — Frozen MiniLM embeddings + LogReg (code only)

> Do **not** download models or fit classifiers unless the user asks.

**Goal:** Code path for frozen `all-MiniLM-L6-v2` embeddings + Logistic Regression, compared to M1 TF-IDF on the same splits.

**Architecture:** `src/embed.py` encodes text; `src/train_embeddings.py` fits linear head; notebook orchestrates side-by-side comparison.

**Tech Stack:** sentence-transformers, scikit-learn, joblib (reuse `src/data.py`, `src/evaluate.py`)

## Constraints

- Encoder frozen (no fine-tuning)
- Same label map / splits helpers as M1
- CPU-friendly path; GPU optional
- No fabricated metrics; code-only handoff
