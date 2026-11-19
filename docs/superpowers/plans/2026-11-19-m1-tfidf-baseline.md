# Milestone 1 — TF-IDF + Logistic Regression (code only)

> **For agentic workers:** Implement task-by-task. Do **not** fit models or download data unless the user asks.

**Goal:** Provide runnable training/eval/explain code for Dummy + TF-IDF+LogReg baselines.

**Architecture:** Thin `src/` helpers + one orchestration notebook. Fit TF-IDF only inside a sklearn `Pipeline` on the train fold.

**Tech Stack:** pandas, datasets, scikit-learn, joblib, matplotlib

## Global Constraints

- subject+body only; no category/action/source_round in features
- binary map urgent/high→1, medium/low→0; fail on unknown
- no fabricated metrics; no execution in this pass
- CPU-only classical ML

---

### Task 1: `src/` modules

- [ ] `src/data.py` — load, clean, map labels, stratified split
- [ ] `src/train.py` — Dummy + TF-IDF/LogReg pipelines, save joblib
- [ ] `src/evaluate.py` — metrics + confusion matrix plot
- [ ] `src/explain.py` — top coefficient terms (illustrative API)
- [ ] `src/__init__.py`

### Task 2: Notebook + docs

- [ ] `notebooks/02_baseline_tfidf.ipynb` — call src helpers, no precomputed scores
- [ ] Update README/INIT M1 checklist to “code ready, not run”
