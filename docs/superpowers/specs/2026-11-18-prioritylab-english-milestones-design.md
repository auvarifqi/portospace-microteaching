# PriorityLab — English Localization + Milestone Roadmap Design

**Date:** 2026-11-18  
**Status:** Approved for documentation update  
**Decisions locked:** Full English (option A); advanced method = frozen MiniLM embeddings + linear head (option 1)

## Problem

PriorityLab docs and the microteaching deck mix English and Indonesian. Milestone sequencing under-emphasizes project setup → data import → EDA as the immediate next step. A later “bigger than TF-IDF” method is desired but must not replace the classical fundamentals milestone.

## Goals

1. Make **all** project materials English: README, INIT (including mentor prompts), slides, speaker notes, and UI chrome.
2. Remove minute timings from slide pills; keep section titles only.
3. Rewrite next actions as milestones: **M0 Setup+Data+EDA → M1 TF-IDF → M2 sentence embeddings → M3 API**.
4. Add curated learning reference links for GitHub setup (including Medium), data import, EDA/CRISP-DM, TF-IDF, and embeddings.
5. Keep M2 scoped as frozen `all-MiniLM-L6-v2` + Logistic Regression comparison vs TF-IDF — not full transformer fine-tuning.

## Non-goals (this change)

- Do not create the EDA notebook yet (unless a later explicit request).
- Do not train models or implement FastAPI.
- Do not regenerate the PDF automatically (HTML is source of truth; re-export later).
- Do not commit unless the user asks.

## Milestone definitions

| Milestone | Deliverable | Gate |
|---|---|---|
| **M0** | Repo structure, GitHub push refs, env deps, load HF dataset, EDA notebook with class/duplicate audit | Binary label mapping feasible; both classes present |
| **M1** | Dummy + TF-IDF + LogReg pipeline, metrics, lexical explanations | Honest held-out metrics + error analysis |
| **M2** | Frozen MiniLM embeddings + LogReg; side-by-side vs TF-IDF | Same splits/API contract; document trade-offs |
| **M3** | FastAPI predict/health/model-info + tests + client demo | External process can call inference |

## Learning references (canonical list)

### Setup / GitHub
- https://medium.com/@ebayeeligbe/how-to-create-a-github-account-and-install-git-on-your-pc-a8b0150b659b
- https://mrmallik.medium.com/how-to-set-up-a-new-github-repository-with-branches-a-step-by-step-guide-a2ae79bc84e7
- https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github
- https://www.freecodecamp.org/news/host-your-first-project-on-github/

### Data import
- https://huggingface.co/docs/datasets/loading
- https://huggingface.co/docs/datasets/en/use_with_pandas
- https://huggingface.co/datasets/weijianzhg/email-triage-action-seed

### EDA / process
- https://www.ibm.com/docs/en/spss-modeler/saas?topic=dm-crisp-help-overview
- https://github.com/MichaelVerdegaal/crisp-dm-template

### M1 — TF-IDF
- https://investigate.ai/text-analysis/a-simple-explanation-of-tf-idf/
- https://rays-python-machine-learning.netlify.app/nlp_text_preprocessing
- https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html

### M2 — Sentence embeddings
- https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- https://devlifted.com/blog/mlp-on-frozen-sentence-embeddings
- https://latawarren.medium.com/static-vectors-vs-sentence-bert-a-b-testing-embeddings-fd7e28417bcf

## Files to update

1. `README.md` — milestones, M0 next actions, learning links, M2 preview
2. `INIT-PROJECT.md` — English mentor prompts; language rule; M0→M2 scope; refs
3. `MICROTEACHING-15MIN.html` — full English; remove minute pills; English control labels

## Constraints preserved

- Text-only, non-personalized importance proxy
- Synthetic HF dataset only for training labels
- CPU-first for M0/M1; M2 may download a small pretrained encoder but no fine-tuning / no GPU requirement for the educational path
- No fabricated metrics
- Notebook not created in this documentation pass
