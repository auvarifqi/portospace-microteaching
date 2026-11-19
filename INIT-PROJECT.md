# INIT-PROJECT.md — PriorityLab

> **Kickoff prompt + initial project scope (v1.1, 18 November 2026)**  
> **Name:** PriorityLab — Explainable Email Importance Classifier  
> **Audience:** grade-10 PortoSpace STEM mentees; beginner-friendly teaching with professional ML Engineering practices  
> **Method:** CRISP-DM (six iterative phases)  
> **Language:** English (students, mentor scripts, docs, slides)  
> **Status:** documentation and teaching deck only. No notebook, training, evaluation, model, or API has been implemented.  
> **Next milestone:** **M0 — project setup → GitHub → import data → EDA**

## 0. MASTER PROMPT — paste this into an AI coding assistant

You are a **senior Data Scientist, pragmatic ML Engineer, and clear STEM mentor**. Collaborate with me on **PriorityLab**, a transparent, CPU-friendly text classifier that flags whether an email deserves attention. The portfolio question is: **"How does Gmail flag important emails, and can we build a simple, explainable content-based proxy and serve it through a REST API?"**

### Non-negotiable constraints

1. Follow **CRISP-DM** in order: Business Understanding → Data Understanding → Data Preparation → Modeling → Evaluation → Deployment. Iterate backward when findings invalidate assumptions. Explain the *why* before the code.
2. Build **a global / non-personalized model** using only `subject` and `body`. **Do not** implement, plan, train, benchmark, or imply personalized importance ranking. No user IDs, user histories, behavior tracking, feedback learning, sender relationships or user-specific weights. **This is deliberately excluded because the chosen dataset does not contain those signals.**
3. Do not claim to reproduce Gmail's proprietary algorithm or match Gmail performance. Google's public description is only problem context. Gmail's behavior uses signals not available here.
4. Use `weijianzhg/email-triage-action-seed` (Hugging Face) as the **synthetic English-language seed corpus**. Card: ~3,905 rows, Apache-2.0, one original `train` split. **Verify live schema and actual observed priority values**; the dataset card lists `urgent/high/medium/low` but the viewer currently reports only three represented priority classes.
5. **Binary target** for v1: `important = 1` if `priority ∈ {urgent, high}`, else `0` if `priority ∈ {medium, low}`. This conversion is a **project rubric**, not a native Gmail label or objectively true human importance. Reject unknown labels and fail loudly if either class is absent; document prevalence.
6. Only `subject` and `body` may become model features. `category`, `action`, `source_round` are for **audit/EDA only**; never train using them or claim to predict them. Avoid leakage from generated templates, duplicate emails, label strings and global vectorizer fitting.
7. **Modeling ladder (do not skip fundamentals):**
   - **M1 (required first):** `TfidfVectorizer` + `LogisticRegression`, with `DummyClassifier` baseline and optional `MultinomialNB`.
   - **M2 (later upgrade):** frozen sentence embeddings from `sentence-transformers/all-MiniLM-L6-v2` + `LogisticRegression`, compared on the **same splits** as M1. Do **not** fine-tune the transformer for the educational path. No paid API, no Gmail inbox integration. GPU is optional convenience, not a requirement.
8. **Do not create `notebooks/01_eda_preprocessing.ipynb` until M0 implementation starts.** Document its plan and sample cells here; create it only when executing M0.
9. Don't invent accuracy, F1, latency, importance probability, feature weights, or successful execution. Every simulated/demo value must be labeled *illustrative* until actually measured.
10. Deliver the full ML system eventually: repeatable training code, saved artifacts, explainability inspection, FastAPI `POST /v1/predict`, `/health`, `/v1/model-info`, Pydantic validation, tests, and an example external client. Never auto-reply, archive, forward or change a Gmail marker.
11. Use the **Google-inspired color palette** blue `#4285F4`, red `#EA4335`, yellow `#FBBC04`, green `#34A853` in EDA, diagrams and teaching material, with accessible contrast. Do not impersonate Google branding.
12. Keep everything portfolio-ready: explain assumptions and trade-offs in **plain English** for students, preserve technical terms, write clean documented Python and reproducible instructions in Markdown.

**Working style for future phases:** Read `INIT-PROJECT.md` and `README.md`; choose the earliest incomplete milestone; show a small execution plan; implement only that milestone; run tests; report observed evidence, risks and the next task. Keep notebooks for exploration and `src/` code for reusable logic. Never pass off a mock as a completed service.

---

## 1. CRISP-DM Phase 1 — Business Understanding

### 1.1 Product question and outcome

**Problem:** An inbox contains time-sensitive requests, routine notifications and promotions. Can an inexpensive classifier **flag likely important emails from text** so another application can sort or highlight them?

**What the user sees:** email subject/body → `important` or `not_important` with transparent evidence. **No per-user relevance** is inferred.

**Research questions:**
- *M1:* Can a TF-IDF + Logistic Regression baseline approximate synthetic email priority labels and generalize to deliberately tricky hold-out emails?
- *M2:* Do frozen MiniLM sentence embeddings improve that baseline on the same evaluation protocol?

### 1.2 Distinguish the concepts

| Concept | Definition | Included? |
|---|---|---|
| `priority` | Original synthetic label `urgent/high/medium/low` | Source label for mapping |
| `important` | Binary derived proxy: `urgent/high` vs `medium/low` | **Yes — target** |
| `category` | Subject domain, e.g. finance, newsletter | **EDA only** |
| `action` | Suggested task such as reply or archive | **EDA only** |
| Spam/phishing safety | Security or unsolicited-message detection | **No** |
| Personalized importance | User-specific preference prediction | **NO, entirely out of scope** |

**Label mapping specification**

```python
PRIORITY_TO_IMPORTANT = {
    "urgent": 1,
    "high": 1,
    "medium": 0,
    "low": 0,
}
```

Handle whitespace/case variations explicitly. **Do not silently map unknown or missing labels to zero**. Because the dataset was generated to satisfy label prompts, binary labels are **synthetic proxy labels**, not judgments by human recipients.

### 1.3 Success definition (acceptance criteria, NOT results)

- Model is trained and evaluated **locally on CPU** (GPU optional for M2 embedding speed only), with fixed seed, artifact metadata, and all dependencies documented.
- **M1:** at least two baselines evaluated: **DummyClassifier** and **TF-IDF + Logistic Regression**; report positive-class precision, recall, F1, overall accuracy, macro F1, support and confusion matrix, as well as 5+ examined mistakes.
- **M2:** frozen MiniLM embeddings + Logistic Regression compared against M1 on identical splits; document gains, losses, latency, and explainability trade-offs.
- Explain why **false negatives** (important email missed) may be more harmful than false positives (routine email flagged). No unverified numerical performance promises.
- Include a separate **hand-reviewed fictional challenge set** (e.g. 20–40 examples) designed before inspecting final model errors. No real private emails.
- A second Python process or curl client can call a locally running REST service and receive a valid, schema-documented result without running the training notebook.
- Portfolio clearly notes that synthetic test scores **do not validate production suitability or Gmail equivalence**.

### 1.4 Hard exclusions

No personalized model, user-specific ranking, user event collection or behavior history; no Gmail API permissions or reading inboxes; no automatic email actions; no trained action classifier; no cloud/paid APIs; no required GPU; no full transformer fine-tuning in M2; no web dashboard required for MVP; no real-time production deployment claims.

---

## 2. CRISP-DM Phase 2 — Data Understanding

**Dataset:** https://huggingface.co/datasets/weijianzhg/email-triage-action-seed  
**Published information:** 3,905 synthetic English emails, Apache-2.0 license, one original `train` split. The schema includes `subject`, `body`, `category`, `priority`, `action`, and `source_round`. These are card-level facts, **not proof of data quality**.

**Required EDA checklist:**

- [ ] Confirm number of rows, dtypes, allowed/observed priority values, and rows per binary target.
- [ ] Audit empty/missing `subject`, `body`, duplicated text and contradictory duplicate labels.
- [ ] Visualize actual distributions of `priority`, `important`, `category` and `action` (only where available).
- [ ] Inspect `priority × action` and `priority × category` tables for synthetic label shortcuts.
- [ ] Examine subject/body length, top terms/bi-grams and presence of templated phrases.
- [ ] Inspect `source_round` as provenance and check whether it creates distribution artifacts.
- [ ] Manually review 5–10 cases per label and document ambiguous examples.
- [ ] Verify whether the intended binary task is meaningful enough to continue; return to Phase 1 if not.

### Notebook specification — planned for M0

**Path to be created in M0:** `notebooks/01_eda_preprocessing.ipynb`

**Proposed notebook sections:**

1. Purpose / hypothesis and data dictionary.
2. Import dependencies and load dataset.
3. Verify schema, counts, missing data, license and represented class set.
4. Map `priority` → `important` and visualize original + derived labels.
5. Explore text lengths, frequent terms, class proportions and apparent shortcuts.
6. Check exact/near duplicates and possible prompt-template effects.
7. Propose deduplicated, reproducible train/validation/test split with holdout discipline.
8. Export *split manifest and summary charts*, not trained weights.
9. Write limitations and go/no-go recommendation for modeling.

**Example cells (spec examples only):**

```python
# Cell A — Import the open dataset
from datasets import load_dataset
import pandas as pd

raw = load_dataset("weijianzhg/email-triage-action-seed", split="train")
df = raw.to_pandas()
required = {"subject", "body", "priority", "category", "action", "source_round"}
assert required.issubset(df.columns), f"Missing: {required - set(df.columns)}"
print(df.shape)
print(df["priority"].value_counts(dropna=False))  # inspect BEFORE assuming classes
```

```python
# Cell B — Validate the modeling target; fail on unknown labels
priority_map = {"urgent": 1, "high": 1, "medium": 0, "low": 0}
df["priority_norm"] = df["priority"].astype("string").str.strip().str.lower()
unknown = set(df["priority_norm"].dropna().unique()) - set(priority_map)
assert not unknown and df["priority_norm"].notna().all(), f"Bad priority labels: {unknown}"
df["important"] = df["priority_norm"].map(priority_map).astype(int)
assert df["important"].nunique() == 2, "Both binary classes must be present"
print(df["important"].value_counts())
```

```python
# Cell C — Google-inspired EDA colors (illustration)
import matplotlib.pyplot as plt
colors = {1: "#4285F4", 0: "#34A853"}
counts = df["important"].value_counts().reindex([1, 0], fill_value=0)
ax = counts.plot.bar(color=[colors[i] for i in counts.index], rot=0)
ax.set_xticklabels(["Important", "Not important"])
ax.set(title="Email labels (derived from synthetic priority)", ylabel="Number of emails")
plt.tight_layout()
plt.show()
```

```python
# Cell D — Input feature engineering only (no label leakage)
df["subject_clean"] = df["subject"].fillna("").astype(str).str.strip()
df["body_clean"] = df["body"].fillna("").astype(str).str.strip()
df["text"] = "subject: " + df["subject_clean"] + " body: " + df["body_clean"]
# The model must NOT see 'priority', 'action', 'category', or 'source_round' in its text.
# Deduplicate and finalize splits BEFORE fitting a vectorizer or encoder.
```

### Dataset risks to discuss with mentee

The source corpus was generated by a single LLM, is English-only and oriented toward professional email scenarios. Its priority labels were part of generation prompts. A random split may contain near-identical templates in both train and test. Therefore: inspect overlap, preferably group similar text, keep a separate manually labeled challenge set and **avoid claiming generalization to personal inboxes**.

---

## 3. CRISP-DM Phase 3 — Data Preparation

**Suggested pipeline:** raw dataset → validate fields → binary mapping → normalize subject/body → hash/dedup → investigate near duplicates → choose split strategy → fit text pipeline **on training fold only** → retain test fold untouched.

**Deliverables (later):** data loader, validation helpers, split manifests, train/validation/test distribution chart, data audit report. Default 70/15/15 stratified split is only a starting point **if** class counts permit; group-aware splitting should take precedence where many similar generated examples exist. Record exact seed and dataset revision/commit when possible.

**Leakage prevention:** `priority`, `important`, `action`, `category`, `source_round` are forbidden input columns. Do not fit TF-IDF (or compute IDF/embedding stats that leak labels) on all emails before splitting. Examine near-duplicate phrases originating from one prompt template. Never tune the model on the final test/challenge set.

---

## 4. CRISP-DM Phase 4 — Modeling

### Milestone 1 — Classical baseline (fundamentals first)

1. `DummyClassifier(strategy="most_frequent")` tells us what chance-free classification looks like.
2. Optional `TfidfVectorizer + MultinomialNB` as a simple comparator.
3. **Primary:** `TfidfVectorizer(ngram_range=(1, 2)) + LogisticRegression(class_weight="balanced")`, comparing class weights as an explicit experiment, not as a default guarantee.

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=10_000)),
    ("classifier", LogisticRegression(max_iter=2_000, random_state=42)),
])
# model.fit(X_train, y_train)  # only after EDA, dedup and split audit
```

### Milestone 2 — Frozen sentence embeddings (later upgrade)

Use a pretrained encoder as a **frozen feature extractor**, then train only a linear head:

```python
# Sketch only — implement in M2, after M1 is evaluated
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
X_train = encoder.encode(train_texts, show_progress_bar=True)  # frozen
X_test = encoder.encode(test_texts, show_progress_bar=True)

clf = LogisticRegression(max_iter=2_000, random_state=42)
# clf.fit(X_train, y_train)  # only after M1 baseline exists for comparison
```

**Rules for M2:** do not fine-tune MiniLM; cache embeddings; compare against TF-IDF on the same splits and challenge set; document that coefficient-style word explanations are weaker than M1.

### Fundamental lesson for grade 10: text → numbers → learned decision

**Bag of Words first:** Let vocabulary = `[deadline, project, sale]`. An email "Project deadline" becomes `[1, 1, 0]`; "Big sale" becomes `[0, 0, 1]`. Explain the key insight: *a computer needs a numerical representation of words before a classical ML classifier can work*.

**Then TF-IDF:** common words carry less discriminative weight, while distinctive words or phrases may carry more. **Then Logistic Regression:** learns weights for text features from labeled examples.

**Later (M2 intuition only in early sessions):** a sentence embedding is a dense vector that captures meaning similarity — "submit report by tomorrow" can be close to "deadline for lab report" even with different words.

**Explainable AI:** For M1, display top terms from logistic-regression coefficients and per-message contributions (`tfidf_value × coefficient`). **Never fabricate feature importance from an untrained model.**

**Target emphasis for the 15-minute microteaching:** Teach *one* concept deeply: **"How can an algorithm read an email if it's just text?"** Mention M2 as a future upgrade, not a second full lesson.

---

## 5. CRISP-DM Phase 5 — Evaluation

**Minimum:** confusion matrix; accuracy; precision/recall/F1 for `important=1`; macro F1; class support and baseline comparison. Plot all metrics using Google-inspired colors and label them clearly with *observed values only*.

**Failure-case challenge set (fictional, manually reviewed):**
- "URGENT: clearance sale ends tonight" — urgent word but potentially promotional.
- "Please submit the chemistry assignment by 8 AM" — real deadline without the token *urgent*.
- Routine newsletter with intimidating subject language.
- Financial/system-risk emails with understated subject lines.
- Very short/blank subject but meaningful body.
- Non-English examples — explicitly out-of-distribution for the English dataset.

**Discuss false negatives vs. false positives.** If using `predict_proba`, do not label it calibrated confidence unless probability calibration has been checked. Challenge-set quality and transparent error analysis matter more than a single high metric on synthetic data.

**Evaluation gate:** if the model performs only on templated synthetic text or collapses to one class, return to data understanding/preparation; do not proceed to an impressive-sounding API as if it were reliable.

---

## 6. CRISP-DM Phase 6 — Deployment (ML Engineering)

**Goal:** another service should consume a prediction over HTTP **without knowing or re-running the training notebook**.

```text
       Training workflow (offline)              Inference workflow (online)
  Hugging Face synthetic dataset                     Any client
              |                                          |
      EDA / preprocessing                         POST JSON over HTTP
              |                                          |
    Train/validation/evaluate                         FastAPI
              |                                  validate subject/body
      Save fitted artifacts                               |
              |------------------------------>     Load fitted artifacts
                                                        |
                                                  Predict binary label
                                                        |
                                            Explain / evidence fields
                                                        |
                                             Return versioned JSON
```

**REST API draft — future implementation, not live:**

```http
POST /v1/predict
Content-Type: application/json
```

```json
{
  "subject": "Report due tomorrow",
  "body": "Please submit the lab report by 08:00."
}
```

```json
{
  "label": "important",
  "is_important": true,
  "top_terms": ["report", "submit"],
  "model_version": "baseline-v1"
}
```

**The response is schema illustration only; not a model run.** `top_terms` must be computed from the fitted M1 pipeline (and may be empty for M2). Never claim it gives the user's true motivation or causal reason. No user ID, user personalization, or Gmail API actions.

**Endpoints:** `GET /health` for readiness; `GET /v1/model-info` for label mapping and version; `POST /v1/predict` for inference. Use a typed Pydantic request with nonempty subject-or-body, length limits, defined HTTP 422 validation, and consistent error responses. Load artifacts from trusted local sources only; don't unpickle user-supplied models. Keep email bodies out of logs; when externally deployed, require basic authentication/rate limiting/HTTPS.

**Engineering handoff:** training code in `src/` → `models/` saved artifacts with version → thin FastAPI adapter in `app/` → pytest coverage in `tests/` → example `curl` and Python `requests` consumer → optional CPU-only Dockerfile → local p50/p95 latency measured, not guessed.

---

## 7. Repository structure and delivery boundaries

```text
PriorityLab/
├── INIT-PROJECT.md                 # This kickoff spec + mentor prompt
├── README.md                       # Public-facing overview
├── MICROTEACHING-15MIN.html        # Browser/print slide source
├── MICROTEACHING-15MIN.pdf         # Exported 16:9 slides
├── docs/superpowers/specs/         # Design decisions
├── notebooks/                      # PLANNED — create in M0
│   └── 01_eda_preprocessing.ipynb
├── src/                            # PLANNED — DS reusable modules
│   ├── data.py
│   ├── train.py
│   ├── evaluate.py
│   └── explain.py
├── app/                            # PLANNED — ML service
│   ├── main.py
│   └── schemas.py
├── tests/                          # PLANNED
├── models/                         # PLANNED — local trained artifacts
└── reports/                        # PLANNED — model card and EDA figures
```

**Do not create placeholder folders just to simulate completion.** Create the EDA notebook when executing **M0**.

---

## 8. PortoSpace microteaching: talk track for 15 minutes

**Scenario:** first 1-on-1 mentorship session **after brainstorming**. The grade-10 learner has already chosen the email project. Do not spend 15 minutes pitching slides; **teach a student** with 3 clear checks and one independent task.

| Block | What you say/show | Student must do |
|---|---|---|
| **Hook** | "We agreed to make an AI that highlights important emails. Which message needs attention?" | Choose an email and give a reason |
| **Lifecycle** | CRISP-DM: question → data → prep → model → evaluate → deploy | See where the algorithm learns vs. where apps use it |
| **Roadmap** | Milestones M0→M3 (setup/EDA → TF-IDF → embeddings → API) | Know we start with setup, not training |
| **Domain** | Gmail public signals; our dataset has only subject/body/priority | Notice a signal we cannot model |
| **BoW** | Vocabulary `deadline / project / sale` → count vectors | Convert one short message to `[count, count, count]` |
| **TF-IDF** | BoW counts → TF-IDF weights → Logistic Regression learns labels | State the role of each step |
| **Challenge** | Why `URGENT SALE` can fool keyword-only rules | Explain the ambiguity |
| **Wrap-up** | M0 setup guide: GitHub, env, import data, EDA | Agree on next build steps + README learning links |

### Mentor prompts and fallback explanations

**Opening:** "We already chose the inbox-assistant project yesterday. If you had 100 emails, which ones would you read first?"  
**Lifecycle checkpoint:** "When does the model *learn*, and when is the model *used*?"  
**Roadmap checkpoint:** "What do we do before training any model?" (Setup + import + EDA.)  
**Domain checkpoint:** "If we only give the AI the text, does it know this email is from a close friend?" (No; sender relationship is not a feature.)  
**Core teaching:** "Computers cannot read words directly. Bag of Words turns text into counts. TF-IDF reweights those counts so common words matter less. Logistic Regression then learns from labels."  
**Student exercise:** `"Project deadline" → [1,1,0]` for `[deadline,project,sale]` (order as stated).  
**Challenge:** "If the subject is 'URGENT SALE', is it always important? Why or why not?"  
**Misconception response:** If they confuse TF-IDF with choosing importance, explain that TF-IDF encodes text; **the classifier** learns the mapping from features to labels.  
**Future peek (optional, short):** "Later (M2) we can try a bigger pretrained model that turns a whole sentence into a meaning vector — after the TF-IDF baseline."

**Next step — Milestone 0 project setup (no fictional-email homework):**  
1. Create/push a GitHub repository (Medium + GitHub Docs links in §10).  
2. Create a Python venv and install core packages (`pandas`, `datasets`, `matplotlib`, `scikit-learn`, `jupyter`).  
3. Import `weijianzhg/email-triage-action-seed` and convert to pandas.  
4. Start the EDA notebook: schema, priority distribution, binary mapping, missingness, duplicates, charts.

---

## 9. Action items with Definition of Done

| Priority | Item | Deliverable | Acceptance evidence |
|---|---|---|---|
| **P0** | Confirm non-personalized scope and binary label mapping | `INIT-PROJECT.md`, `README.md` | No user-specific features or training goals |
| **P0** | Make teaching deck and rehearse 15-minute flow | HTML (+ PDF re-export) | Slides include lifecycle → milestones → BoW → TF-IDF → setup guide; English; no minute pills; PDF regenerated after HTML |
| **M0** | Project setup + GitHub remote | repo layout, env, remote URL | Local project linked to GitHub following learning refs |
| **M0** | Import dataset + EDA notebook | `01_eda_preprocessing.ipynb` | Actual class counts, missingness, duplicates and risks shown |
| **M0** | Formalize rubric & challenge examples | rubric + 20–40 fake labeled emails | Reviewed independently of model training |
| **M0** | Split and leakage audit | split manifest and checks | Unseen held-out examples; features fit train-only |
| **M1** | TF-IDF baselines + evaluation | trained pipeline, `metrics.json`, confusion matrix | Dummy vs LogReg compared, false negatives analyzed |
| **M1** | Explainable AI review | top-feature chart + 5 case analyses | Model evidence vs human explanation distinguished |
| **M2** | Frozen MiniLM + LogReg | embedding artifacts + comparison report | Same splits as M1; encoder not fine-tuned |
| **M3** | Model packaging + serving | joblib/artifacts + FastAPI + tests | curl/pytest demonstrate inference from another process |
| **M3** | Optional Docker, latency and model card | Dockerfile + benchmarks + limitations | CPU startup, p50/p95 measured, no unsupported claims |

### Immediate next action after this handoff

**Start M0:** project setup (with GitHub references) → import the Hugging Face dataset → run EDA. Do **not** jump directly to fitting Logistic Regression or downloading MiniLM. If the data cannot support our binary mapping, revise the task definition in Phase 1 and document why.

### Scope-completion ledger (as of 18 Nov 2026)

- [x] Problem, objectives, exclusions, data source, label mapping and CRISP-DM stages specified.
- [x] English teaching deck without minute timings; M0→M2 roadmap and learning links documented.
- [x] Starter notebook cells documented; notebook **not created**.
- [x] CPU modeling, evaluation, explainability, FastAPI, and M2 embedding upgrade plans specified.
- [ ] EDA executed, model trained, API implemented, or latency measured — **none performed yet**.

---

## 10. Learning references

### Project setup & GitHub
1. [Medium — Create a GitHub account and install Git](https://medium.com/@ebayeeligbe/how-to-create-a-github-account-and-install-git-on-your-pc-a8b0150b659b)
2. [Medium — Set up a new GitHub repository with branches](https://mrmallik.medium.com/how-to-set-up-a-new-github-repository-with-branches-a-step-by-step-guide-a2ae79bc84e7)
3. [GitHub Docs — Add locally hosted code to GitHub](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github)
4. [freeCodeCamp — Host your project on GitHub](https://www.freecodecamp.org/news/host-your-first-project-on-github/)

### Import data
5. [Hugging Face — Load a dataset](https://huggingface.co/docs/datasets/loading)
6. [Hugging Face — Use datasets with Pandas](https://huggingface.co/docs/datasets/en/use_with_pandas)
7. [Dataset — Email Triage Action Seed](https://huggingface.co/datasets/weijianzhg/email-triage-action-seed)

### EDA & CRISP-DM
8. [IBM — CRISP-DM overview](https://www.ibm.com/docs/en/spss-modeler/saas?topic=dm-crisp-help-overview)
9. [GitHub — CRISP-DM project template](https://github.com/MichaelVerdegaal/crisp-dm-template)

### Milestone 1 — TF-IDF + Logistic Regression
10. [Simple explanation of TF-IDF](https://investigate.ai/text-analysis/a-simple-explanation-of-tf-idf/)
11. [TF-IDF + Logistic Regression text classification](https://rays-python-machine-learning.netlify.app/nlp_text_preprocessing)
12. [scikit-learn — Document classification with sparse features](https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html)

### Milestone 2 — Frozen sentence embeddings
13. [Model card — all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
14. [Frozen sentence embeddings + classifier head](https://devlifted.com/blog/mlp-on-frozen-sentence-embeddings)
15. [Medium — Static vectors vs Sentence-BERT](https://latawarren.medium.com/static-vectors-vs-sentence-bert-a-b-testing-embeddings-fd7e28417bcf)

### Domain context
16. [Google Gmail Help — Importance markers](https://support.google.com/mail/answer/186543?hl=en)

*Independent educational work, not affiliated with or endorsed by Google or Gmail. Google palette is a style reference, not a claim of product affiliation.*
