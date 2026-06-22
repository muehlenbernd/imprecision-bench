# CLAUDE.md — imprecision-bench Handover for Claude Code

This file is the primary handover document for Claude Code. Read it fully
before taking any action. It covers:
1. Project overview
2. Dual-repository setup (GitHub + HuggingFace)
3. First push checklist
4. Evaluation notebook setup
5. Open decisions to discuss before proceeding

---

## 1. Project Overview

`imprecision-bench` is a multimodal pragmatics benchmark derived from
Mühlenbernd & Solt (2022) "Modeling (im)precision in context"
(*Linguistics Vanguard*, DOI: 10.1515/lingvan-2022-0035).

**What it tests:** Whether LLMs and VLMs calibrate linguistic precision
to pragmatic context — specifically, whether a model adjusts how precisely
it reports a time (e.g. "8:30" vs. "around half past eight") depending on
whether the interlocutor is a police officer or a casual neighbor.

**Data:** 475 human productions × 24 conditions (12 clock states × 2
contexts: police / neighbor). Each row includes a clock image, a textual
clock description, rendered prompts for two evaluation tasks, and
multi-label motive annotations.

**Two evaluation tasks:**
- **Task 1 (production):** given a clock + scenario, complete "It happened ___."
- **Task 2 (motive elicitation):** given the production + context, explain
  why that wording was chosen.

**HuggingFace repo (data home):** `muehlenbernd/imprecision-bench`
**GitHub repo (code home):** `muehlenbernd/imprecision-bench` (to be created)

---

## 2. Repository Architecture

One local folder, two remotes. Raw data files stay local only.

```
imprecision-bench/
├── images/                 ← gitignored; clock PNGs live here locally
│   ├── clock825.png
│   ├── clock826.png
│   ├── ...
│   └── clocka830.png
├── data_clean.csv          ← gitignored; source data, stays local only
├── build_dataset.py        ← builds HF dataset from CSV + images
├── push.py                 ← builds (if needed) + pushes to HF Hub
├── evaluate.py             ← runs model evaluation; outputs results.csv
├── notebook.ipynb          ← analysis notebook (to be created — see §4)
├── results.csv             ← model outputs (committed to GitHub)
├── README.md               ← shared: HF dataset card + GitHub README
├── requirements.txt        ← Python dependencies
├── CLAUDE.md               ← this file
├── .env                    ← gitignored; credentials
├── .env.example            ← committed; credential template
└── .gitignore
```

**Key principle:** GitHub hosts the *code*; HuggingFace hosts the *data*.
`data_clean.csv` and `images/` are never committed to GitHub.

---

## 3. Dual-Repository Setup

### 3a. GitHub — create and initialise

1. Create the repo at https://github.com/new:
   - Name: `imprecision-bench`
   - Visibility: Public
   - Do NOT initialise with README (we have one already)

2. From the local project root:

```bash
git init
git add .
git commit -m "init: imprecision-bench dataset build and evaluation pipeline"
git remote add origin https://github.com/muehlenbernd/imprecision-bench.git
git push -u origin main
```

3. Verify `.gitignore` excludes these before committing:
   - `.env`
   - `data_clean.csv`
   - `images/`
   - `imprecision_bench_dataset/`
   - `__pycache__/`
   - `*.pyc`
   - `.DS_Store`

   The `.gitignore` already in the project should cover these. Double-check
   with `git status` before pushing.

### 3b. HuggingFace — push dataset and card

The HF repo already exists at `muehlenbernd/imprecision-bench`. To push
updates to the dataset or README:

```bash
# Push the dataset (runs build_dataset.py if needed, then push_to_hub)
python push.py

# Push only the README (dataset card update)
python - << 'EOF'
from huggingface_hub import HfApi
from dotenv import load_dotenv
import os
load_dotenv()
HfApi().upload_file(
    path_or_fileobj="README.md",
    path_in_repo="README.md",
    repo_id=os.environ["HF_REPO_ID"],
    repo_type="dataset",
)
print("README pushed.")
EOF
```

### 3c. .env.example

Make sure `.env.example` is committed to GitHub (safe — no real credentials):

```
HF_TOKEN=your_huggingface_write_token
HF_REPO_ID=muehlenbernd/imprecision-bench
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
```

---

## 4. Evaluation Notebook (`notebook.ipynb`)

### Purpose

A short, well-commented Jupyter notebook that:
1. Loads the benchmark from HF Hub
2. Runs model evaluation on a sample (1 row per condition = 24 rows)
3. Displays results clearly — what models do well, where they fail
4. Frames failures as open challenges, not just errors

### Framing principle

The notebook is not a paper. It is a **benchmark orientation guide** for
future users. The tone should be:

> "Here is what current models do on this benchmark. Here is where they
> succeed. Here is where they fail — and why that failure is interesting
> and worth solving."

Two headline findings to highlight (from the initial GPT-4o + Claude run):
- **Modality gap:** models are unreliable at reading clock images (GPT-4o:
  2/22 correct hour; Claude: 14/22) but accurate at reading textual clock
  descriptions (both ~22/22). Open challenge: better visual grounding for
  analog clock reading.
- **Pragmatic shift failure:** both models produce exact times regardless
  of context (police vs. neighbor). Humans round in casual contexts. Open
  challenge: context-sensitive precision calibration.

### Notebook structure

**Section 1 — Setup and data loading**
```python
# Load from HF Hub
from datasets import load_dataset
ds = load_dataset("muehlenbernd/imprecision-bench", split="train")
# Show condition breakdown, a sample row, the clock image
```

**Section 2 — Sampling**
```python
# Select 1 row per condition (24 rows = 12 times × 2 contexts)
# This is the same sampling strategy as evaluate.py
```

**Section 3 — Task 1 evaluation (production)**
```python
# Run Task 1 image and Task 1 text for the selected model
# Display: target_time | context | human | model_image | model_text
# Use evaluate.py's async infrastructure — import or call directly
```

**Section 4 — Results analysis**
```python
# Clock reading accuracy: image vs text
# Pragmatic shift: police vs neighbor exact-time rate
# Show side-by-side comparison table
```

**Section 5 — Open challenges**
```
Markdown cell discussing:
- Why image clock-reading is hard for current VLMs
- Why pragmatic precision calibration is a genuine open problem
- What "success" would look like on this benchmark
```

**Section 6 — Extending the evaluation**
```
Markdown cell covering:
- Running the full dataset (--full flag in evaluate.py)
- Adding new models (see §5 below)
- Task 2 (motive elicitation) — not yet scored; open for contribution
- Computing Wasserstein distance against human baseline
```

### Initial model

Start with **`gpt-4o-mini`** for the notebook run (faster and cheaper than
`gpt-4o` for a 24-row sample; results are indicative). The evaluate.py
script already supports this via `--model gpt-4o-mini`.

Results from the prior `gpt-4o` and `claude-opus-4-5` runs are already in
`results.csv` — load and display those in the notebook rather than re-running
them (saves API cost).

---

## 5. Model Selection — Discussion Required

**Do not decide this unilaterally.** Before adding new models to the
evaluation, discuss with Roland (via Claude AI or directly) which models
to include. The choice affects:
- API cost (full 475-row run × 3 calls per row)
- Benchmark narrative (which model families are represented)
- Reproducibility (model version pinning)

**Current runs already in results.csv:**
- `gpt-4o` (OpenAI)
- `claude-opus-4-5` (Anthropic)

**Candidate models for discussion:**
- `gpt-4o-mini` — cheaper GPT-4o; good for notebook baseline
- `claude-haiku-4-5` — cheaper Claude; tests whether smaller = worse pragmatics
- `gemini-1.5-flash` — completes the "big three" provider comparison; directly
  relevant given Roland's CMCL paper finding that Gemini exaggerates effect sizes
- `llava` or `idefics` — open-source VLM baseline for the image task
- Text-only models (GPT-3.5, Llama) — for text task only (no image column)

**Recommendation to raise with Roland:** Gemini is the most scientifically
motivated addition given the CMCL paper lineage. But confirm before adding.

---

## 6. First Push Checklist

Work through this in order. Check off each step before moving to the next.

- [ ] Verify `.gitignore` covers all raw data files and credentials
- [ ] `git status` — confirm no sensitive files are staged
- [ ] Create GitHub repo at https://github.com/new (Public, no init)
- [ ] `git init && git add . && git commit -m "init: ..."`
- [ ] `git remote add origin https://github.com/muehlenbernd/imprecision-bench.git`
- [ ] `git push -u origin main`
- [ ] Verify GitHub repo looks correct (README renders, no .env visible)
- [ ] `pip install -r requirements.txt`
- [ ] `python push.py` — confirm HF dataset push succeeds
- [ ] Smoke test: `from datasets import load_dataset; ds = load_dataset("muehlenbernd/imprecision-bench", split="train"); print(len(ds))`  → should be 475
- [ ] Create `notebook.ipynb` (see §4)
- [ ] Run notebook with `gpt-4o-mini` on 24-row sample; save outputs
- [ ] Commit notebook + results: `git add notebook.ipynb results.csv && git commit -m "add: evaluation notebook and initial results"`
- [ ] `git push origin main`
- [ ] Update README "Baseline Results" section with notebook findings
- [ ] Push updated README to HF: `python push.py` (or README-only push above)

---

## 7. Key Files Reference

| File | Purpose | Committed to GitHub? |
|------|---------|---------------------|
| `build_dataset.py` | Builds HF dataset from CSV + images | ✓ |
| `push.py` | Pushes dataset + README to HF Hub | ✓ |
| `evaluate.py` | Runs model eval; writes results.csv | ✓ |
| `notebook.ipynb` | Analysis notebook (to be created) | ✓ |
| `results.csv` | Model outputs from eval runs | ✓ |
| `README.md` | Dataset card + repo README | ✓ |
| `requirements.txt` | Python deps | ✓ |
| `.env.example` | Credential template | ✓ |
| `CLAUDE.md` | This file | ✓ |
| `.env` | Live credentials | ✗ (gitignored) |
| `data_clean.csv` | Source data | ✗ (gitignored) |
| `images/` | Clock PNGs | ✗ (gitignored) |
| `imprecision_bench_dataset/` | Built HF dataset (local cache) | ✗ (gitignored) |

---

## 8. Contact / Escalation

If a decision is needed that isn't covered here — especially around model
selection, benchmark framing, or README content — flag it and discuss with
Roland before proceeding. Do not make narrative or scientific framing
decisions unilaterally.
