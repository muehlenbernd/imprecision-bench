---
license: cc-by-4.0
language:
- en
task_categories:
- text-generation
- visual-question-answering
pretty_name: imprecision-bench
tags:
- pragmatics
- computational-linguistics
- multimodal
- time-reference
- benchmarking
- LLM-evaluation
---

# imprecision-bench

[![License](https://img.shields.io/badge/license-CC--BY--4.0-green)](https://creativecommons.org/licenses/by/4.0/)
[![HuggingFace](https://img.shields.io/badge/🤗-Dataset-yellow)](https://huggingface.co/datasets/RolandM/imprecision-bench)
[![Paper](https://img.shields.io/badge/Paper-Linguistics%20Vanguard-blue)](https://doi.org/10.1515/lingvan-2022-0035)
[![Source Data](https://img.shields.io/badge/Source-figshare-lightgrey)](https://doi.org/10.6084/m9.figshare.21629531)

> **A multimodal benchmark for evaluating whether LLMs calibrate linguistic precision to pragmatic context, paired with 475 human productions and a peer-reviewed RSA baseline (r² ≈ 0.97).**

This dataset accompanies the paper:

**Modeling (Im)precision in Context**
Roland Mühlenbernd, Stephanie Solt
*Linguistics Vanguard, 2022*
[[Paper]](https://doi.org/10.1515/lingvan-2022-0035) · [[Source Data]](https://doi.org/10.6084/m9.figshare.21629531) · [[Companion Repo]](https://github.com/muehlenbernd/imprecision-in-context)

---

## Notebook

**`notebook.ipynb`** — guided walkthrough: data loading, sample evaluation (1 row per condition), clock-reading accuracy, pragmatic shift analysis, and Wasserstein distance against the human baseline.

| | |
|---|---|
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/muehlenbernd/imprecision-bench/blob/main/notebook.ipynb) | Interactive (Google account required) |
| [![Launch Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/muehlenbernd/imprecision-bench/main?filepath=notebook.ipynb) | Interactive (no account needed; slower start) |
| [![View on GitHub](https://img.shields.io/badge/view-on%20GitHub-lightgrey?logo=github)](https://github.com/muehlenbernd/imprecision-bench/blob/main/notebook.ipynb) | Read-only rendered view |

---

## Overview

When a witness tells a police officer "It happened at 8:31" rather than "It happened around 8:30," the choice of precision level is not arbitrary — it reflects the interlocutor's needs, the communicative context, and the speaker's pragmatic judgment. This benchmark tests whether large language models and vision-language models make the same kind of context-sensitive precision adjustments that humans do.

The dataset contains **475 human time-reference productions** across 24 conditions (12 clock states × 2 pragmatic contexts: police witness statement vs. neighbor at a party). Each item includes the clock image used as stimulus, a textual clock description for text-only LLM evaluation, annotated motive labels (12 categories, multi-label), and motive-explanation free text from a follow-up elicitation task.

The benchmark supports two evaluation tasks:
- **Task 1 — Production:** given a clock + scenario, produce a time expression to complete `"It happened ___."`
- **Task 2 — Motive elicitation:** given the production + context, explain why that wording was chosen.

---

## Dataset

| Property | Value |
|----------|-------|
| Size | 475 items, 312 unique speakers |
| Conditions | 12 clock states × 2 contexts (police / neighbor) |
| Language | English |
| Modalities | Clock images (PNG) + textual clock descriptions |
| Annotation | Multi-label motive categories (12 classes); free-text motive explanations |
| Source | Mühlenbernd & Solt (2022), figshare DOI 10.6084/m9.figshare.21629531 |
| License | CC BY 4.0 |

### Clock states

11 precise times (8:25–8:35, one minute apart) plus one range stimulus (8:26–8:34, shown as a yellow wedge on the clock face). The precise times span ±5 minutes around the canonical half-hour, allowing analysis of how offset magnitude interacts with rounding behavior.

### Pragmatic contexts

- **Police:** formal witness statement; officer establishing a detailed event timeline.
- **Neighbor:** casual party conversation; neighbor curious about what happened.

Human productions show a reliable cross-context shift: police context elicits more precise time expressions; neighbor context elicits more rounding and approximation.

---

## Quick Start

```python
from datasets import load_dataset

ds = load_dataset("RolandM/imprecision-bench", split="train")

# Inspect a row
row = ds[0]
print(row["prompt"])           # Task 1 scenario text
print(row["clock_description"])# Text clock stimulus
print(row["production"])       # Human reference answer
print(row["motive_labels"])    # Annotated motive categories
```

---

## Data Format

Each row represents one human production. Key columns:

| Column | Type | Description |
|--------|------|-------------|
| `item_id` | string | Unique item identifier |
| `subject_id` | int | Anonymized participant ID |
| `context` | ClassLabel | `police` or `neighbor` |
| `target_time` | string | Canonical time (e.g. `"8:30"`, `"8:26-8:34"`) |
| `stimulus_type` | ClassLabel | `precise` or `range` |
| `signed_offset` | int8 | Minutes offset from 8:30 (null for range) |
| `abs_offset` | int8 | Absolute offset in minutes (null for range) |
| `production` | string | Human time expression (fills `"It happened ___."`)|
| `production_code` | int8 | Precision code for the production |
| `approximator` | ClassLabel | Lexical approximator used (e.g. `around`, `about`, `none`) |
| `motive_labels` | Sequence(ClassLabel) | Multi-label motive annotation (12 categories) |
| `motive_text` | string | Free-text motive explanation from follow-up task |
| `clock_image` | Image | PNG clock image (432 × 429 px) |
| `clock_description` | string | Textual clock description (with `Clock description:` tag) |
| `prompt` | string | Task 1 scenario text (unified across modalities) |
| `prompt_motive` | string | Task 2 context-embedded motive elicitation prompt |

### Motive label categories

`Precision`, `Accuracy`, `Info lack`, `Misinfo`, `Safe`, `H needs`, `Context`, `S ease`, `H ease`, `Habit`, `Sound`, `Other`

### Approximator values

`none`, `around`, `about`, `just before/after`, `approximately`, `ish`, `nearly`, `roughly`, `round about`

---

## Evaluation

### Task 1 — Production (multimodal)

```python
import anthropic, base64
from datasets import load_dataset

ds = load_dataset("RolandM/imprecision-bench", split="train")
client = anthropic.Anthropic()

row = ds[0]
img_bytes = row["clock_image"].tobytes()  # PIL Image
img_b64 = base64.b64encode(img_bytes).decode()

response = client.messages.create(
    model="claude-opus-4-6",
    max_tokens=50,
    messages=[{
        "role": "user",
        "content": [
            {"type": "image",
             "source": {"type": "base64",
                        "media_type": "image/png",
                        "data": img_b64}},
            {"type": "text", "text": row["prompt"]},
        ],
    }],
)
print(response.content[0].text)
```

### Task 1 — Production (text-description)

```python
content = row["clock_description"] + "\n\n" + row["prompt"]
response = client.messages.create(
    model="claude-opus-4-6",
    max_tokens=50,
    messages=[{"role": "user", "content": content}],
)
print(response.content[0].text)
```

### Task 2 — Motive elicitation

```python
response = client.messages.create(
    model="claude-opus-4-6",
    max_tokens=200,
    messages=[{"role": "user", "content": row["prompt_motive"]}],
)
print(response.content[0].text)
```

### Suggested metrics

- **Production task:** distribution distance from human productions (Wasserstein distance over precision codes); cross-context shift detection (does the model produce more rounding in neighbor vs. police context?); approximator usage rate.
- **Motive task:** label match against `motive_labels` (multi-label F1); qualitative analysis of free-text explanations.
- **ESR / CDS:** effect-size ratio and calibration distance score, as defined in Mühlenbernd (2026, CMCL).

### Human baseline (RSA model)

A Rational Speech Act (RSA) speaker model fit to this dataset achieves **r² ≈ 0.97** against human production distributions, providing a strong peer-reviewed reference baseline. See the companion repository for the model implementation.

---

## Baseline Results

Results from three vision-language models evaluated on the full dataset (n = 475), using `evaluate.py`. All numbers are from Task 1 (production task); Task 2 (motive elicitation) is not yet scored and open for contribution.

### Finding 1 — Modality gap

Models read textual clock descriptions far more accurately than clock images. Accuracy here means the response correctly conveys the target time (exact digit or equivalent word form, e.g. *"thirty-one"* for 8:31); range-stimulus rows are excluded.

| Model | Image accuracy | Text accuracy |
|-------|:-:|:-:|
| GPT-4o mini | 3.6% | 19.9% |
| Claude Haiku 4.5 | 0.3% | 44.8% |
| Gemini 2.5 Flash | 18.3% | 39.1% |

Even the best image reader (Gemini, 18.3%) falls well short of its own text-description accuracy (39.1%). Claude Haiku essentially fails to read clock images (0.3%) yet reads textual descriptions at 44.8%. **Open challenge:** reliable analog-clock reading for current VLMs.

### Finding 2 — Pragmatic shift

Humans reliably use more precise time expressions in the police context than in the casual neighbor context (cross-context Wasserstein distance WD = 0.27). Models show a much weaker shift or no shift at all, measured as the rate of digital-format responses (*"8:31"*) per context.

| Model | Police | Neighbor | Δ | WD (image) | WD (text) |
|-------|:-:|:-:|:-:|:-:|:-:|
| GPT-4o mini | 0.4% | 4.5% | −4.1% | 0.04 | 0.01 |
| Claude Haiku 4.5 | 15.6% | 1.2% | +14.4% | 0.14 | 0.19 |
| Gemini 2.5 Flash | 30.7% | 21.7% | +9.0% | 0.09 | 0.12 |
| **Human baseline** | — | — | — | **0.27** | — |

GPT-4o mini inverts the expected direction (more digital in casual context). Claude Haiku and Gemini shift in the right direction but at roughly one-third to one-half the magnitude of the human effect. All model Wasserstein distances are well below the human baseline of 0.27. **Open challenge:** context-sensitive precision calibration at human magnitude.

---

## Prompts

### Task 1 — Police context, precise stimulus

```
[clock stimulus here — image or description]

One morning when you leave your house, you witness an automobile accident
in your street. You look at your watch when it happens. Later that day you
are invited to the police station to give a formal witness statement about
the accident. The police officer is trying to establish a detailed timeline
of the event. He asks you: "What time did the accident happen?" You remember
that it happened at the time shown on the clock as given above.

How would you answer in this situation? (Fill the blank)

"It happened ___."
```

For the **range stimulus**, "at the time" is replaced by "in the time range."
For the **neighbor context**, the police station passage is replaced by a party-at-a-neighbor's-house framing; "He asks" becomes "She asks."

### Task 2 — Police context, precise stimulus (example)

```
One morning when you leave your house, you witness an automobile accident
in your street. You look at your watch when it happens. Later, you gave a
formal witness statement at a police station. The officer, trying to
establish a detailed timeline, asked you what time the accident happened.
You knew that the accident happened at 8:31, and your answer was
"It happened just after half past eight". Why did you choose to answer
this way?
```

---

## Caveats and Limitations

- **Human baseline is image-only.** Participants saw clock images; they did not see textual descriptions. Comparisons between text-description LLM outputs and human productions are meaningful but involve a modality translation step that should be flagged in analyses.
- **Description format is a benchmark design choice.** One canonical textual description format is fixed as part of the benchmark specification. Results may differ with alternative description strategies.
- **Original stimulus wording.** The original M&S 2022 experiment used "the clock on the left" (referring to GUI layout). This dataset uses "the clock as given above," which is layout-agnostic but a minor deviation from the source wording.
- **Gendered interlocutors.** The original stimuli used "He asks" for the police officer and "She asks" for the neighbor. These gender assignments are preserved here as faithful to the source experiment. Researchers should be aware of potential gender-stereotyping effects.
- **Task 2 prompt is context-embedded.** The original follow-up task gave participants only the minimal prompt ("In your task, you knew that…"). This dataset's `prompt_motive` embeds the police/neighbor context so single-turn LLM evaluation has access to the pragmatic framing. This is a deliberate design choice for LLM eval; the source wording is preserved in the paper.
- **5 rows have empty `motive_text`.** Participants declined to respond on those items. `prompt_motive` is still valid; human-baseline comparisons for Task 2 simply have no ground-truth for those rows.
- **English only.** All productions are in English by English-speaking participants.

---

## Citation

If you use this benchmark, please cite the original paper:

```bibtex
@article{muehlenbernd2022imprecision,
  title   = {Modeling (im)precision in context},
  author  = {M{\"u}hlenbernd, Roland and Solt, Stephanie},
  journal = {Linguistics Vanguard},
  year    = {2022},
  doi     = {10.1515/lingvan-2022-0035}
}
```

Please also cite the source data:

```bibtex
@misc{muehlenbernd2022imprecision_data,
  title  = {Modeling (im)precision in context — supplementary data},
  author = {M{\"u}hlenbernd, Roland and Solt, Stephanie},
  year   = {2022},
  doi    = {10.6084/m9.figshare.21629531}
}
```

---

## License

This dataset is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), consistent with the source data license on figshare. You are free to share and adapt the material for any purpose, provided appropriate credit is given.

---

## Related Resources

- **Companion repo** (RSA model + analysis notebook): [muehlenbernd/imprecision-in-context](https://github.com/muehlenbernd/imprecision-in-context)
- **LLM social calibration paper** (ESR/CDS metrics): [muehlenbernd/llm-social-calibration](https://github.com/muehlenbernd/llm-social-calibration) · [arXiv 2604.02512](https://arxiv.org/abs/2604.02512)
