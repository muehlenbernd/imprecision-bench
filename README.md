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
| `clock_description` | string | Textual clock description — finger-position format (added for LLM evaluation; not part of the original experiment) |
| `clock_description_minutemark` | string | Textual clock description — minute-mark format (added for LLM fine-tuning; see below) |
| `prompt` | string | Task 1 scenario text (unified across modalities) |
| `prompt_motive` | string | Task 2 context-embedded motive elicitation prompt |

### Motive label categories

`Precision`, `Accuracy`, `Info lack`, `Misinfo`, `Safe`, `H needs`, `Context`, `S ease`, `H ease`, `Habit`, `Sound`, `Other`

### Approximator values

`none`, `around`, `about`, `just before/after`, `approximately`, `ish`, `nearly`, `roughly`, `round about`

### Clock description formats

The dataset includes two textual representations of the clock stimulus, both stored as strings that can be prepended to the task prompt:

**Important:** the original human experiment used **clock images** as stimuli — participants never saw textual descriptions. Both `clock_description` columns were added to the dataset specifically for LLM evaluation, to provide text-based alternatives for models that struggle with analog clock reading.

**`clock_description`** — a finger-position format added for LLM text evaluation (e.g. *"Clock description: Hour hand between 8 and 9. Minute hand one minute-mark past the 5."*). This format describes the clock state qualitatively but does not state the exact minute explicitly, which can cause systematic misreadings in open-weight LLMs (14–48% accuracy in pilot studies).

**`clock_description_minutemark`** — a redesigned format that states the exact minute value explicitly (e.g. *"Clock description: Hour hand somewhere between 8 and 9. Minute hand at the 26-minute mark."*). This format was developed to eliminate comprehension failures: it achieves 100% accuracy across tested open-weight models (Llama 3.1 8B, Qwen 2.5 7B, Gemma 3 4B). The hour-hand description is deliberately uniform across all target times to avoid giving additional positional cues. Because this format makes the exact minute unambiguous, any rounding observed in model outputs is a pure pragmatic choice — rounding despite an explicit minute value is a stronger behavioural signal than rounding from an ambiguous description.

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

Each response is classified as **precise** (conveys the exact target time), **rounded** (conveys a canonical approximation — 8:25, 8:30, or 8:35 — but not the exact time), or **other** (wrong, vague, or non-answer). The same classifier is applied to human productions for a direct apples-to-apples comparison.

### The comprehension confound

Both the image task and the finger-position text task create systematic comprehension failures that confound pragmatic shift analysis. A model producing a wrong time ("other") is not making a pragmatic choice — it has simply misread the clock. The minute-mark results below should be read as the primary pragmatic test; the image and finger-position text results characterise how comprehension failures mask that capacity.

### Finding 1 — Modality gap

Models read textual clock descriptions far more accurately than clock images. Accuracy = fraction of responses where the exact target time is conveyed.

| Model | Image accuracy | Text accuracy |
|-------|:-:|:-:|
| GPT-4o mini | 2.7% | 35.8% |
| Claude Haiku 4.5 | 0.2% | 38.1% |
| Gemini 2.5 Flash | 26.9% | 54.3% |

GPT-4o mini and Claude Haiku essentially fail to read clock images (2.7% and 0.2%). Gemini reaches 26.9% — better but still far below its text performance. The finger-position text description improves comprehension but leaves high misreading rates (25–43% "other"). **Open challenge:** reliable analog-clock reading for current VLMs.

### Finding 2 — Pragmatic rounding capacity revealed by minute-mark descriptions

The minute-mark format eliminates comprehension ambiguity by stating the exact minute explicitly. With this input, two of three models show a clear, human-like pragmatic shift — more rounding in the casual neighbor context than in the formal police context.

| Source | Police precise | Neighbor precise | Police rnd | Neighbor rnd | WD |
|--------|:-:|:-:|:-:|:-:|:-:|
| GPT-4o mini (mm) | 100.0% | 100.0% | 0.0% | 0.0% | 0.000 |
| Claude Haiku 4.5 (mm) | 87.0% | 79.1% | 13.0% | 19.7% | 0.091 |
| Gemini 2.5 Flash (mm) | 76.2% | 62.7% | 23.8% | 37.3% | 0.135 |
| **Human baseline** | **62.3%** | **53.3%** | **32.9%** | **42.6%** | **0.097** |

WD = Wasserstein distance between police and neighbor distributions (3-way coding: 0=precise, 1=rounded, 2=other). Higher = more context-sensitive.

**Claude Haiku 4.5** shifts correctly (police more precise than neighbor), with a WD (0.091) matching the human baseline (0.097). **Gemini 2.5 Flash** shows the strongest shift: neighbor rounding (37.3%) substantially exceeds police rounding (23.8%), and WD (0.135) exceeds the human baseline. **GPT-4o mini** becomes over-literal — it reports the exact minute 100% of the time in both contexts, eliminating rounding entirely regardless of context.

**LLMs are demonstrably capable of context-sensitive pragmatic rounding.** The null and inverted results from the image and finger-position-text runs were artefacts of comprehension failure, not evidence of absent pragmatic sensitivity.

### Finding 3 — Pragmatic shift with finger-position text (confounded baseline)

For completeness, results with the finger-position `clock_description` format. These should be interpreted with caution given the high "other" rates. "Δ cond" = conditional shift (precise / (precise + rounded), excluding misreadings).

| Source | Police precise | Neighbor precise | Δ marg | Δ cond | WD |
|--------|:-:|:-:|:-:|:-:|:-:|
| GPT-4o mini | 32.5% | 38.9% | −6.5% | −5.3% | 0.120 |
| Claude Haiku 4.5 | 31.2% | 37.7% | −6.5% | −10.7% | 0.066 |
| Gemini 2.5 Flash | 55.0% | 53.7% | +1.3% | +4.7% | 0.044 |
| **Human baseline** | **62.3%** | **53.3%** | **+9.1%** | **+9.9%** | **0.097** |

GPT-4o mini and Claude Haiku show an inverted shift; Gemini moves in the correct direction but at less than half the human magnitude. In light of the minute-mark results (Finding 2), these patterns are best understood as artefacts of description comprehension failures.

### Finding 4 — Off-round subset: finger-position text and minute-mark

Restricting to unambiguous rounding targets (8:26–8:29, 8:31–8:34, n = 244) — the methodology of Mühlenbernd & Solt (2022). Δ cond = conditional precise shift (police − neighbor); p-value from one-tailed Mann-Whitney U (H₁: neighbor rounds more). The mm column uses `clock_description_minutemark`.

| Source | Police prec / rnd | Neighbor prec / rnd | Δ cond | p |
|--------|:-:|:-:|:-:|:-:|
| GPT-4o mini | 24.3% / 0.9% | 30.2% / 4.7% | +9.9% | 0.039 * |
| GPT-4o mini (mm) | 100.0% / 0.0% | 100.0% / 0.0% | 0.0% | — |
| Claude Haiku 4.5 | 21.7% / 0.0% | 26.4% / 0.0% | 0.0% | 1.000 |
| Claude Haiku 4.5 (mm) | 100.0% / 0.0% | 98.4% / 0.0% | 0.0% | 1.000 |
| Gemini 2.5 Flash | 51.3% / 0.0% | 50.4% / 8.5% | +14.5% | 0.001 *** |
| Gemini 2.5 Flash (mm) | 52.2% / 47.8% | 52.7% / 47.3% | −0.5% | 0.534 |
| **Human baseline** | **70.4% / 23.5%** | **51.2% / 43.4%** | **+20.9%** | **0.001 *****|

The off-round analysis qualifies Finding 2: the pragmatic shift visible in the full-dataset mm results does **not** come from context-sensitive rounding of off-round targets. Rather, it comes from approximator usage on canonical targets — models (especially Gemini and Claude) use hedges such as "around" or "half past" more often in the neighbor context when the target is a canonical time. This is context-sensitive precision calibration, but it is distinct from the off-round rounding behaviour measured in M&S 2022. Interestingly, Gemini does round off-round times heavily with mm (~48%), but context-insensitively. GPT-4o mini and Claude Haiku become maximally precise on off-round targets with mm, never rounding regardless of context.

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

- **Human experiment used clock images only.** Participants saw analog clock images as stimuli; there were no textual descriptions in the original experiment. Both `clock_description` columns were added specifically for LLM evaluation and are not part of the original experimental design.
- **Two description formats are provided.** `clock_description` uses a finger-position format; `clock_description_minutemark` states the exact minute explicitly. Results differ between formats and should not be compared directly across them.
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
