"""
evaluate.py — Evaluate a VLM on imprecision-bench (Tasks 1 & 2).

Per row, three calls are made concurrently:
  Task 1 image : clock image (base64 PNG) + scenario prompt → time expression
  Task 1 text  : clock_description + scenario prompt        → time expression
  Task 2       : prompt_motive                              → motive explanation

Results land in results.csv with model-name-prefixed columns, e.g.:
  gpt_4o_task1_image | gpt_4o_task1_text | gpt_4o_task2
  claude_opus_4_5_task1_image | ...

All API calls within a run are made concurrently (asyncio), so total runtime
is roughly the latency of the single slowest call rather than the sum.

Supported models (pass to --model):
  OpenAI      : gpt-4o (default), gpt-4o-mini
  Anthropic   : claude-opus-4-5, claude-haiku-4-5  (or any claude-* vision model)
  OpenRouter  : google/gemini-2.5-flash
                (any provider/model slug supported by openrouter.ai)
  Together.ai : together/meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo
                (prefix any Together.ai model slug with "together/")

Usage:
    python -m pip install -r requirements.txt
    # Set API keys in .env (see .env.example), then:
    python evaluate.py                                                          # GPT-4o, 24-row sample
    python evaluate.py --model claude-haiku-4-5                                 # Claude Haiku
    python evaluate.py --model google/gemini-2.5-flash                          # Gemini via OpenRouter
    python evaluate.py --model together/meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo  # LLaMA via Together
    python evaluate.py --rows 50                                                # first 50 rows
    python evaluate.py --full                                                   # all 475 rows
    python evaluate.py --out PATH                                               # custom output CSV
"""

import argparse
import asyncio
import io
import base64
import os
import re
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from datasets import load_dataset
from scipy.stats import wasserstein_distance

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

load_dotenv()

HF_REPO_ID          = os.environ.get("HF_REPO_ID", "RolandM/imprecision-bench")
OPENAI_API_KEY      = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY   = os.environ.get("ANTHROPIC_API_KEY", "")
OPENROUTER_API_KEY  = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
TOGETHER_API_KEY    = os.environ.get("TOGETHER_API_KEY", "")
TOGETHER_BASE_URL   = "https://api.together.xyz/v1"

DEFAULT_MODEL = "gpt-4o"
DEFAULT_OUT   = Path(__file__).parent / "results.csv"

# Instruct the model to return only the completed sentence, no reasoning.
TASK1_SYSTEM = (
    "You are a participant in a psychology experiment. "
    "Respond with ONLY the single completed sentence requested — "
    "no explanation, no punctuation beyond the sentence itself."
)
TASK2_SYSTEM = (
    "You are explaining a language choice made in a social experiment. "
    "Give a concise explanation in 2–4 sentences."
)


def col_prefix(model: str) -> str:
    """'claude-opus-4-5' → 'claude_opus_4_5'; 'google/gemini-2.5-flash' → 'google_gemini_2_5_flash'"""
    return re.sub(r"[^a-zA-Z0-9]", "_", model)


def is_openai(model: str) -> bool:
    return model.startswith("gpt-") or model.startswith("o1") or model.startswith("o3")


def is_anthropic(model: str) -> bool:
    return model.startswith("claude")


def is_openrouter(model: str) -> bool:
    return "/" in model and not model.startswith("together/")


def is_together(model: str) -> bool:
    return model.startswith("together/")


def _no_thinking(model: str) -> dict:
    """Disable thinking/reasoning tokens via extra_body."""
    if model.startswith("google/gemini"):
        return {"extra_body": {"thinking": {"type": "disabled"}}}
    if "qwen3" in model:
        return {"extra_body": {"enable_thinking": False}}
    return {}


def _api_model(model: str) -> str:
    """Resolve model ID for the API call — strips the 'together/' routing prefix."""
    if model.startswith("together/"):
        return model[len("together/"):]
    return model


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def pil_to_b64(pil_image) -> str:
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# Async OpenAI calls
# ---------------------------------------------------------------------------

async def openai_call(client, messages: list, model: str, max_tokens: int, extra_body: dict | None = None) -> str:
    resp = await client.chat.completions.create(
        model=_api_model(model),
        max_tokens=max_tokens,
        temperature=0,
        messages=messages,
        **({"extra_body": extra_body} if extra_body else {}),
    )
    return resp.choices[0].message.content.strip()


async def openai_task1_image(client, row: dict, model: str) -> str:
    img_b64 = pil_to_b64(row["clock_image"])
    return await openai_call(client, [
        {"role": "system", "content": TASK1_SYSTEM},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
            {"type": "text", "text": row["prompt"]},
        ]},
    ], model, max_tokens=40, **_no_thinking(model))


async def openai_task1_text(client, row: dict, model: str) -> str:
    return await openai_call(client, [
        {"role": "system", "content": TASK1_SYSTEM},
        {"role": "user", "content": row["clock_description"] + "\n\n" + row["prompt"]},
    ], model, max_tokens=40, **_no_thinking(model))


async def openai_task2(client, row: dict, model: str) -> str:
    return await openai_call(client, [
        {"role": "system", "content": TASK2_SYSTEM},
        {"role": "user", "content": row["prompt_motive"]},
    ], model, max_tokens=150, **_no_thinking(model))


# ---------------------------------------------------------------------------
# Async Anthropic calls
# ---------------------------------------------------------------------------

async def anthropic_call(client, system: str, content, model: str, max_tokens: int) -> str:
    resp = await client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": content}],
    )
    return resp.content[0].text.strip()


async def anthropic_task1_image(client, row: dict, model: str) -> str:
    img_b64 = pil_to_b64(row["clock_image"])
    return await anthropic_call(client, TASK1_SYSTEM, [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img_b64}},
        {"type": "text", "text": row["prompt"]},
    ], model, max_tokens=40)


async def anthropic_task1_text(client, row: dict, model: str) -> str:
    return await anthropic_call(
        client, TASK1_SYSTEM,
        row["clock_description"] + "\n\n" + row["prompt"],
        model, max_tokens=40,
    )


async def anthropic_task2(client, row: dict, model: str) -> str:
    return await anthropic_call(
        client, TASK2_SYSTEM,
        row["prompt_motive"],
        model, max_tokens=150,
    )


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

async def run_row(client, row: dict, model: str) -> tuple[str, str, str]:
    """Run all three tasks for one row concurrently; return (img, txt, t2)."""
    if is_anthropic(model):
        return await asyncio.gather(
            anthropic_task1_image(client, row, model),
            anthropic_task1_text(client, row, model),
            anthropic_task2(client, row, model),
        )
    else:  # OpenAI or OpenRouter (OpenAI-compatible)
        return await asyncio.gather(
            openai_task1_image(client, row, model),
            openai_task1_text(client, row, model),
            openai_task2(client, row, model),
        )


def make_client(model: str):
    if is_openai(model):
        if not OPENAI_API_KEY:
            sys.exit("ERROR: OPENAI_API_KEY not set in .env")
        from openai import AsyncOpenAI
        return AsyncOpenAI(api_key=OPENAI_API_KEY)
    elif is_anthropic(model):
        if not ANTHROPIC_API_KEY:
            sys.exit("ERROR: ANTHROPIC_API_KEY not set in .env")
        import anthropic
        return anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    elif is_openrouter(model):
        if not OPENROUTER_API_KEY:
            sys.exit("ERROR: OPENROUTER_API_KEY not set in .env")
        from openai import AsyncOpenAI
        return AsyncOpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)
    elif is_together(model):
        if not TOGETHER_API_KEY:
            sys.exit("ERROR: TOGETHER_API_KEY not set in .env")
        from openai import AsyncOpenAI
        return AsyncOpenAI(api_key=TOGETHER_API_KEY, base_url=TOGETHER_BASE_URL)
    else:
        sys.exit(f"ERROR: Unknown model '{model}'. Use gpt-*, claude-*, provider/model (OpenRouter), or together/model.")


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_one_per_condition(ds) -> list[int]:
    context_names = ds.features["context"].names
    seen, indices = set(), []
    for i, row in enumerate(ds):
        key = (row["target_time"], context_names[row["context"]])
        if key not in seen:
            seen.add(key)
            indices.append(i)
        if len(seen) == 24:
            break
    return sorted(indices)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def summarise(results_df: pd.DataFrame, full_df: pd.DataFrame, model: str, out_path: Path) -> None:
    col     = col_prefix(model)
    t1_img  = f"{col}_task1_image"
    t1_txt  = f"{col}_task1_text"
    t2_col  = f"{col}_task2"

    print("\n" + "=" * 70)
    print(f"RESULTS SUMMARY   model={model}   rows={len(results_df)}")
    print("=" * 70)

    print("\n--- Task 1: image vs text-description ---")
    for _, r in results_df.iterrows():
        print(
            f"  [{r['context']:8s}] {r['target_time']:8s}"
            f" | human: {str(r['human_production']):28s}"
            f" | img: {str(r[t1_img]):22s}"
            f" | txt: {str(r[t1_txt])}"
        )

    print("\n--- Task 2: first 3 model motive responses ---")
    for _, r in results_df.head(3).iterrows():
        print(f"  [{r['context']:8s}] {r['target_time']}")
        print(f"    Human labels : {r['human_motive_labels']}")
        print(f"    Model        : {str(r[t2_col])[:120]!r}")
        print()

    police   = full_df[full_df["context_str"] == "police"]["production_code"].astype(float)
    neighbor = full_df[full_df["context_str"] == "neighbor"]["production_code"].astype(float)
    wd = wasserstein_distance(police, neighbor)
    print(f"Human cross-context Wasserstein (police vs neighbor): {wd:.4f}")
    print(f"\nResults saved to: {out_path}")


# ---------------------------------------------------------------------------
# Async main
# ---------------------------------------------------------------------------

async def async_main(args) -> None:
    model  = args.model
    col    = col_prefix(model)
    client = make_client(model)

    print(f"Model  : {model}")
    print(f"Loading dataset from {HF_REPO_ID} ...")
    ds = load_dataset(HF_REPO_ID, split="train")
    print(f"Loaded {len(ds)} rows.")

    context_names  = ds.features["context"].names
    stimulus_names = ds.features["stimulus_type"].names
    motive_names   = ds.features["motive_labels"].feature.names

    full_df = ds.to_pandas()
    full_df["context_str"]       = full_df["context"].map(lambda x: context_names[x])
    full_df["stimulus_type_str"] = full_df["stimulus_type"].map(lambda x: stimulus_names[x])

    if args.full:
        eval_indices = list(range(len(ds)))
        mode = "full dataset"
    elif args.rows is not None:
        eval_indices = list(range(min(args.rows, len(ds))))
        mode = f"first {len(eval_indices)} rows"
    else:
        eval_indices = sample_one_per_condition(ds)
        mode = f"sample ({len(eval_indices)} rows, 1 per condition)"
    print(f"Mode   : {mode}   |   {3 * len(eval_indices)} concurrent API calls\n")

    # Fire all rows concurrently
    async def process(n: int, idx: int) -> dict:
        row    = ds[idx]
        ctx    = context_names[row["context"]]
        stim   = stimulus_names[row["stimulus_type"]]
        labels = [motive_names[i] for i in row["motive_labels"]]
        r_img, r_txt, r_t2 = await run_row(client, row, model)
        print(f"  [{n:3d}/{len(eval_indices)}] {row['target_time']:8s} / {ctx:8s}"
              f"  img={r_img!r:24s}  txt={r_txt!r:24s}  t2={str(r_t2)[:40]!r}")
        return {
            "item_id":               row["item_id"],
            "target_time":           row["target_time"],
            "context":               ctx,
            "stimulus_type":         stim,
            "signed_offset":         row["signed_offset"],
            "human_production":      row["production"],
            "human_production_code": row["production_code"],
            "human_motive_labels":   "|".join(labels),
            "human_motive_text":     row["motive_text"],
            f"{col}_task1_image":    r_img,
            f"{col}_task1_text":     r_txt,
            f"{col}_task2":          r_t2,
        }

    tasks   = [process(n, idx) for n, idx in enumerate(eval_indices, 1)]
    records = await asyncio.gather(*tasks)
    results_df = pd.DataFrame(records)

    # Merge into existing CSV, preserving metadata and all other model columns.
    # combine_first: results_df wins where non-NaN (new model cols + metadata for new rows);
    # existing fills in where results_df has NaN (old model cols for all rows).
    if args.out.exists():
        existing = pd.read_csv(args.out)
        existing["item_id"]   = existing["item_id"].astype(str)
        results_df["item_id"] = results_df["item_id"].astype(str)
        existing = existing.drop(columns=[c for c in existing.columns if c.startswith(col)], errors="ignore")
        results_df = (
            results_df.set_index("item_id")
            .combine_first(existing.set_index("item_id"))
            .reset_index()
        )

    results_df.to_csv(args.out, index=False)
    summarise(results_df, full_df, model, args.out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a VLM on imprecision-bench.")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="Model slug, e.g. gpt-4o or claude-opus-4-5 (default: gpt-4o)")
    parser.add_argument("--rows", type=int, default=None, help="Evaluate first N rows.")
    parser.add_argument("--full", action="store_true", help="Evaluate all 475 rows.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output CSV path.")
    args = parser.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
