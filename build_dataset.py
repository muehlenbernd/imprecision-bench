"""
build_dataset.py — Build the imprecision-bench HF dataset.

Sources (read-only):
  - data_clean.csv          (475 productions × 22 columns, project root)
  - images/clock*.png       (12 clock images)

Output (writable):
  - imprecision_bench_dataset/   (HF dataset, save_to_disk, project root)
"""

import re
from pathlib import Path

import pandas as pd
from datasets import (
    Dataset, Features, Value, ClassLabel, Sequence, Image,
)

PROJECT_ROOT = Path(__file__).parent
IMAGES_DIR = PROJECT_ROOT / "images"
OUTPUT_DIR = PROJECT_ROOT / "imprecision_bench_dataset"


# ---------------------------------------------------------------------------
# Locked design — clock descriptions
# ---------------------------------------------------------------------------

RANGE_TARGET = "8:26-8:34"

CLOCK_DESCRIPTIONS = {
    "8:25": "Clock description: Hour hand between 8 and 9. Minute hand at the 5.",
    "8:26": "Clock description: Hour hand between 8 and 9. Minute hand one minute-mark past the 5.",
    "8:27": "Clock description: Hour hand between 8 and 9. Minute hand two minute-marks past the 5.",
    "8:28": "Clock description: Hour hand between 8 and 9. Minute hand three minute-marks past the 5.",
    "8:29": "Clock description: Hour hand between 8 and 9. Minute hand four minute-marks past the 5.",
    "8:30": "Clock description: Hour hand between 8 and 9, halfway. Minute hand at the 6.",
    "8:31": "Clock description: Hour hand between 8 and 9. Minute hand one minute-mark past the 6.",
    "8:32": "Clock description: Hour hand between 8 and 9. Minute hand two minute-marks past the 6.",
    "8:33": "Clock description: Hour hand between 8 and 9. Minute hand three minute-marks past the 6.",
    "8:34": "Clock description: Hour hand between 8 and 9. Minute hand four minute-marks past the 6.",
    "8:35": "Clock description: Hour hand between 8 and 9. Minute hand at the 7.",
    RANGE_TARGET: (
        "Clock description: Hour hand between 8 and 9. "
        "The minute hand is replaced by a shaded yellow region spanning "
        "from near the 5 to near the 7, indicating that the minute hand "
        "is somewhere within that region."
    ),
}

CLOCK_IMAGE_FILES = {
    "8:25": IMAGES_DIR / "clock825.png",
    "8:26": IMAGES_DIR / "clock826.png",
    "8:27": IMAGES_DIR / "clock827.png",
    "8:28": IMAGES_DIR / "clock828.png",
    "8:29": IMAGES_DIR / "clock829.png",
    "8:30": IMAGES_DIR / "clock830.png",
    "8:31": IMAGES_DIR / "clock831.png",
    "8:32": IMAGES_DIR / "clock832.png",
    "8:33": IMAGES_DIR / "clock833.png",
    "8:34": IMAGES_DIR / "clock834.png",
    "8:35": IMAGES_DIR / "clock835.png",
    RANGE_TARGET: IMAGES_DIR / "clocka830.png",
}


# ---------------------------------------------------------------------------
# Locked design — prompt template
# ---------------------------------------------------------------------------

SCENARIO_A = (
    "One morning when you leave your house, you witness an automobile "
    "accident in your street. You look at your watch when it happens."
)

SCENARIO_C = (
    'How would you answer in this situation? (Fill the blank)\n\n'
    '"It happened ___."'
)


def part_b(context: str, stimulus_type: str) -> str:
    """Build Part B given context (police/neighbor) and stimulus_type (precise/range)."""
    time_phrase = "at the time" if stimulus_type == "precise" else "in the time range"
    if context == "police":
        return (
            "Later that day you are invited to the police station to give a "
            "formal witness statement about the accident. The police officer "
            "is trying to establish a detailed timeline of the event. "
            "He asks you: \"What time did the accident happen?\" "
            f"You remember that it happened {time_phrase} shown on the clock "
            "as given above."
        )
    if context == "neighbor":
        return (
            "Later that day you are invited to a party at a neighbor's house, "
            "where people are talking about the accident. Your neighbor is "
            "eager to find out what you saw of the event. "
            "She asks you: \"What time did the accident happen?\" "
            f"You remember that it happened {time_phrase} shown on the clock "
            "as given above."
        )
    raise ValueError(f"Unknown context: {context!r}")


def build_prompt(context: str, stimulus_type: str) -> str:
    return f"{SCENARIO_A} {part_b(context, stimulus_type)}\n\n{SCENARIO_C}"


# ---------------------------------------------------------------------------
# Locked design — Task 2 (motive elicitation) prompt template
# ---------------------------------------------------------------------------

def format_time_phrase(target_time: str, stimulus_type: str) -> str:
    """Render the ground-truth time as a natural-language phrase."""
    if stimulus_type == "precise":
        return f"at {target_time}"
    # range: "8:26-8:34" → "in the time range between 8:26 and 8:34"
    start, end = target_time.split("-")
    return f"in the time range between {start} and {end}"


def part_b_motive(context: str, time_phrase: str, production: str) -> str:
    """Build Task 2's context-embedded recap + motive question."""
    quoted_answer = f'"It happened {production}"'
    if context == "police":
        return (
            "Later, you gave a formal witness statement at a police "
            "station. The officer, trying to establish a detailed "
            "timeline, asked you what time the accident happened. "
            f"You knew that the accident happened {time_phrase}, and "
            f"your answer was {quoted_answer}. "
            "Why did you choose to answer this way?"
        )
    if context == "neighbor":
        return (
            "Later, at a party at a neighbor's house, your neighbor — "
            "eager to find out what you saw — asked you what time the "
            "accident happened. "
            f"You knew that the accident happened {time_phrase}, and "
            f"your answer was {quoted_answer}. "
            "Why did you choose to answer this way?"
        )
    raise ValueError(f"Unknown context: {context!r}")


def build_prompt_motive(
    context: str, stimulus_type: str, target_time: str, production: str
) -> str:
    time_phrase = format_time_phrase(target_time, stimulus_type)
    return f"{SCENARIO_A} {part_b_motive(context, time_phrase, production)}"


# ---------------------------------------------------------------------------
# State parsing
# ---------------------------------------------------------------------------

_PRECISE_STATE_RE = re.compile(r"^at (\d+):(\d+)$")

def parse_state(state_str: str) -> dict:
    """Parse the source CSV `state` column into normalized fields.

    Returns dict with: target_time, stimulus_type, signed_offset, abs_offset.
    signed_offset is offset (in minutes) from 8:30; null for range.
    """
    s = state_str.strip()
    m = _PRECISE_STATE_RE.match(s)
    if m:
        h, mm = int(m.group(1)), int(m.group(2))
        target_time = f"{h}:{mm:02d}"
        offset = mm - 30
        return {
            "target_time": target_time,
            "stimulus_type": "precise",
            "signed_offset": offset,
            "abs_offset": abs(offset),
        }
    if "-" in s:
        return {
            "target_time": RANGE_TARGET,
            "stimulus_type": "range",
            "signed_offset": None,
            "abs_offset": None,
        }
    raise ValueError(f"Unknown state value: {state_str!r}")


# ---------------------------------------------------------------------------
# Class label vocabularies
# ---------------------------------------------------------------------------

MOTIVE_CLASSES = [
    "Precision", "Accuracy", "Info lack", "Misinfo", "Safe", "H needs",
    "Context", "S ease", "H ease", "Habit", "Sound", "Other",
]

APPROXIMATOR_CLASSES = [
    "none", "around", "about", "just before/after",
    "approximately", "ish", "nearly", "roughly", "round about",
]


def normalize_appx(raw) -> str:
    """Map source appxC value to the canonical class label."""
    if pd.isna(raw) or raw == "0":
        return "none"
    return str(raw)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def main() -> Dataset:
    df = pd.read_csv(PROJECT_ROOT / "data_clean.csv")

    rows = []
    for _, src in df.iterrows():
        state_info = parse_state(src["state"])
        motives = [m for m in MOTIVE_CLASSES if src.get(m) == "x"]
        motive_text = "" if pd.isna(src["motive"]) else str(src["motive"])

        rows.append({
            "item_id":           str(src["id"]),
            "subject_id":        int(src["sid"]),
            "context":           src["context"],
            "target_time":       state_info["target_time"],
            "stimulus_type":     state_info["stimulus_type"],
            "signed_offset":     state_info["signed_offset"],
            "abs_offset":        state_info["abs_offset"],
            "production":        str(src["answer"]),
            "production_code":   int(src["answerC"]),
            "approximator":      normalize_appx(src["appxC"]),
            "motive_labels":     motives,
            "motive_text":       motive_text,
            "clock_image":       str(CLOCK_IMAGE_FILES[state_info["target_time"]]),
            "clock_description": CLOCK_DESCRIPTIONS[state_info["target_time"]],
            "prompt":            build_prompt(src["context"], state_info["stimulus_type"]),
            "prompt_motive":     build_prompt_motive(
                src["context"],
                state_info["stimulus_type"],
                state_info["target_time"],
                str(src["answer"]),
            ),
        })

    features = Features({
        "item_id":           Value("string"),
        "subject_id":        Value("int32"),
        "context":           ClassLabel(names=["police", "neighbor"]),
        "target_time":       Value("string"),
        "stimulus_type":     ClassLabel(names=["precise", "range"]),
        "signed_offset":     Value("int8"),
        "abs_offset":        Value("int8"),
        "production":        Value("string"),
        "production_code":   Value("int8"),
        "approximator":      ClassLabel(names=APPROXIMATOR_CLASSES),
        "motive_labels":     Sequence(ClassLabel(names=MOTIVE_CLASSES)),
        "motive_text":       Value("string"),
        "clock_image":       Image(),
        "clock_description": Value("string"),
        "prompt":            Value("string"),
        "prompt_motive":     Value("string"),
    })

    ds = Dataset.from_list(rows, features=features)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(OUTPUT_DIR))

    print(f"Saved {len(ds)} rows to {OUTPUT_DIR}")
    print(f"Features:\n{ds.features}")
    return ds


if __name__ == "__main__":
    main()
