"""
push.py — Build the dataset locally and push to HuggingFace Hub.

Run from the project root:
    python push.py

Requires:
    pip install datasets huggingface_hub pillow python-dotenv
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import login, HfApi
from datasets import load_from_disk

# ---------------------------------------------------------------------------
# Load credentials
# ---------------------------------------------------------------------------

load_dotenv()
HF_TOKEN = os.environ["HF_TOKEN"]
HF_REPO_ID = os.environ["HF_REPO_ID"]

PROJECT_ROOT = Path(__file__).parent
DATASET_DIR = PROJECT_ROOT / "imprecision_bench_dataset"
README_PATH = PROJECT_ROOT / "README.md"

# ---------------------------------------------------------------------------
# Step 1 — Build the dataset (if not already built)
# ---------------------------------------------------------------------------

if not DATASET_DIR.exists():
    print("Building dataset...")
    import build_dataset
    build_dataset.main()
else:
    print(f"Dataset already built at {DATASET_DIR}. Skipping build.")

# ---------------------------------------------------------------------------
# Step 2 — Login and push dataset
# ---------------------------------------------------------------------------

print(f"Logging in to HuggingFace...")
login(token=HF_TOKEN)

print(f"Pushing dataset to {HF_REPO_ID}...")
ds = load_from_disk(str(DATASET_DIR))
ds.push_to_hub(HF_REPO_ID, private=False)
print("Dataset pushed.")

# ---------------------------------------------------------------------------
# Step 3 — Push README (dataset card)
# ---------------------------------------------------------------------------

print("Pushing README.md (dataset card)...")
api = HfApi()
api.upload_file(
    path_or_fileobj=str(README_PATH),
    path_in_repo="README.md",
    repo_id=HF_REPO_ID,
    repo_type="dataset",
)
print("README pushed.")

print(f"\nDone. Dataset live at: https://huggingface.co/datasets/{HF_REPO_ID}")
