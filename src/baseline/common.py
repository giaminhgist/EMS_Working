"""Shared paths and utilities for EMS-Baseline baselines."""
from pathlib import Path

ROOT = Path("/root/EMS-Minh")
PROCESSED = ROOT / "processed_dataset"
RESULTS = ROOT / "docs" / "baseline" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

SEED = 42
THRESHOLD = 0.5
NUM_STIMULI = 100
CATEGORIES = ["social", "natural", "synthetic", "manipulated"]

# stimulus filename prefix -> category (same mapping as EMS-Minh/src/common.py)
PREFIX_CATEGORY = {
    "act": "social", "por": "social", "soc": "social",
    "ind": "natural", "land": "natural", "outman": "natural", "sat": "natural",
    "art": "synthetic", "cat": "synthetic", "pat": "synthetic",
    "low": "manipulated", "mood": "manipulated", "noi": "manipulated",
    "patch": "manipulated", "rand": "manipulated",
}


def image_category(name):
    return PREFIX_CATEGORY[name.replace(".jpg", "").split("_")[0]]
