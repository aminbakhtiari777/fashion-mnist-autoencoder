"""Evaluate exported Fashion Vision Lab arrays stored in NPZ format."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fashion_vision.evaluation import (  # noqa: E402
    classification_summary,
    generation_summary,
    reconstruction_summary,
)


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, float) and not np.isfinite(value):
        return "Infinity" if value > 0 else "-Infinity"
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("track", choices=["classification", "reconstruction", "generation"])
    parser.add_argument("--input", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    arrays = np.load(args.input)
    if args.track == "classification":
        report = classification_summary(arrays["labels"], arrays["probabilities"])
    elif args.track == "reconstruction":
        report = reconstruction_summary(arrays["original"], arrays["reconstructed"])
    else:
        report = generation_summary(arrays["images"])
    print(json.dumps(json_safe(report), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
