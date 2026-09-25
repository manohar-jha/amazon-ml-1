"""LightGBM matching model module stub.

Phase 0 Interface: Defines model training, scoring, and persistence contracts.
Implementation will occur in Phase 5.
"""

import argparse
import sys
from pathlib import Path


def main() -> None:
    """CLI stub for LightGBM model training / inference."""
    parser = argparse.ArgumentParser(description="Train or infer with LightGBM entity matching model.")
    parser.add_argument("--mode", choices=["train", "predict"], default="train", help="Mode: train or predict")
    parser.add_argument("--features-path", type=str, required=True, help="Path to extracted features")
    parser.add_argument("--model-path", type=str, required=True, help="Path to save or load LightGBM model")
    parser.add_argument("--output-scores", type=str, default=None, help="Output path for predicted match scores")
    args = parser.parse_args()

    print(f"[Phase 0 Foundation] LightGBM module interface defined.")
    print(f"Mode: {args.mode}, Model Path: {args.model_path}")


if __name__ == "__main__":
    main()
