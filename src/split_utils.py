"""Deterministic and reproducible split utilities for Entity Resolution pipeline.

Enforces entity-level splitting using Source 1 entities as the grouping unit,
saving stable split ID files once to disk to avoid data leakage or seed drift.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np

from src.config import OUTPUT_DIR

DEFAULT_SPLITS_DIR = OUTPUT_DIR / "splits"


def create_entity_splits(
    s1_ids: List[str],
    train_ratio: float = 0.70,
    calib_ratio: float = 0.15,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """Create deterministic entity-level train / calibration / threshold-validation splits.

    Source 1 entity IDs are used as the grouping unit to prevent entity leakage.

    Args:
        s1_ids: List of all Source 1 entity IDs.
        train_ratio: Proportion of S1 entities for training matching models.
        calib_ratio: Proportion for score calibration and model fusion.
        val_ratio: Proportion reserved untouched for final threshold selection and reporting.
        seed: Random seed for reproducibility.

    Returns:
        Dict[str, List[str]]: Dictionary with keys 'train', 'calibration', 'validation'.
    """
    total_ratio = train_ratio + calib_ratio + val_ratio
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0 (got {total_ratio})")

    # Sort first for cross-platform deterministic baseline
    sorted_ids = sorted(list(set(s1_ids)))
    n_total = len(sorted_ids)

    rng = np.random.RandomState(seed)
    shuffled_indices = rng.permutation(n_total)

    n_train = int(n_total * train_ratio)
    n_calib = int(n_total * calib_ratio)

    train_idx = shuffled_indices[:n_train]
    calib_idx = shuffled_indices[n_train:n_train + n_calib]
    val_idx = shuffled_indices[n_train + n_calib:]

    train_ids = sorted([sorted_ids[i] for i in train_idx])
    calib_ids = sorted([sorted_ids[i] for i in calib_idx])
    val_ids = sorted([sorted_ids[i] for i in val_idx])

    return {
        "train": train_ids,
        "calibration": calib_ids,
        "validation": val_ids,
    }


def save_split_ids(
    splits: Dict[str, List[str]],
    output_dir: Union[Path, str] = DEFAULT_SPLITS_DIR,
) -> None:
    """Save split entity ID lists to disk as persistent text and JSON files.

    Args:
        splits: Mapping from split name ('train', 'calibration', 'validation') to ID lists.
        output_dir: Directory where split files are written.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Save individual TXT files (one ID per line)
    for split_name, ids in splits.items():
        txt_path = out_path / f"{split_name}_s1_ids.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            for entity_id in ids:
                f.write(f"{entity_id}\n")

    # Save metadata summary
    summary = {
        split_name: {
            "count": len(ids),
            "sample_ids": ids[:5] if ids else [],
        }
        for split_name, ids in splits.items()
    }
    with open(out_path / "split_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def load_split_ids(
    splits_dir: Union[Path, str] = DEFAULT_SPLITS_DIR,
) -> Dict[str, Set[str]]:
    """Load pre-computed stable split ID sets from disk.

    Args:
        splits_dir: Directory containing saved split files.

    Returns:
        Dict[str, Set[str]]: Mapping from split name to set of Source 1 entity IDs.
    """
    dir_path = Path(splits_dir)
    if not dir_path.exists():
        raise FileNotFoundError(f"Splits directory not found at: {dir_path.resolve()}")

    splits: Dict[str, Set[str]] = {}
    for split_name in ["train", "calibration", "validation"]:
        txt_path = dir_path / f"{split_name}_s1_ids.txt"
        if not txt_path.exists():
            raise FileNotFoundError(f"Required split file missing: {txt_path.resolve()}")
        with open(txt_path, "r", encoding="utf-8") as f:
            splits[split_name] = {line.strip() for line in f if line.strip()}

    return splits
