"""CLI command to create and save persistent entity-level validation splits.

Usage:
  python -m src.split --seed 42 --train-ratio 0.70 --calib-ratio 0.15 --val-ratio 0.15
"""

import argparse
import sys
from pathlib import Path

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TRAIN_DIR, OUTPUT_DIR
from src.io_utils import read_tsv
from src.schemas import COL_ENTITY_ID, COL_SOURCE1_ID
from src.split_utils import create_entity_splits, save_split_ids


def main() -> None:
    """Run split creation from training Source 1."""
    parser = argparse.ArgumentParser(description="Create deterministic entity-level dataset splits.")
    parser.add_argument("--s1-path", type=str, default=str(TRAIN_DIR / "train_source1.tsv"), help="Path to train_source1.tsv")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR / "splits"), help="Output directory for split ID files")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Train partition ratio (default: 0.70)")
    parser.add_argument("--calib-ratio", type=float, default=0.15, help="Calibration/fusion partition ratio (default: 0.15)")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation/threshold partition ratio (default: 0.15)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting (default: 42)")

    args = parser.parse_args()

    s1_path = Path(args.s1_path)
    if not s1_path.exists():
        print(f"[ERROR] Source 1 file not found at: {s1_path.resolve()}")
        sys.exit(1)

    print(f"Loading Source 1 entities from '{s1_path.name}'...")
    df_s1 = read_tsv(s1_path)
    id_col = COL_SOURCE1_ID if COL_SOURCE1_ID in df_s1.columns else COL_ENTITY_ID
    s1_ids = df_s1[id_col].tolist()

    print(f"Creating deterministic splits for {len(s1_ids):,} entities (seed={args.seed})...")
    splits = create_entity_splits(
        s1_ids=s1_ids,
        train_ratio=args.train_ratio,
        calib_ratio=args.calib_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    out_dir = Path(args.output_dir)
    save_split_ids(splits, output_dir=out_dir)

    print(f"[SUCCESS] Splits generated and saved to '{out_dir.resolve()}':")
    for name, ids in splits.items():
        print(f"  - {name:12s}: {len(ids):,} entities ({len(ids)/len(s1_ids)*100:.1f}%) -> {name}_s1_ids.txt")


if __name__ == "__main__":
    main()
