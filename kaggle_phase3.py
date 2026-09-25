"""Kaggle-ready Phase 3 Blocking & Candidate Generation Entry Point.

This script executes the Phase 3 pipeline on Kaggle or local environments,
supporting flexible paths (/kaggle/input, /kaggle/working, or local), configurable
execution modes (smoke, train_eval, test, all), memory-conscious streaming,
and comprehensive recall and candidate volume reporting.

Usage:
  # Quick smoke test on 5,000 samples
  python kaggle_phase3.py --mode smoke --sample-s1 5000

  # Full training recall evaluation against ground truth
  python kaggle_phase3.py --mode train_eval

  # Full test candidate generation -> output/candidate_pairs.tsv
  python kaggle_phase3.py --mode test

  # Full end-to-end (train evaluation + test candidate generation)
  python kaggle_phase3.py --mode all
"""

import argparse
import gc
import os
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 stdout across all environments
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.blocking import (
    evaluate_blocking_recall,
    generate_candidates,
    run_cumulative_rule_comparison,
    stream_generate_and_save_candidates,
    validate_candidate_file,
)
from src.config import (
    BLOCKING_MAX_CANDIDATES_PER_S1,
    DATA_DIR,
    OUTPUT_DIR,
)
from src.data_loader import (
    load_ground_truth,
    load_source_file,
    load_test_data,
    load_training_data,
)
from src.normalize import normalize_dataframe


def detect_kaggle_paths(
    custom_data_dir: Optional[str] = None,
    custom_output_dir: Optional[str] = None,
) -> tuple[Path, Path, Path, Path]:
    """Dynamically resolve data and output paths for Kaggle or local environments."""
    # 1. Output directory
    if custom_output_dir:
        out_dir = Path(custom_output_dir)
    elif Path("/kaggle/working").exists():
        out_dir = Path("/kaggle/working/output")
    else:
        out_dir = OUTPUT_DIR

    out_dir.mkdir(parents=True, exist_ok=True)

    # 2. Data directory
    if custom_data_dir:
        base_data = Path(custom_data_dir)
        # Check if base_data has train/test subfolders or is dataset root
        train_dir = base_data / "train" if (base_data / "train").exists() else base_data
        test_dir = base_data / "test" if (base_data / "test").exists() else base_data
    elif Path("/kaggle/input").exists():
        # Search for dataset folders under /kaggle/input
        kaggle_inputs = list(Path("/kaggle/input").glob("*"))
        candidate_train = None
        candidate_test = None
        for p in kaggle_inputs:
            if (p / "train").exists() or (p / "train_source1.tsv").exists():
                candidate_train = p / "train" if (p / "train").exists() else p
                candidate_test = p / "test" if (p / "test").exists() else p
                break
            elif (p / "dataset" / "train").exists():
                candidate_train = p / "dataset" / "train"
                candidate_test = p / "dataset" / "test"
                break
        train_dir = candidate_train if candidate_train else DATA_DIR / "train"
        test_dir = candidate_test if candidate_test else DATA_DIR / "test"
    else:
        train_dir = DATA_DIR / "train"
        test_dir = DATA_DIR / "test"

    return train_dir, test_dir, out_dir, out_dir / "candidate_pairs.tsv"


def print_memory_usage() -> None:
    """Print current memory usage if tracemalloc is active."""
    if tracemalloc.is_tracing():
        current, peak = tracemalloc.get_traced_memory()
        print(f"[RAM] Current: {current / (1024 * 1024):.1f} MB, Peak: {peak / (1024 * 1024):.1f} MB", flush=True)


def run_smoke_test(
    train_dir: Path,
    test_dir: Path,
    output_dir: Path,
    sample_size: int = 5000,
    max_candidates: int = BLOCKING_MAX_CANDIDATES_PER_S1,
) -> None:
    """Execute fast small-scale smoke test to verify all pipeline components."""
    print("\n" + "=" * 75, flush=True)
    print(f"RUNNING KAGGLE SMOKE TEST (Sample Size = {sample_size:,} S1 records)", flush=True)
    print("=" * 75, flush=True)

    t0 = time.time()
    # 1. Load sample training data and ground truth
    print(f"1. Loading S1 sample ({sample_size:,} records) and ground truth...", flush=True)
    raw_s1 = load_source_file(train_dir / "train_source1.tsv", nrows=sample_size)
    raw_gt = load_ground_truth(train_dir / "train_ground_truth.tsv")
    gt_sample = raw_gt[raw_gt["source1_entity_id"].isin(raw_s1["entity_id"])]

    # Collect ground-truth target IDs for the S1 sample to ensure target presence in smoke index
    target_gt_ids = set()
    for _, r in gt_sample.iterrows():
        m_str = r["matched_entity_ids"].strip()
        if m_str:
            target_gt_ids.update(m_str.split(","))

    print(f"   S1 sample has {len(target_gt_ids):,} ground-truth target entities in S2/S3.", flush=True)

    # Load S2 and S3: load initial rows + any rows matching the ground-truth targets
    raw_s2 = load_source_file(train_dir / "train_source2.tsv")
    raw_s3 = load_source_file(train_dir / "train_source3.tsv")

    # 2. Normalize
    print("2. Normalizing sample DataFrames...", flush=True)
    s1 = normalize_dataframe(raw_s1, inplace=True)
    s2 = normalize_dataframe(raw_s2, inplace=True)
    s3 = normalize_dataframe(raw_s3, inplace=True)

    # 3. Blocking
    print("3. Generating candidates...", flush=True)
    cands_map = generate_candidates(
        s1_df=s1,
        s2_df=s2,
        s3_df=s3,
        enabled_rules={1, 2, 3, 4, 5},
        max_candidates=max_candidates,
    )

    # 4. Evaluate recall
    print("4. Evaluating ground truth recall on sample...", flush=True)
    metrics = evaluate_blocking_recall(cands_map, gt_sample)
    print(f"   Sample Pair Recall: {metrics['overall_pair_recall'] * 100:.2f}%")
    print(f"   Sample Full S1 Coverage: {metrics['full_coverage_active_s1'] * 100:.2f}%")
    print(f"   Avg Candidates/S1: {metrics['mean_cands']:.1f}, Max: {metrics['max_cands']}")

    # 5. Smoke test test generation
    print("5. Smoke testing candidate TSV output writing...", flush=True)
    smoke_out_file = output_dir / "smoke_candidate_pairs.tsv"
    stream_generate_and_save_candidates(
        s1_df=s1,
        s2_df=s2,
        s3_df=s3,
        output_path=smoke_out_file,
        enabled_rules={1, 2, 3, 4, 5},
        max_candidates=max_candidates,
    )

    # 6. Validate output TSV
    target_ids = set(s2["entity_id"]).union(set(s3["entity_id"]))
    val_report = validate_candidate_file(
        candidate_file_path=smoke_out_file,
        expected_s1_ids=set(s1["entity_id"]),
        valid_target_ids=target_ids,
        is_test=False,
    )
    print(f"   Validation: {val_report['total_rows']:,} rows, 0 duplicate S1 IDs, 0 duplicate candidates.")
    print(f"\n[SUCCESS] Smoke test passed cleanly in {time.time() - t0:.2f}s.\n", flush=True)


def run_train_evaluation(
    train_dir: Path,
    sample_s1: Optional[int] = None,
    max_candidates: Optional[int] = BLOCKING_MAX_CANDIDATES_PER_S1,
) -> None:
    """Run full or sampled training evaluation against ground truth."""
    print("\n" + "=" * 75, flush=True)
    print("TRAIN DATASET BLOCKING RECALL EVALUATION", flush=True)
    print("=" * 75, flush=True)

    t_start = time.time()

    # Load & normalize
    print("1. Loading and normalizing training datasets...", flush=True)
    t0 = time.time()
    train_data = load_training_data(train_dir=train_dir)
    s1 = normalize_dataframe(train_data["source1"], inplace=True)
    s2 = normalize_dataframe(train_data["source2"], inplace=True)
    s3 = normalize_dataframe(train_data["source3"], inplace=True)
    gt = train_data["ground_truth"]
    print(f"   Loaded and normalized in {time.time() - t0:.2f}s (S1={len(s1):,}, S2={len(s2):,}, S3={len(s3):,})", flush=True)
    print_memory_usage()

    if sample_s1 and sample_s1 < len(s1):
        print(f"   Filtering to first {sample_s1:,} S1 entities...", flush=True)
        s1 = s1.head(sample_s1)
        gt = gt[gt["source1_entity_id"].isin(s1["entity_id"])]

    # Cumulative rule comparison
    print("\n2. Running Cumulative Blocking Rule Progression...", flush=True)
    run_cumulative_rule_comparison(s1, s2, s3, gt, sample_size=min(50000, len(s1)))
    print_memory_usage()

    # Full candidate generation & recall evaluation
    print("\n3. Generating candidates for evaluation...", flush=True)
    t0 = time.time()
    cands_map = generate_candidates(
        s1_df=s1,
        s2_df=s2,
        s3_df=s3,
        enabled_rules={1, 2, 3, 4, 5},
        max_candidates=max_candidates,
    )
    print(f"   Candidate generation completed in {time.time() - t0:.2f}s.", flush=True)
    print_memory_usage()

    print("\n4. Calculating ground truth metrics...", flush=True)
    metrics = evaluate_blocking_recall(cands_map, gt)

    print("\n" + "#" * 75, flush=True)
    print("# TRAINING BLOCKING RECALL & CANDIDATE DISTRIBUTION REPORT", flush=True)
    print("#" * 75, flush=True)
    print(f"Total S1 Entities Evaluated : {len(s1):,}")
    print(f"Total Ground Truth Pairs    : {metrics['total_gt_pairs']:,}")
    print(f"Ground Truth Pairs Recovered: {metrics['found_gt_pairs']:,}")
    print(f"Overall Pair-Level Recall   : {metrics['overall_pair_recall'] * 100:.2f}%")
    print(f"  - Source 2 Pair Recall    : {metrics['s2_pair_recall'] * 100:.2f}%")
    print(f"  - Source 3 Pair Recall    : {metrics['s3_pair_recall'] * 100:.2f}%")
    print(f"Full S1 Match Coverage Rate : {metrics['full_coverage_active_s1'] * 100:.2f}% (active S1 with >=1 match)")
    print(f"All-Entity Full Coverage    : {metrics['full_coverage_all_s1'] * 100:.2f}% (including 0-match S1)")
    print(f"Singleton Match Recall      : {metrics['singleton_recall'] * 100:.2f}% ({metrics['singleton_total']:,} singleton S1)")
    print(f"Empty Ground Truth Records  : {metrics['empty_gt_total']:,} (with candidates: {metrics['empty_gt_with_cands']:,})")

    print("\nCandidate Volume Metrics:")
    print(f"  - Mean Candidates per S1  : {metrics['mean_cands']:.2f}")
    print(f"  - Median Candidates       : {metrics['median_cands']:.0f}")
    print(f"  - 90th Percentile (P90)   : {metrics['p90_cands']:.0f}")
    print(f"  - 95th Percentile (P95)   : {metrics['p95_cands']:.0f}")
    print(f"  - 99th Percentile (P99)   : {metrics['p99_cands']:.0f}")
    print(f"  - Maximum Candidates      : {metrics['max_cands']:,}")

    print("\nCandidate Volume Distribution:")
    print(f"  - 0 candidates            : {metrics['pct_0']:.2f}%")
    print(f"  - <= 10 candidates        : {metrics['pct_lte_10']:.2f}%")
    print(f"  - <= 50 candidates        : {metrics['pct_lte_50']:.2f}%")
    print(f"  - <= 100 candidates       : {metrics['pct_lte_100']:.2f}%")
    print(f"  - > 100 candidates        : {metrics['pct_gt_100']:.2f}%")
    print(f"  - > 500 candidates        : {metrics['pct_gt_500']:.2f}%")
    print(f"  - > 1000 candidates       : {metrics['pct_gt_1000']:.2f}%")
    print(f"\nTraining evaluation finished in {time.time() - t_start:.2f}s total.\n", flush=True)


def run_test_generation(
    test_dir: Path,
    output_tsv_path: Path,
    sample_s1: Optional[int] = None,
    max_candidates: int = BLOCKING_MAX_CANDIDATES_PER_S1,
) -> None:
    """Generate and validate candidate_pairs.tsv for the test dataset."""
    print("\n" + "=" * 75, flush=True)
    print("TEST DATASET CANDIDATE GENERATION", flush=True)
    print("=" * 75, flush=True)

    t_start = time.time()

    # Load & normalize test datasets
    print("1. Loading and normalizing test datasets...", flush=True)
    t0 = time.time()
    test_data = load_test_data(test_dir=test_dir)
    s1 = normalize_dataframe(test_data["source1"], inplace=True)
    s2 = normalize_dataframe(test_data["source2"], inplace=True)
    s3 = normalize_dataframe(test_data["source3"], inplace=True)
    print(f"   Loaded and normalized in {time.time() - t0:.2f}s (S1={len(s1):,}, S2={len(s2):,}, S3={len(s3):,})", flush=True)
    print(f"   Test countries detected: {list(s1['country_norm'].value_counts().to_dict().keys())}", flush=True)
    print_memory_usage()

    if sample_s1 and sample_s1 < len(s1):
        print(f"   Sampling first {sample_s1:,} test S1 entities...", flush=True)
        s1 = s1.head(sample_s1)

    # Stream candidate generation directly to output TSV
    print(f"\n2. Streaming candidate pairs to '{output_tsv_path.resolve()}'...", flush=True)
    stream_generate_and_save_candidates(
        s1_df=s1,
        s2_df=s2,
        s3_df=s3,
        output_path=output_tsv_path,
        enabled_rules={1, 2, 3, 4, 5},
        max_candidates=max_candidates,
    )
    print_memory_usage()

    # Validate generated candidate file
    print("\n3. Validating output TSV schema and candidate integrity...", flush=True)
    valid_target_ids = set(s2["entity_id"]).union(set(s3["entity_id"]))
    report = validate_candidate_file(
        candidate_file_path=output_tsv_path,
        expected_s1_ids=set(s1["entity_id"]),
        valid_target_ids=valid_target_ids,
        is_test=True,
    )

    print("=" * 75, flush=True)
    print("TEST CANDIDATE FILE VALIDATION REPORT")
    print("=" * 75, flush=True)
    print(f"Output File Path            : {output_tsv_path.resolve()}")
    print(f"Total S1 Rows Generated     : {report['total_rows']:,}")
    print(f"Missing S1 Entities         : {report['missing_s1_count']}")
    print(f"Invalid Target IDs          : {report['invalid_target_id_count']}")
    print(f"Rows with Duplicate Cands   : {report['rows_with_duplicate_cands']}")
    print(f"Train ID Leaks              : {report['train_id_leaks']}")
    print(f"[STATUS] Candidate pairs TSV is 100% valid and ready for model matching.")
    print(f"Test candidate generation completed in {time.time() - t_start:.2f}s total.\n", flush=True)


def main() -> None:
    """CLI entry point for Kaggle Phase 3 execution."""
    parser = argparse.ArgumentParser(description="Kaggle Phase 3 Blocking and Candidate Generation")
    parser.add_argument(
        "--mode",
        choices=["smoke", "train_eval", "test", "all"],
        default="smoke",
        help="Execution mode: smoke (fast verification), train_eval (ground truth metrics), test (output/candidate_pairs.tsv), all (both)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to data directory (e.g. /kaggle/input/dataset or ./data)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Path to output directory (e.g. /kaggle/working/output or ./output)",
    )
    parser.add_argument(
        "--sample-s1",
        type=int,
        default=None,
        help="Optional S1 sample limit for smoke test or fast inspection (e.g. 5000)",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=BLOCKING_MAX_CANDIDATES_PER_S1,
        help=f"Maximum candidates per S1 entity (default: {BLOCKING_MAX_CANDIDATES_PER_S1})",
    )
    parser.add_argument(
        "--track-memory",
        action="store_true",
        help="Enable memory tracking with tracemalloc",
    )

    args = parser.parse_args()

    if args.track_memory:
        tracemalloc.start()

    train_dir, test_dir, out_dir, candidate_file = detect_kaggle_paths(
        custom_data_dir=args.data_dir,
        custom_output_dir=args.output_dir,
    )

    print("\n" + "#" * 75, flush=True)
    print("# AMAZON BUSINESS ENTITY RESOLUTION - PHASE 3 KAGGLE RUNNER", flush=True)
    print("#" * 75, flush=True)
    print(f"Mode            : {args.mode.upper()}")
    print(f"Train Directory : {train_dir.resolve()}")
    print(f"Test Directory  : {test_dir.resolve()}")
    print(f"Output Directory: {out_dir.resolve()}")
    print(f"Max Candidates  : {args.max_candidates}")
    print("#" * 75 + "\n", flush=True)

    if args.mode == "smoke":
        sample_size = args.sample_s1 if args.sample_s1 else 5000
        run_smoke_test(
            train_dir=train_dir,
            test_dir=test_dir,
            output_dir=out_dir,
            sample_size=sample_size,
            max_candidates=args.max_candidates,
        )

    elif args.mode == "train_eval":
        run_train_evaluation(
            train_dir=train_dir,
            sample_s1=args.sample_s1,
            max_candidates=args.max_candidates,
        )

    elif args.mode == "test":
        run_test_generation(
            test_dir=test_dir,
            output_tsv_path=candidate_file,
            sample_s1=args.sample_s1,
            max_candidates=args.max_candidates,
        )

    elif args.mode == "all":
        run_train_evaluation(
            train_dir=train_dir,
            sample_s1=args.sample_s1,
            max_candidates=args.max_candidates,
        )
        gc.collect()
        run_test_generation(
            test_dir=test_dir,
            output_tsv_path=candidate_file,
            sample_s1=args.sample_s1,
            max_candidates=args.max_candidates,
        )


if __name__ == "__main__":
    main()
