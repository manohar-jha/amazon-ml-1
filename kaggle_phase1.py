"""Kaggle entry point for Phase 1 Candidate Generation & Blocking.

Executes:
1. Inverted token indexing with posting caps and rare token prioritization.
2. Character 3-gram MinHash LSH for sublinear approximate name matching.
3. Bounded index-backed fuzzy fallback with soft country preference.
4. Validation split candidate recall evaluation on saved output/splits/validation_s1_ids.txt.
5. Deterministic serialization of candidate_pairs.tsv and pair-level candidate_provenance.tsv.

Usage:
  # Fast smoke test
  python kaggle_phase1.py --mode smoke --sample-s1 1000

  # Validation recall evaluation on saved validation split
  python kaggle_phase1.py --mode eval_val

  # Full test candidate generation
  python kaggle_phase1.py --mode test

  # Full end-to-end (validation evaluation + test candidate generation)
  python kaggle_phase1.py --mode all
"""

import argparse
import gc
import json
import os
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict, Optional, Set
import pandas as pd

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 stdout
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.batch_pipeline import (
    build_sharded_target_index,
    check_resource_headroom,
    merge_candidate_partitions,
    run_disk_backed_retrieval_pipeline,
    stream_process_s1_batches,
)
from src.blocking import (
    evaluate_validation_recall,
    generate_candidate_union,
    write_candidate_outputs,
)
from src.config import (
    BATCH_SIZE_S1_QUERIES,
    BLOCKING_MAX_CANDIDATES_PER_S1,
    BLOCKING_PER_SHARD_CANDIDATE_CAP,
    DATA_DIR,
    MANIFEST_PATH,
    MIN_FREE_DISK_GB,
    MIN_FREE_RAM_GB,
    OUTPUT_DIR,
    PARTITIONS_DIR,
    PILOT_CHUNKSIZE,
    PILOT_SCAN_ROWS,
    PILOT_SAMPLE_S1,
    PILOT_SAMPLE_TARGET,
    PILOT_SEED,
    PILOT_SUMMARY_PATH,
    RAM_SAFETY_MARGIN,
    SHARD_SIZE_TARGETS,
    SPLITS_DIR,
)
from src.data_loader import (
    load_ground_truth,
    load_source_file,
    load_test_data,
    load_training_data,
    scan_and_sample_source_file,
)
from src.io_utils import read_tsv
from src.normalize import normalize_dataframe
from src.sharded_index import benchmark_shard_memory_rss, get_process_rss_bytes
from src.split_utils import load_split_ids
from utils.validate_submission import validate_submission_file, validate_submission_streaming


def get_process_memory_mb() -> Dict[str, float]:
    """Get current and peak process memory usage in megabytes (MB)."""
    mem_info: Dict[str, float] = {}

    # 1. OS-level process memory via psutil
    try:
        import psutil
        proc = psutil.Process(os.getpid())
        mem_info["rss_mb"] = proc.memory_info().rss / (1024 * 1024)
        if hasattr(proc.memory_info(), "peak_wset"):
            mem_info["peak_rss_mb"] = proc.memory_info().peak_wset / (1024 * 1024)
    except Exception:
        pass

    # 2. OS-level process memory via resource (standard on Linux / Kaggle)
    try:
        import resource
        ru = resource.getrusage(resource.RUSAGE_SELF)
        # On Linux ru_maxrss is in KB; on macOS in bytes
        if sys.platform == "darwin":
            mem_info["peak_rss_mb"] = ru.ru_maxrss / (1024 * 1024)
        else:
            mem_info["peak_rss_mb"] = ru.ru_maxrss / 1024.0
    except Exception:
        pass

    # 3. Tracemalloc Python heap tracker
    if tracemalloc.is_tracing():
        cur, peak = tracemalloc.get_traced_memory()
        mem_info["tracemalloc_cur_mb"] = cur / (1024 * 1024)
        mem_info["tracemalloc_peak_mb"] = peak / (1024 * 1024)

    return mem_info


def resolve_paths(
    custom_data_dir: Optional[str] = None,
    custom_splits_dir: Optional[str] = None,
    custom_output_dir: Optional[str] = None,
) -> tuple[Path, Path, Path, Path]:
    """Resolve data, split, and output directories for Kaggle or local environments."""
    # Output directory
    if custom_output_dir:
        out_dir = Path(custom_output_dir)
    elif Path("/kaggle/working").exists():
        out_dir = Path("/kaggle/working/output")
    else:
        out_dir = OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Splits directory
    if custom_splits_dir:
        splits_dir = Path(custom_splits_dir)
    elif (out_dir / "splits").exists():
        splits_dir = out_dir / "splits"
    elif SPLITS_DIR.exists():
        splits_dir = SPLITS_DIR
    else:
        splits_dir = out_dir / "splits"

    # Data directory
    if custom_data_dir:
        base_data = Path(custom_data_dir)
        train_dir = base_data / "train" if (base_data / "train").exists() else base_data
        test_dir = base_data / "test" if (base_data / "test").exists() else base_data
    elif Path("/kaggle/input").exists():
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

    return train_dir, test_dir, splits_dir, out_dir


def print_memory() -> None:
    """Log current and peak RAM usage."""
    if tracemalloc.is_tracing():
        cur, peak = tracemalloc.get_traced_memory()
        print(f"[RAM] Current: {cur / (1024 * 1024):.1f} MB, Peak: {peak / (1024 * 1024):.1f} MB", flush=True)


def run_smoke_test(
    train_dir: Path,
    out_dir: Path,
    sample_size: int = 1000,
    sample_target: Optional[int] = 5000,
    max_candidates: int = BLOCKING_MAX_CANDIDATES_PER_S1,
) -> None:
    """Execute synthetic smoke test verifying MinHash LSH, soft country fallback, and candidate tables."""
    print("\n" + "=" * 75, flush=True)
    print(f"RUNNING PHASE 1 SMOKE TEST (Sample Size = {sample_size:,} S1, Targets = {sample_target if sample_target else 'Full'})", flush=True)
    print("=" * 75, flush=True)

    t0 = time.time()
    # 1. Load sample S1 and sample/full S2/S3
    raw_s1 = load_source_file(train_dir / "train_source1.tsv", nrows=sample_size)
    raw_s2 = load_source_file(train_dir / "train_source2.tsv", nrows=sample_target)
    raw_s3 = load_source_file(train_dir / "train_source3.tsv", nrows=sample_target)
    raw_gt = load_ground_truth(train_dir / "train_ground_truth.tsv")
    gt_sample = raw_gt[raw_gt["source1_entity_id"].isin(raw_s1["entity_id"])]

    # 2. Normalize
    s1 = normalize_dataframe(raw_s1, inplace=True)
    s2 = normalize_dataframe(raw_s2, inplace=True)
    s3 = normalize_dataframe(raw_s3, inplace=True)

    # 3. Generate candidate union with provenance
    cands_map, prov_records, stats = generate_candidate_union(
        s1_df=s1,
        s2_df=s2,
        s3_df=s3,
        max_candidates=max_candidates,
    )

    # 4. Serialize to test output files
    smoke_pairs_path = out_dir / "smoke_candidate_pairs.tsv"
    smoke_prov_path = out_dir / "smoke_candidate_provenance.tsv"
    write_candidate_outputs(
        candidates_map=cands_map,
        provenance_records=prov_records,
        ordered_s1_ids=s1["entity_id"].tolist(),
        candidate_pairs_path=smoke_pairs_path,
        provenance_path=smoke_prov_path,
    )

    # 5. Validate output TSV
    target_ids = set(s2["entity_id"]).union(set(s3["entity_id"]))
    is_valid, errors = validate_submission_file(
        submission_path=smoke_pairs_path,
        expected_s1_ids=set(s1["entity_id"]),
        valid_target_ids=target_ids,
        is_candidate_file=True,
    )

    val_res = evaluate_validation_recall(
        candidates_map=cands_map,
        val_s1_ids=set(s1["entity_id"]),
        gt_df=gt_sample,
    )

    print("\n--- Smoke Test Results ---", flush=True)
    print(f"Sample Pair Recall          : {val_res['val_pair_recall'] * 100:.2f}% ({val_res['val_found_gt_pairs']:,} / {val_res['val_total_gt_pairs']:,} pairs)", flush=True)
    print(f"Full S1 Match Coverage Rate : {val_res['val_full_s1_coverage'] * 100:.2f}%", flush=True)
    print(f"Mean Candidates per S1      : {stats['mean_candidates']:.1f} (Median: {stats['median_candidates']:.0f}, P95: {stats['p95_candidates']:.0f}, Max: {stats['max_candidates']})", flush=True)
    print(f"Candidate Truncation Events : {stats['truncation_count']:,} ({stats['truncation_rate'] * 100:.2f}%)", flush=True)
    print(f"Validation Schema Status    : {'100% VALID' if is_valid else 'ERRORS: ' + str(errors)}", flush=True)
    print(f"[SUCCESS] Smoke test completed in {time.time() - t0:.2f}s.\n", flush=True)


def run_validation_evaluation(
    train_dir: Path,
    splits_dir: Path,
    out_dir: Path,
    sample_s1: Optional[int] = None,
    batch_size: int = BATCH_SIZE_S1_QUERIES,
    shard_size: int = SHARD_SIZE_TARGETS,
    chunksize: int = PILOT_CHUNKSIZE,
    max_candidates: int = BLOCKING_MAX_CANDIDATES_PER_S1,
    min_disk_gb: float = MIN_FREE_DISK_GB,
    min_ram_gb: float = MIN_FREE_RAM_GB,
    resume: bool = True,
    clean_partitions: bool = False,
) -> None:
    """Evaluate candidate generation recall strictly on the saved validation split using streaming sharded index."""
    print("\n" + "=" * 75, flush=True)
    print("PHASE 1 STREAMING DISK-BACKED VALIDATION RECALL EVALUATION", flush=True)
    print("=" * 75, flush=True)

    t_start = time.time()

    # 1. Resolve and load saved validation split IDs first (read-only)
    val_txt = splits_dir / "validation_s1_ids.txt"
    if not val_txt.exists():
        raise FileNotFoundError(
            f"Validation split file not found at '{val_txt.resolve()}'. "
            "Please run 'python -m src.split' first to generate persistent split files."
        )

    with open(val_txt, "r", encoding="utf-8") as f:
        val_s1_ids = {line.strip() for line in f if line.strip()}
    print(f"Loaded {len(val_s1_ids):,} saved validation S1 IDs from '{val_txt.name}'.", flush=True)

    if sample_s1 and sample_s1 < len(val_s1_ids):
        print(f"Sampling first {sample_s1:,} validation S1 entities for evaluation...", flush=True)
        val_s1_ids = set(sorted(val_s1_ids)[:sample_s1])

    s1_eval_count = len(val_s1_ids)

    # 2. Benchmark target slice memory footprint (provisional RSS scaling with representative sample and query batch)
    print("Measuring empirical target index memory scaling (OS RSS)...", flush=True)
    sample_s2 = load_source_file(train_dir / "train_source2.tsv", nrows=2500)
    sample_s3 = load_source_file(train_dir / "train_source3.tsv", nrows=2500)
    sample_targets = pd.concat([sample_s2, sample_s3], ignore_index=True)
    sample_s1_df = load_source_file(train_dir / "train_source1.tsv", nrows=200)

    bench = benchmark_shard_memory_rss(
        sample_df=sample_targets,
        query_batch_df=sample_s1_df,
        shard_size_targets=shard_size,
    )
    measured_bpt = bench["rss_bytes_per_target"]
    print(
        f"  -> [PROVISIONAL ESTIMATE] Measured {measured_bpt:.1f} bytes/target from sample (n={len(sample_targets):,}) "
        f"with query batch (n={len(sample_s1_df):,}).\n"
        f"     Provisional estimated shard footprint: {bench['estimated_shard_rss_gb']:.2f} GB per {shard_size:,} shard ({RAM_SAFETY_MARGIN}x safety margin).\n"
        f"     (Note: actual peak RSS is monitored and logged per resident shard during execution.)",
        flush=True,
    )

    # 3. Pre-flight resource safety check with measured bytes per target and actual S1 count
    is_safe, res_info, msg = check_resource_headroom(
        out_dir,
        min_disk_gb=min_disk_gb,
        min_ram_gb=min_ram_gb,
        shard_size=shard_size,
        measured_bytes_per_target=measured_bpt,
        safety_margin=RAM_SAFETY_MARGIN,
        estimated_s1_count=s1_eval_count,
        estimated_target_count=10300000,
        per_shard_candidate_cap=BLOCKING_PER_SHARD_CANDIDATE_CAP,
        max_candidates=max_candidates,
        sample_targets_df=sample_targets,
        sample_s1_df=sample_s1_df,
    )
    print(f"[RESOURCE CHECK] {msg}", flush=True)
    if "disk_breakdown_gib" in res_info:
        db = res_info["disk_breakdown_gib"]
        print(
            f"  -> Disk Breakdown (GiB): Target Shards={db['target_shards_gib']:.2f} GiB, S1 Batches={db['s1_batches_gib']:.2f} GiB,\n"
            f"     Intermediate: Conservative={db['conservative_intermediate_gib']:.2f} GiB (Empirical={db['empirical_intermediate_gib']:.2f} GiB),\n"
            f"     Merged Batches: Conservative={db['merged_batches_conservative_gib']:.2f} GiB (Empirical={db['merged_batches_empirical_gib']:.2f} GiB),\n"
            f"     Final Outputs: Conservative={db['final_outputs_conservative_gib']:.2f} GiB (Empirical={db['final_outputs_empirical_gib']:.2f} GiB),\n"
            f"     Merge Coexistence / Tmp: {db['tmp_coexistence_conservative_gib']:.2f} GiB (Empirical={db['tmp_coexistence_empirical_gib']:.2f} GiB)",
            flush=True,
        )
    if not is_safe:
        raise RuntimeError(f"Resource safety check failed: {msg}")

    # 4. Load ground truth lookup strictly for validation entities (~25 MB RAM)
    print("Loading validation split ground-truth lookup (memory bounded)...", flush=True)
    gt_path = train_dir / "train_ground_truth.tsv"
    gt_lookup: Dict[str, Set[str]] = {}
    with open(gt_path, "r", encoding="utf-8") as f_gt:
        header = f_gt.readline()
        for line in f_gt:
            parts = line.rstrip("\r\n").split("\t")
            s1_id = parts[0].strip()
            if s1_id in val_s1_ids:
                m_str = parts[1].strip() if len(parts) > 1 else ""
                gt_lookup[s1_id] = set(m_str.split(",")) if m_str else set()
    print(f"Loaded ground truth for {len(gt_lookup):,} validation entities.", flush=True)

    # 5. Execute single-shard-resident disk-backed retrieval pipeline
    partitions_dir = out_dir / "partitions"
    manifest_path = out_dir / "validation_manifest.json"

    batch_stats = run_disk_backed_retrieval_pipeline(
        s1_path=train_dir / "train_source1.tsv",
        s2_path=train_dir / "train_source2.tsv",
        s3_path=train_dir / "train_source3.tsv",
        partitions_dir=partitions_dir,
        manifest_path=manifest_path,
        mode="eval_val",
        split_file=val_txt,
        filter_s1_ids=val_s1_ids,
        sample_s1=sample_s1,
        gt_lookup=gt_lookup,
        batch_size=batch_size,
        shard_size=shard_size,
        chunksize=chunksize,
        max_candidates=max_candidates,
        per_shard_candidate_cap=BLOCKING_PER_SHARD_CANDIDATE_CAP,
        resume=resume,
    )

    # 6. Merge partition files into final candidate files
    final_pairs = out_dir / "candidate_pairs.tsv"
    final_prov = out_dir / "candidate_provenance.tsv"
    merge_info = merge_candidate_partitions(
        partitions_dir=partitions_dir,
        output_pairs_path=final_pairs,
        output_prov_path=final_prov,
        clean_partitions=clean_partitions,
    )

    # 7. Validate output TSV schema
    is_valid, errors = validate_submission_file(
        submission_path=final_pairs,
        expected_s1_ids=val_s1_ids,
        is_candidate_file=True,
    )

    total_elapsed = time.time() - t_start
    mem_info = get_process_memory_mb()

    # 8. Write validation summary JSON
    summary = {
        "mode": "eval_val",
        "total_validation_s1": len(val_s1_ids),
        "total_s1_processed": batch_stats["total_s1_processed"],
        "total_candidate_pairs": batch_stats["total_candidate_pairs"],
        "candidate_distribution": {
            "mean": batch_stats["mean_candidates"],
            "median": batch_stats["median_candidates"],
            "p90": batch_stats["p90_candidates"],
            "p95": batch_stats["p95_candidates"],
            "p99": batch_stats["p99_candidates"],
            "max": batch_stats["max_candidates"],
            "empty_candidate_count": batch_stats["empty_candidate_count"],
            "empty_candidate_rate": batch_stats["empty_candidate_rate"],
            "truncation_count": batch_stats["total_truncations"],
            "truncation_rate": batch_stats["truncation_rate"],
            "total_shard_candidates_pruned": batch_stats.get("total_shard_candidates_pruned", 0),
        },
        "recall_metrics": {
            "total_gt_pairs": batch_stats["val_total_gt_pairs"],
            "found_gt_pairs": batch_stats["val_found_gt_pairs"],
            "overall_pair_recall": round(batch_stats["val_pair_recall"], 5),
            "source2_pair_recall": round(batch_stats["val_s2_recall"], 5),
            "source3_pair_recall": round(batch_stats["val_s3_recall"], 5),
            "full_s1_coverage_rate": round(batch_stats["val_full_s1_coverage"], 5),
        },
        "timing_and_throughput": {
            "total_elapsed_seconds": round(total_elapsed, 2),
            "overall_throughput_qps": batch_stats["overall_throughput_qps"],
        },
        "measured_shard_ram_mb": batch_stats.get("measured_shard_ram_mb", []),
        "memory_mb": mem_info,
        "validation_schema_valid": is_valid,
        "validation_schema_errors": errors,
    }

    summary_path = out_dir / "validation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # 9. Print formatted console report
    print("\n" + "#" * 75, flush=True)
    print("# VALIDATION SPLIT CANDIDATE RECALL REPORT (DISK-BACKED)", flush=True)
    print("#" * 75, flush=True)
    print(f"Summary JSON Written        : {summary_path.resolve()}", flush=True)
    print(f"Validation S1 Entities      : {len(val_s1_ids):,}", flush=True)
    print(f"Total True GT Pairs         : {batch_stats['val_total_gt_pairs']:,}", flush=True)
    print(f"True Pairs Recovered        : {batch_stats['val_found_gt_pairs']:,}", flush=True)
    print(f"Overall Pair Recall         : {batch_stats['val_pair_recall'] * 100:.2f}%", flush=True)
    print(f"  - Source 2 Pair Recall    : {batch_stats['val_s2_recall'] * 100:.2f}%", flush=True)
    print(f"  - Source 3 Pair Recall    : {batch_stats['val_s3_recall'] * 100:.2f}%", flush=True)
    print(f"Full S1 Match Coverage Rate : {batch_stats['val_full_s1_coverage'] * 100:.2f}%", flush=True)
    print("\nCandidate Volume & Truncation Metrics:", flush=True)
    print(f"  - Total Generated Pairs   : {batch_stats['total_candidate_pairs']:,}", flush=True)
    print(f"  - Mean Candidates per S1  : {batch_stats['mean_candidates']:.2f}", flush=True)
    print(f"  - Median Candidates       : {batch_stats['median_candidates']:.0f}", flush=True)
    print(f"  - P95 Candidates          : {batch_stats['p95_candidates']:.0f}", flush=True)
    print(f"  - Maximum Candidates      : {batch_stats['max_candidates']:,}", flush=True)
    print(f"  - Empty Candidate Rate    : {batch_stats['empty_candidate_rate'] * 100:.2f}% ({batch_stats['empty_candidate_count']:,} queries)", flush=True)
    print(f"  - Truncation Rate         : {batch_stats['truncation_rate'] * 100:.2f}% ({batch_stats['total_truncations']:,} queries hit cap)", flush=True)
    print(f"  - Shard Candidates Pruned : {batch_stats.get('total_shard_candidates_pruned', 0):,}", flush=True)
    print(f"Throughput & Resource Profile:", flush=True)
    print(f"  - Query Throughput        : {batch_stats['overall_throughput_qps']:.1f} queries/second", flush=True)
    print(f"  - Elapsed Time            : Total {total_elapsed:.2f}s", flush=True)
    if summary["measured_shard_ram_mb"]:
        print(f"  - Measured Shard RAM (MB) : {summary['measured_shard_ram_mb']}", flush=True)
    if mem_info.get("peak_rss_mb"):
        print(f"  - Peak Process RSS        : {mem_info['peak_rss_mb']:.1f} MB", flush=True)
    elif mem_info.get("rss_mb"):
        print(f"  - Current Process RSS     : {mem_info['rss_mb']:.1f} MB", flush=True)
    print(f"TSV Schema Status           : {'100% VALID' if is_valid else 'FAILED: ' + str(errors)}", flush=True)
    print("=" * 75, flush=True)
    print("[SUCCESS] Validation recall evaluation completed cleanly.\n", flush=True)


def run_test_generation(
    test_dir: Path,
    out_dir: Path,
    sample_s1: Optional[int] = None,
    batch_size: int = BATCH_SIZE_S1_QUERIES,
    shard_size: int = SHARD_SIZE_TARGETS,
    chunksize: int = PILOT_CHUNKSIZE,
    max_candidates: int = BLOCKING_MAX_CANDIDATES_PER_S1,
    min_disk_gb: float = MIN_FREE_DISK_GB,
    min_ram_gb: float = MIN_FREE_RAM_GB,
    resume: bool = True,
    clean_partitions: bool = False,
) -> None:
    """Generate final test candidate_pairs.tsv and pair-level candidate_provenance.tsv."""
    print("\n" + "=" * 75, flush=True)
    print("PHASE 1 STREAMING DISK-BACKED TEST CANDIDATE GENERATION", flush=True)
    print("=" * 75, flush=True)

    t_start = time.time()

    # 1. Benchmark target slice memory footprint (provisional RSS scaling with representative sample and query batch)
    print("Measuring empirical test target index memory scaling (OS RSS)...", flush=True)
    sample_s2 = load_source_file(test_dir / "test_source2.tsv", nrows=2500)
    sample_s3 = load_source_file(test_dir / "test_source3.tsv", nrows=2500)
    sample_targets = pd.concat([sample_s2, sample_s3], ignore_index=True)
    sample_s1_df = load_source_file(test_dir / "test_source1.tsv", nrows=200)

    bench = benchmark_shard_memory_rss(
        sample_df=sample_targets,
        query_batch_df=sample_s1_df,
        shard_size_targets=shard_size,
    )
    measured_bpt = bench["rss_bytes_per_target"]
    print(
        f"  -> [PROVISIONAL ESTIMATE] Measured {measured_bpt:.1f} bytes/target from sample (n={len(sample_targets):,}) "
        f"with query batch (n={len(sample_s1_df):,}).\n"
        f"     Provisional estimated shard footprint: {bench['estimated_shard_rss_gb']:.2f} GB per {shard_size:,} shard ({RAM_SAFETY_MARGIN}x safety margin).\n"
        f"     (Note: actual peak RSS is monitored and logged per resident shard during execution.)",
        flush=True,
    )

    s1_test_count = sample_s1 if sample_s1 is not None else 1730000

    # 2. Pre-flight resource safety check
    is_safe, res_info, msg = check_resource_headroom(
        out_dir,
        min_disk_gb=min_disk_gb,
        min_ram_gb=min_ram_gb,
        shard_size=shard_size,
        measured_bytes_per_target=measured_bpt,
        safety_margin=RAM_SAFETY_MARGIN,
        estimated_s1_count=s1_test_count,
        estimated_target_count=10000000,
        per_shard_candidate_cap=BLOCKING_PER_SHARD_CANDIDATE_CAP,
        max_candidates=max_candidates,
        sample_targets_df=sample_targets,
        sample_s1_df=sample_s1_df,
    )
    print(f"[RESOURCE CHECK] {msg}", flush=True)
    if "disk_breakdown_gib" in res_info:
        db = res_info["disk_breakdown_gib"]
        print(
            f"  -> Disk Breakdown (GiB): Target Shards={db['target_shards_gib']:.2f} GiB, S1 Batches={db['s1_batches_gib']:.2f} GiB,\n"
            f"     Intermediate: Conservative={db['conservative_intermediate_gib']:.2f} GiB (Empirical={db['empirical_intermediate_gib']:.2f} GiB),\n"
            f"     Merged Batches: Conservative={db['merged_batches_conservative_gib']:.2f} GiB (Empirical={db['merged_batches_empirical_gib']:.2f} GiB),\n"
            f"     Final Outputs: Conservative={db['final_outputs_conservative_gib']:.2f} GiB (Empirical={db['final_outputs_empirical_gib']:.2f} GiB),\n"
            f"     Merge Coexistence / Tmp: {db['tmp_coexistence_conservative_gib']:.2f} GiB (Empirical={db['tmp_coexistence_empirical_gib']:.2f} GiB)",
            flush=True,
        )
    if not is_safe:
        raise RuntimeError(f"Resource safety check failed: {msg}")

    # 3. Execute single-shard-resident disk-backed retrieval pipeline
    partitions_dir = out_dir / "partitions"
    manifest_path = out_dir / "test_manifest.json"

    batch_stats = run_disk_backed_retrieval_pipeline(
        s1_path=test_dir / "test_source1.tsv",
        s2_path=test_dir / "test_source2.tsv",
        s3_path=test_dir / "test_source3.tsv",
        partitions_dir=partitions_dir,
        manifest_path=manifest_path,
        mode="test",
        split_file=None,
        filter_s1_ids=None,
        sample_s1=sample_s1,
        gt_lookup=None,
        batch_size=batch_size,
        shard_size=shard_size,
        chunksize=chunksize,
        max_candidates=max_candidates,
        per_shard_candidate_cap=BLOCKING_PER_SHARD_CANDIDATE_CAP,
        resume=resume,
    )

    # 4. Merge partition files into final candidate files
    final_pairs = out_dir / "candidate_pairs.tsv"
    final_prov = out_dir / "candidate_provenance.tsv"
    merge_info = merge_candidate_partitions(
        partitions_dir=partitions_dir,
        output_pairs_path=final_pairs,
        output_prov_path=final_prov,
        clean_partitions=clean_partitions,
    )

    # 5. Stream-validate output TSV against test Source1 and Source2/Source3 universe
    is_valid, errors = validate_submission_streaming(
        submission_path=final_pairs,
        s1_source_path=test_dir / "test_source1.tsv",
        target_source_paths=[test_dir / "test_source2.tsv", test_dir / "test_source3.tsv"],
        sample_s1=sample_s1,
        is_candidate_file=True,
    )

    total_elapsed = time.time() - t_start
    mem_info = get_process_memory_mb()

    # 6. Write test summary JSON
    summary = {
        "mode": "test",
        "total_s1_processed": batch_stats["total_s1_processed"],
        "total_candidate_pairs": batch_stats["total_candidate_pairs"],
        "candidate_distribution": {
            "mean": batch_stats["mean_candidates"],
            "median": batch_stats["median_candidates"],
            "p90": batch_stats["p90_candidates"],
            "p95": batch_stats["p95_candidates"],
            "p99": batch_stats["p99_candidates"],
            "max": batch_stats["max_candidates"],
            "empty_candidate_count": batch_stats["empty_candidate_count"],
            "empty_candidate_rate": batch_stats["empty_candidate_rate"],
            "truncation_count": batch_stats["total_truncations"],
            "truncation_rate": batch_stats["truncation_rate"],
        },
        "timing_and_throughput": {
            "total_elapsed_seconds": round(total_elapsed, 2),
            "overall_throughput_qps": batch_stats["overall_throughput_qps"],
        },
        "memory_mb": mem_info,
        "validation_schema_valid": is_valid,
        "validation_schema_errors": errors,
    }

    summary_path = out_dir / "test_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # 7. Print formatted console report
    print("\n" + "=" * 75, flush=True)
    print("TEST CANDIDATE OUTPUT SUMMARY (DISK-BACKED)", flush=True)
    print("=" * 75, flush=True)
    print(f"Summary JSON Written        : {summary_path.resolve()}", flush=True)
    print(f"Candidate Pairs TSV         : {final_pairs.resolve()}", flush=True)
    print(f"Provenance Pairs TSV        : {final_prov.resolve()}", flush=True)
    print(f"Total Test S1 Entities      : {batch_stats['total_s1_processed']:,}", flush=True)
    print(f"Total Candidate Pairs       : {batch_stats['total_candidate_pairs']:,}", flush=True)
    print(f"Mean Candidates per S1      : {batch_stats['mean_candidates']:.2f}", flush=True)
    print(f"Median Candidates           : {batch_stats['median_candidates']:.0f}", flush=True)
    print(f"P95 Candidates              : {batch_stats['p95_candidates']:.0f}", flush=True)
    print(f"Max Candidates              : {batch_stats['max_candidates']:,}", flush=True)
    print(f"Empty Candidate Rate        : {batch_stats['empty_candidate_rate'] * 100:.2f}% ({batch_stats['empty_candidate_count']:,} queries)", flush=True)
    print(f"Truncation Count            : {batch_stats['total_truncations']:,} ({batch_stats['truncation_rate'] * 100:.2f}%)", flush=True)
    print(f"Throughput & Resource Profile:", flush=True)
    print(f"  - Query Throughput        : {batch_stats['overall_throughput_qps']:.1f} queries/second", flush=True)
    print(f"  - Elapsed Time            : Total {total_elapsed:.2f}s", flush=True)
    if mem_info.get("peak_rss_mb"):
        print(f"  - Peak Process RSS        : {mem_info['peak_rss_mb']:.1f} MB", flush=True)
    print(f"TSV Validation Status       : {'100% VALID' if is_valid else 'FAILED: ' + str(errors)}", flush=True)
    print("=" * 75, flush=True)
    print(f"Test candidate generation completed in {total_elapsed:.2f}s total.\n", flush=True)


def run_pilot(
    train_dir: Path,
    splits_dir: Path,
    out_dir: Path,
    scan_rows: int = PILOT_SCAN_ROWS,
    chunksize: int = PILOT_CHUNKSIZE,
    seed: int = PILOT_SEED,
    sample_s1: int = PILOT_SAMPLE_S1,
    sample_target: int = PILOT_SAMPLE_TARGET,
    max_candidates: int = BLOCKING_MAX_CANDIDATES_PER_S1,
) -> None:
    """Execute bounded pilot run for plumbing, resource, and candidate pipeline verification."""
    print("\n" + "=" * 75, flush=True)
    print("PHASE 1 BOUNDED PILOT EXECUTION", flush=True)
    print("=" * 75, flush=True)
    print("NOTICE: Pilot mode runs bounded sampling for resource and plumbing verification.", flush=True)
    print("Candidate recall is NOT calculated in pilot mode (target-sampling is non-representative).", flush=True)
    print(f"Scan Window     : First {scan_rows:,} rows per source file", flush=True)
    print(f"Chunk Size      : {chunksize:,} rows", flush=True)
    print(f"Random Seed     : {seed}", flush=True)
    print(f"Target Samples  : S1 Queries = {sample_s1:,}, S2 Targets = {sample_target:,}, S3 Targets = {sample_target:,}", flush=True)
    print("=" * 75, flush=True)

    t_start = time.time()

    # 1. Load saved validation split IDs (read-only)
    val_txt = splits_dir / "validation_s1_ids.txt"
    if not val_txt.exists():
        raise FileNotFoundError(
            f"Validation split file not found at '{val_txt.resolve()}'. "
            "Please ensure output/splits/validation_s1_ids.txt exists."
        )

    with open(val_txt, "r", encoding="utf-8") as f:
        val_s1_ids = {line.strip() for line in f if line.strip()}
    print(f"Loaded {len(val_s1_ids):,} saved validation S1 IDs from '{val_txt.name}'.", flush=True)

    # 2. Stream-scan and sample Source 1 restricted to validation split IDs
    print(f"\n1. Scanning 'train_source1.tsv' (up to {scan_rows:,} rows, filtered on validation IDs)...", flush=True)
    t_s1 = time.time()
    s1_raw, s1_scanned = scan_and_sample_source_file(
        path=train_dir / "train_source1.tsv",
        scan_rows=scan_rows,
        sample_size=sample_s1,
        chunksize=chunksize,
        seed=seed,
        filter_ids=val_s1_ids,
    )
    s1_load_time = time.time() - t_s1
    print(f"   -> Selected {len(s1_raw):,} validation S1 queries (scanned {s1_scanned:,} rows in {s1_load_time:.2f}s).", flush=True)

    # 3. Stream-scan and sample Source 2
    print(f"\n2. Scanning 'train_source2.tsv' (up to {scan_rows:,} rows, deterministic hash sample)...", flush=True)
    t_s2 = time.time()
    s2_raw, s2_scanned = scan_and_sample_source_file(
        path=train_dir / "train_source2.tsv",
        scan_rows=scan_rows,
        sample_size=sample_target,
        chunksize=chunksize,
        seed=seed,
    )
    s2_load_time = time.time() - t_s2
    print(f"   -> Selected {len(s2_raw):,} S2 target entities (scanned {s2_scanned:,} rows in {s2_load_time:.2f}s).", flush=True)

    # 4. Stream-scan and sample Source 3
    print(f"\n3. Scanning 'train_source3.tsv' (up to {scan_rows:,} rows, deterministic hash sample)...", flush=True)
    t_s3 = time.time()
    s3_raw, s3_scanned = scan_and_sample_source_file(
        path=train_dir / "train_source3.tsv",
        scan_rows=scan_rows,
        sample_size=sample_target,
        chunksize=chunksize,
        seed=seed,
    )
    s3_load_time = time.time() - t_s3
    print(f"   -> Selected {len(s3_raw):,} S3 target entities (scanned {s3_scanned:,} rows in {s3_load_time:.2f}s).", flush=True)

    data_scan_time = time.time() - t_start

    # 5. Normalization
    print("\n4. Normalizing bounded DataFrames...", flush=True)
    t_norm = time.time()
    s1 = normalize_dataframe(s1_raw, inplace=True)
    s2 = normalize_dataframe(s2_raw, inplace=True)
    s3 = normalize_dataframe(s3_raw, inplace=True)
    norm_time = time.time() - t_norm
    print(f"   -> Normalization completed in {norm_time:.2f}s.", flush=True)

    # 6. Candidate Generation
    total_targets = len(s2) + len(s3)
    print(f"\n5. Generating candidate union for {len(s1):,} S1 queries against {total_targets:,} targets...", flush=True)
    t_gen = time.time()
    cands_map, prov_records, stats = generate_candidate_union(
        s1_df=s1,
        s2_df=s2,
        s3_df=s3,
        max_candidates=max_candidates,
    )
    gen_time = time.time() - t_gen
    throughput = len(s1) / gen_time if gen_time > 0 else 0.0
    print(f"   -> Candidate generation completed in {gen_time:.2f}s ({throughput:.1f} queries/s).", flush=True)

    # 7. Serialize pilot outputs
    pilot_pairs_path = out_dir / "pilot_candidate_pairs.tsv"
    pilot_prov_path = out_dir / "pilot_candidate_provenance.tsv"
    print(f"\n6. Writing pilot candidate files...", flush=True)
    write_candidate_outputs(
        candidates_map=cands_map,
        provenance_records=prov_records,
        ordered_s1_ids=s1["entity_id"].tolist(),
        candidate_pairs_path=pilot_pairs_path,
        provenance_path=pilot_prov_path,
    )

    # 8. Validate output TSV schema
    target_ids = set(s2["entity_id"]).union(set(s3["entity_id"]))
    is_valid, errors = validate_submission_file(
        submission_path=pilot_pairs_path,
        expected_s1_ids=set(s1["entity_id"]),
        valid_target_ids=target_ids,
        is_candidate_file=True,
    )

    total_elapsed = time.time() - t_start
    mem_info = get_process_memory_mb()

    # 9. Build summary metrics
    summary = {
        "mode": "pilot",
        "notice": (
            "Pilot mode uses prefix-window and sampled targets. Candidate recall is NOT "
            "computed because target sampling is not representative of full-corpus recall."
        ),
        "configuration": {
            "scan_rows": scan_rows,
            "chunksize": chunksize,
            "seed": seed,
            "sample_s1_requested": sample_s1,
            "sample_target_requested": sample_target,
            "max_candidates": max_candidates,
        },
        "sampled_counts": {
            "source1_validation_queries": len(s1),
            "source1_scanned_rows": s1_scanned,
            "source2_targets": len(s2),
            "source2_scanned_rows": s2_scanned,
            "source3_targets": len(s3),
            "source3_scanned_rows": s3_scanned,
            "total_targets": total_targets,
            "total_candidate_pairs": len(prov_records),
        },
        "candidate_distribution": {
            "mean": stats["mean_candidates"],
            "median": stats["median_candidates"],
            "p90": stats["p90_candidates"],
            "p95": stats["p95_candidates"],
            "p99": stats["p99_candidates"],
            "max": stats["max_candidates"],
            "empty_candidate_count": stats["empty_candidate_count"],
            "empty_candidate_rate": stats["empty_candidate_rate"],
            "truncation_count": stats["truncation_count"],
            "truncation_rate": stats["truncation_rate"],
        },
        "timing_seconds": {
            "data_scan_and_load": round(data_scan_time, 3),
            "normalization": round(norm_time, 3),
            "candidate_generation": round(gen_time, 3),
            "total_elapsed": round(total_elapsed, 3),
            "query_throughput_qps": round(throughput, 1),
        },
        "memory_mb": {
            "current_rss_mb": round(mem_info.get("rss_mb", 0.0), 2),
            "peak_rss_mb": round(mem_info.get("peak_rss_mb", 0.0), 2),
            "tracemalloc_peak_mb": round(mem_info.get("tracemalloc_peak_mb", 0.0), 2),
        },
        "validation_schema_valid": is_valid,
        "validation_schema_errors": errors,
    }

    # 10. Write summary JSON
    summary_path = out_dir / "pilot_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # 11. Print formatted console report
    print("\n" + "#" * 75, flush=True)
    print("# PILOT EXECUTION SUMMARY", flush=True)
    print("#" * 75, flush=True)
    print(f"Summary JSON Written        : {summary_path.resolve()}", flush=True)
    print(f"Validation S1 Queries       : {len(s1):,} (scanned {s1_scanned:,} rows)", flush=True)
    print(f"Target Entities (S2 + S3)   : {total_targets:,} ({len(s2):,} S2 + {len(s3):,} S3)", flush=True)
    print(f"Candidate Pairs Generated   : {len(prov_records):,}", flush=True)
    print(f"Query Throughput            : {throughput:.1f} queries/second", flush=True)
    print(f"Candidate Count / Query     : Mean={stats['mean_candidates']:.1f}, Median={stats['median_candidates']:.0f}, P95={stats['p95_candidates']:.0f}, Max={stats['max_candidates']}", flush=True)
    print(f"Empty Candidate Rate        : {stats['empty_candidate_rate'] * 100:.2f}% ({stats['empty_candidate_count']:,} queries)", flush=True)
    print(f"Candidate Truncation Rate   : {stats['truncation_rate'] * 100:.2f}% ({stats['truncation_count']:,} queries hit limit)", flush=True)
    print(f"Elapsed Time                : Total {total_elapsed:.2f}s (Load: {data_scan_time:.2f}s, Norm: {norm_time:.2f}s, Gen: {gen_time:.2f}s)", flush=True)
    if mem_info.get("peak_rss_mb"):
        print(f"Peak Process Memory (RSS)   : {mem_info['peak_rss_mb']:.1f} MB", flush=True)
    elif mem_info.get("rss_mb"):
        print(f"Current Process Memory (RSS): {mem_info['rss_mb']:.1f} MB", flush=True)
    if mem_info.get("tracemalloc_peak_mb"):
        print(f"Tracemalloc Peak Heap       : {mem_info['tracemalloc_peak_mb']:.1f} MB", flush=True)
    print(f"TSV Schema Validation       : {'100% VALID' if is_valid else 'ERRORS: ' + str(errors)}", flush=True)
    print("=" * 75, flush=True)
    print("[SUCCESS] Bounded pilot completed successfully.\n", flush=True)


def main() -> None:
    """Main CLI entry point for Phase 1 candidate generation on Kaggle."""
    parser = argparse.ArgumentParser(description="Kaggle Phase 1 Candidate Generation & Validation Evaluation")
    parser.add_argument(
        "--mode",
        choices=["pilot", "smoke", "eval_val", "test", "all"],
        default="pilot",
        help="Mode: pilot (bounded plumbing/resource verification), smoke (fast check), eval_val (disk-backed validation recall), test (disk-backed test candidates), all (both)",
    )
    parser.add_argument("--data-dir", type=str, default=None, help="Path to data directory")
    parser.add_argument("--splits-dir", type=str, default=None, help="Path to splits directory (output/splits)")
    parser.add_argument("--output-dir", type=str, default=None, help="Path to output directory")
    parser.add_argument("--pilot-scan-rows", type=int, default=PILOT_SCAN_ROWS, help="Initial rows to scan per source TSV in pilot mode")
    parser.add_argument("--pilot-chunksize", type=int, default=PILOT_CHUNKSIZE, help="Chunk size for streaming TSV reads in pilot mode")
    parser.add_argument("--pilot-seed", type=int, default=PILOT_SEED, help="Seed for deterministic ID-hash sampling in pilot mode")
    parser.add_argument("--sample-s1", type=int, default=None, help="Sample limit for S1 queries (default: 1000 in pilot mode)")
    parser.add_argument("--sample-target", type=int, default=None, help="Sample limit for S2/S3 targets (default: 5000 in pilot mode)")
    parser.add_argument("--batch-size-s1", type=int, default=BATCH_SIZE_S1_QUERIES, help="Batch size for S1 queries in streaming pipeline")
    parser.add_argument("--shard-size-targets", type=int, default=SHARD_SIZE_TARGETS, help="Target count per index shard")
    parser.add_argument("--max-candidates", type=int, default=BLOCKING_MAX_CANDIDATES_PER_S1, help="Max candidates per S1")
    parser.add_argument("--min-disk-gb", type=float, default=MIN_FREE_DISK_GB, help="Minimum free disk required in GB")
    parser.add_argument("--min-ram-gb", type=float, default=MIN_FREE_RAM_GB, help="Minimum available RAM required in GB")
    parser.add_argument("--no-resume", action="store_true", help="Disable batch resume from existing manifest")
    parser.add_argument("--clean-partitions", action="store_true", help="Delete partition files after final merge")
    parser.add_argument("--track-memory", action="store_true", help="Enable memory tracking")

    args = parser.parse_args()

    if args.track_memory:
        tracemalloc.start()

    train_dir, test_dir, splits_dir, out_dir = resolve_paths(
        custom_data_dir=args.data_dir,
        custom_splits_dir=args.splits_dir,
        custom_output_dir=args.output_dir,
    )

    print("\n" + "#" * 75, flush=True)
    print("# AMAZON BUSINESS ENTITY RESOLUTION - PHASE 1 CANDIDATE GENERATOR", flush=True)
    print("#" * 75, flush=True)
    print(f"Execution Mode   : {args.mode.upper()}", flush=True)
    print(f"Train Directory  : {train_dir.resolve()}", flush=True)
    print(f"Test Directory   : {test_dir.resolve()}", flush=True)
    print(f"Splits Directory : {splits_dir.resolve()}", flush=True)
    print(f"Output Directory : {out_dir.resolve()}", flush=True)
    print(f"Max Candidates   : {args.max_candidates}", flush=True)
    print("#" * 75 + "\n", flush=True)

    if args.mode == "pilot":
        sample_s1 = args.sample_s1 if args.sample_s1 is not None else PILOT_SAMPLE_S1
        sample_target = args.sample_target if args.sample_target is not None else PILOT_SAMPLE_TARGET
        run_pilot(
            train_dir=train_dir,
            splits_dir=splits_dir,
            out_dir=out_dir,
            scan_rows=args.pilot_scan_rows,
            chunksize=args.pilot_chunksize,
            seed=args.pilot_seed,
            sample_s1=sample_s1,
            sample_target=sample_target,
            max_candidates=args.max_candidates,
        )

    elif args.mode == "smoke":
        sample_size = args.sample_s1 if args.sample_s1 is not None else 1000
        sample_target = args.sample_target if args.sample_target is not None else 5000
        run_smoke_test(
            train_dir=train_dir,
            out_dir=out_dir,
            sample_size=sample_size,
            sample_target=sample_target,
            max_candidates=args.max_candidates,
        )

    elif args.mode == "eval_val":
        run_validation_evaluation(
            train_dir=train_dir,
            splits_dir=splits_dir,
            out_dir=out_dir,
            sample_s1=args.sample_s1,
            batch_size=args.batch_size_s1,
            shard_size=args.shard_size_targets,
            chunksize=args.pilot_chunksize,
            max_candidates=args.max_candidates,
            min_disk_gb=args.min_disk_gb,
            min_ram_gb=args.min_ram_gb,
            resume=not args.no_resume,
            clean_partitions=args.clean_partitions,
        )

    elif args.mode == "test":
        run_test_generation(
            test_dir=test_dir,
            out_dir=out_dir,
            sample_s1=args.sample_s1,
            batch_size=args.batch_size_s1,
            shard_size=args.shard_size_targets,
            chunksize=args.pilot_chunksize,
            max_candidates=args.max_candidates,
            min_disk_gb=args.min_disk_gb,
            min_ram_gb=args.min_ram_gb,
            resume=not args.no_resume,
            clean_partitions=args.clean_partitions,
        )

    elif args.mode == "all":
        run_validation_evaluation(
            train_dir=train_dir,
            splits_dir=splits_dir,
            out_dir=out_dir,
            sample_s1=args.sample_s1,
            batch_size=args.batch_size_s1,
            shard_size=args.shard_size_targets,
            chunksize=args.pilot_chunksize,
            max_candidates=args.max_candidates,
            min_disk_gb=args.min_disk_gb,
            min_ram_gb=args.min_ram_gb,
            resume=not args.no_resume,
            clean_partitions=args.clean_partitions,
        )
        gc.collect()
        run_test_generation(
            test_dir=test_dir,
            out_dir=out_dir,
            sample_s1=args.sample_s1,
            batch_size=args.batch_size_s1,
            shard_size=args.shard_size_targets,
            chunksize=args.pilot_chunksize,
            max_candidates=args.max_candidates,
            min_disk_gb=args.min_disk_gb,
            min_ram_gb=args.min_ram_gb,
            resume=not args.no_resume,
            clean_partitions=args.clean_partitions,
        )


if __name__ == "__main__":
    main()


