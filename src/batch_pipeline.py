"""Streaming batch pipeline for disk-backed candidate generation and incremental validation recall.

Processes Source 1 entities in bounded, resumable query batches, writes partition files
directly to disk, and computes exact validation recall without holding multiple index shards in RAM.
"""

import gc
import hashlib
import json
import math
import os
import shutil
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd

from src.config import (
    BATCH_SIZE_S1_QUERIES,
    BLOCKING_FUZZY_MIN_CANDS,
    BLOCKING_FUZZY_TOP_K,
    BLOCKING_MAX_ADDR_KEY_FREQ,
    BLOCKING_MAX_CANDIDATES_PER_S1,
    BLOCKING_MAX_PAIR_TOKEN_FREQ,
    BLOCKING_MAX_POSTING_LEN,
    BLOCKING_MAX_RARE_TOKENS_PER_S1,
    BLOCKING_MAX_TOKEN_DOC_FREQ,
    BLOCKING_MIN_TOKEN_LEN,
    BLOCKING_PER_SHARD_CANDIDATE_CAP,
    BLOCKING_SAFETY_NET_MAX_BUCKET,
    BLOCKING_SOFT_COUNTRY_MODE,
    ESTIMATED_BYTES_PER_INTERMEDIATE_ROW,
    LSH_MAX_BUCKET_SIZE,
    LSH_MAX_CANDIDATES,
    LSH_NUM_BANDS,
    LSH_NUM_PERMUTATIONS,
    LSH_SHINGLE_N,
    MANIFEST_PATH,
    MIN_FREE_DISK_GB,
    MIN_FREE_RAM_GB,
    PARTITIONS_DIR,
    PILOT_CHUNKSIZE,
    RAM_SAFETY_MARGIN,
    SHARD_SIZE_TARGETS,
)
from src.normalize import normalize_dataframe
from src.schemas import COL_CANDIDATE_IDS, COL_SOURCE1_ID
from src.sharded_index import (
    ShardedTargetIndex,
    TargetIndexShard,
    benchmark_shard_memory_rss,
    get_process_rss_bytes,
)
from utils.validate_submission import validate_submission_file


def get_available_ram_bytes() -> Optional[int]:
    """Determine available system RAM across Linux, Windows, and macOS without hardcoded fallbacks.
    
    Returns:
        Optional[int]: Available memory in bytes, or None if unmeasurable.
    """
    # 1. psutil
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:
        pass

    # 2. Linux /proc/meminfo
    try:
        if os.path.exists("/proc/meminfo"):
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        kb = int(line.split()[1])
                        return kb * 1024
    except Exception:
        pass

    # 3. Windows GlobalMemoryStatusEx via ctypes
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", wintypes.DWORD),
                    ("dwMemoryLoad", wintypes.DWORD),
                    ("ullTotalPhys", ctypes.c_uint64),
                    ("ullAvailPhys", ctypes.c_uint64),
                    ("ullTotalPageFile", ctypes.c_uint64),
                    ("ullAvailPageFile", ctypes.c_uint64),
                    ("ullTotalVirtual", ctypes.c_uint64),
                    ("ullAvailVirtual", ctypes.c_uint64),
                    ("ullAvailExtendedVirtual", ctypes.c_uint64),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return int(stat.ullAvailPhys)
    except Exception:
        pass

    return None


def get_available_ram_gb() -> Optional[float]:
    """Return available system RAM in gigabytes (GB), or None if unmeasurable."""
    b = get_available_ram_bytes()
    return round(b / (1024 ** 3), 2) if b is not None else None


def check_resource_headroom(
    target_dir: Union[Path, str],
    min_disk_gb: float = MIN_FREE_DISK_GB,
    min_ram_gb: float = MIN_FREE_RAM_GB,
    shard_size: int = SHARD_SIZE_TARGETS,
    measured_bytes_per_target: Optional[float] = None,
    safety_margin: float = RAM_SAFETY_MARGIN,
    estimated_s1_count: int = 330000,
    estimated_target_count: int = 10300000,
    per_shard_candidate_cap: int = BLOCKING_PER_SHARD_CANDIDATE_CAP,
    max_candidates: int = BLOCKING_MAX_CANDIDATES_PER_S1,
) -> Tuple[bool, Dict[str, Any], str]:
    """Check free disk space and available system RAM against required shard headroom.

    Args:
        target_dir: Output or scratch directory to check disk usage.
        min_disk_gb: Minimum required free disk in gigabytes.
        min_ram_gb: Minimum required available RAM in gigabytes.
        shard_size: Number of target records per shard.
        measured_bytes_per_target: Measured index peak RSS bytes per target record.
        safety_margin: Multiplier for conservative RAM headroom.
        estimated_s1_count: Estimated count of S1 queries.
        estimated_target_count: Estimated total target records.
        per_shard_candidate_cap: Max candidate matches written per shard per query.
        max_candidates: Final candidate upper bound.

    Returns:
        Tuple[bool, Dict[str, Any], str]: (is_safe, resource_dict, status_message).
    """
    path = Path(target_dir)
    path.mkdir(parents=True, exist_ok=True)

    # 1. Disk usage calculation
    total, used, free = shutil.disk_usage(path)
    free_disk_gb = free / (1024 ** 3)

    if shard_size >= 100000:
        num_shards = max(1, math.ceil(estimated_target_count / max(shard_size, 1)))
        est_s1 = estimated_s1_count
    else:
        num_shards = 4
        est_s1 = min(estimated_s1_count, 1000)

    # Estimate intermediate disk rows (bounded by per_shard_candidate_cap)
    est_intermediate_rows = est_s1 * num_shards * min(per_shard_candidate_cap, 50)
    est_intermediate_gb = (est_intermediate_rows * ESTIMATED_BYTES_PER_INTERMEDIATE_ROW) / (1024 ** 3)
    est_final_gb = (est_s1 * max_candidates * 40) / (1024 ** 3)
    required_disk_gb = max(min_disk_gb, round(est_intermediate_gb + est_final_gb + 0.5, 2))

    # 2. RAM availability
    avail_ram_bytes = get_available_ram_bytes()
    avail_ram_gb = round(avail_ram_bytes / (1024 ** 3), 2) if avail_ram_bytes is not None else None

    if measured_bytes_per_target is None:
        # Fails safely if RAM cannot be estimated from a measurement
        return False, {
            "free_disk_gb": round(free_disk_gb, 2),
            "available_ram_gb": avail_ram_gb,
            "required_disk_gb": required_disk_gb,
            "required_ram_gb": None,
        }, "Resource check failed: Measured bytes per target was not provided. Run memory benchmark before proceeding."

    # Dynamic RAM required for 1 resident shard + query buffer + safety margin + OS headroom
    shard_ram_gb = (shard_size * measured_bytes_per_target * safety_margin) / (1024 ** 3)
    required_ram_gb = max(min_ram_gb, round(shard_ram_gb + 0.5, 2))

    res: Dict[str, Any] = {
        "free_disk_gb": round(free_disk_gb, 2),
        "required_disk_gb": required_disk_gb,
        "available_ram_gb": avail_ram_gb,
        "required_ram_gb": required_ram_gb,
        "measured_bytes_per_target": round(measured_bytes_per_target, 2),
    }

    if free_disk_gb < required_disk_gb:
        msg = (
            f"Insufficient disk space in '{path.resolve()}': {free_disk_gb:.2f} GB free "
            f"(estimated required: {required_disk_gb:.2f} GB for intermediate partitions and final candidate tables)."
        )
        return False, res, msg

    if avail_ram_gb is None:
        msg = "Resource check failed: Available system RAM could not be measured and no safe metric was available."
        return False, res, msg

    if avail_ram_gb < required_ram_gb:
        msg = (
            f"Insufficient available RAM: {avail_ram_gb:.2f} GB available "
            f"(minimum required: {required_ram_gb:.2f} GB based on measured {measured_bytes_per_target:.1f} bytes/target "
            f"for shard size {shard_size:,} with {safety_margin:.1f}x safety margin)."
        )
        return False, res, msg

    msg = (
        f"Resource check passed: {free_disk_gb:.2f} GB disk free (req: {required_disk_gb:.2f} GB), "
        f"{avail_ram_gb:.2f} GB RAM available (req: {required_ram_gb:.2f} GB)."
    )
    return True, res, msg


def compute_manifest_fingerprint(
    mode: str,
    input_files: List[Union[Path, str]],
    split_file: Optional[Union[Path, str]] = None,
    filter_s1_ids: Optional[Set[str]] = None,
    sample_s1: Optional[int] = None,
    shard_size: int = SHARD_SIZE_TARGETS,
    batch_size: int = BATCH_SIZE_S1_QUERIES,
    max_candidates: int = BLOCKING_MAX_CANDIDATES_PER_S1,
    per_shard_cap: int = BLOCKING_PER_SHARD_CANDIDATE_CAP,
    config_dict: Optional[Dict[str, Any]] = None,
) -> str:
    """Compute deterministic cryptographic fingerprint of all pipeline inputs and hyperparameters."""
    hasher = hashlib.sha256()

    # 1. Execution mode
    hasher.update(f"mode:{mode}".encode("utf-8"))

    # 2. Input file metadata (path, size, mtime)
    for p in sorted(input_files, key=lambda x: str(x)):
        p_path = Path(p)
        if p_path.exists():
            stat = p_path.stat()
            hasher.update(f"{p_path.name}:{stat.st_size}:{stat.st_mtime}".encode("utf-8"))
        else:
            hasher.update(f"{p_path.name}:missing".encode("utf-8"))

    # 3. Split file metadata and filtered S1 ID hash
    if split_file:
        s_path = Path(split_file)
        if s_path.exists():
            stat = s_path.stat()
            hasher.update(f"split:{s_path.name}:{stat.st_size}:{stat.st_mtime}".encode("utf-8"))

    if filter_s1_ids is not None:
        sorted_ids = sorted(filter_s1_ids)
        id_hash = hashlib.sha256(",".join(sorted_ids).encode("utf-8")).hexdigest()
        hasher.update(f"filter_s1_hash:{id_hash}".encode("utf-8"))
        hasher.update(f"filter_s1_count:{len(filter_s1_ids)}".encode("utf-8"))

    if sample_s1 is not None:
        hasher.update(f"sample_s1:{sample_s1}".encode("utf-8"))

    # 4. Execution dimensions
    hasher.update(f"shard_size:{shard_size}".encode("utf-8"))
    hasher.update(f"batch_size:{batch_size}".encode("utf-8"))
    hasher.update(f"max_candidates:{max_candidates}".encode("utf-8"))
    hasher.update(f"per_shard_cap:{per_shard_cap}".encode("utf-8"))

    # 5. Retrieval hyperparameters
    cfg = config_dict or {
        "BLOCKING_MAX_TOKEN_DOC_FREQ": BLOCKING_MAX_TOKEN_DOC_FREQ,
        "BLOCKING_MAX_POSTING_LEN": BLOCKING_MAX_POSTING_LEN,
        "BLOCKING_MAX_RARE_TOKENS_PER_S1": BLOCKING_MAX_RARE_TOKENS_PER_S1,
        "BLOCKING_MIN_TOKEN_LEN": BLOCKING_MIN_TOKEN_LEN,
        "BLOCKING_MAX_PAIR_TOKEN_FREQ": BLOCKING_MAX_PAIR_TOKEN_FREQ,
        "LSH_NUM_PERMUTATIONS": LSH_NUM_PERMUTATIONS,
        "LSH_NUM_BANDS": LSH_NUM_BANDS,
        "LSH_SHINGLE_N": LSH_SHINGLE_N,
        "LSH_MAX_BUCKET_SIZE": LSH_MAX_BUCKET_SIZE,
        "LSH_MAX_CANDIDATES": LSH_MAX_CANDIDATES,
        "BLOCKING_MAX_ADDR_KEY_FREQ": BLOCKING_MAX_ADDR_KEY_FREQ,
        "BLOCKING_FUZZY_MIN_CANDS": BLOCKING_FUZZY_MIN_CANDS,
        "BLOCKING_FUZZY_TOP_K": BLOCKING_FUZZY_TOP_K,
        "BLOCKING_SAFETY_NET_MAX_BUCKET": BLOCKING_SAFETY_NET_MAX_BUCKET,
        "BLOCKING_SOFT_COUNTRY_MODE": BLOCKING_SOFT_COUNTRY_MODE,
    }
    hasher.update(json.dumps(cfg, sort_keys=True).encode("utf-8"))

    return hasher.hexdigest()


def partition_target_sources(
    s2_path: Union[Path, str],
    s3_path: Union[Path, str],
    target_shards_dir: Path,
    shard_size: int = SHARD_SIZE_TARGETS,
    chunksize: int = PILOT_CHUNKSIZE,
) -> List[Path]:
    """Stream-normalize and slice target records into disk-backed target shard files with atomic writes."""
    target_shards_dir.mkdir(parents=True, exist_ok=True)
    shard_paths: List[Path] = []

    curr_shard_idx = 0
    curr_shard_count = 0
    curr_shard_file = target_shards_dir / f"target_shard_{curr_shard_idx:04d}.tsv"
    curr_tmp_file = curr_shard_file.with_suffix(".tsv.tmp")
    f_out = open(curr_tmp_file, "w", encoding="utf-8", newline="")
    f_out.write("entity_id\tbusiness_name_norm\tbusiness_address_norm\tcountry_norm\tsource_dataset\n")

    try:
        for source_label, file_path in [("source2", Path(s2_path)), ("source3", Path(s3_path))]:
            if not file_path.exists():
                raise FileNotFoundError(f"Target file not found at: {file_path.resolve()}")

            print(f"Partitioning targets from '{file_path.name}' ({source_label})...", flush=True)
            reader = pd.read_csv(
                file_path,
                sep="\t",
                dtype=str,
                keep_default_na=False,
                chunksize=chunksize,
            )

            try:
                for chunk in reader:
                    chunk_norm = normalize_dataframe(chunk, inplace=True)
                    eids = chunk_norm["entity_id"].tolist()
                    names = chunk_norm["business_name_norm"].tolist()
                    addrs = chunk_norm["business_address_norm"].tolist()
                    ctys = chunk_norm["country_norm"].tolist() if "country_norm" in chunk_norm.columns else [""] * len(chunk_norm)

                    for eid, name, addr, cty in zip(eids, names, addrs, ctys):
                        if curr_shard_count >= shard_size:
                            f_out.close()
                            os.replace(curr_tmp_file, curr_shard_file)
                            shard_paths.append(curr_shard_file)

                            curr_shard_idx += 1
                            curr_shard_count = 0
                            curr_shard_file = target_shards_dir / f"target_shard_{curr_shard_idx:04d}.tsv"
                            curr_tmp_file = curr_shard_file.with_suffix(".tsv.tmp")
                            f_out = open(curr_tmp_file, "w", encoding="utf-8", newline="")
                            f_out.write("entity_id\tbusiness_name_norm\tbusiness_address_norm\tcountry_norm\tsource_dataset\n")

                        f_out.write(f"{eid}\t{name}\t{addr}\t{cty}\t{source_label}\n")
                        curr_shard_count += 1
            finally:
                reader.close()
    finally:
        f_out.close()

    if curr_shard_count > 0:
        os.replace(curr_tmp_file, curr_shard_file)
        shard_paths.append(curr_shard_file)
    elif curr_tmp_file.exists():
        try:
            curr_tmp_file.unlink()
        except Exception:
            pass

    print(f"Target partitioning complete: {len(shard_paths)} disk shards created in '{target_shards_dir.name}'.", flush=True)
    return shard_paths


def partition_s1_queries(
    s1_path: Union[Path, str],
    s1_batches_dir: Path,
    batch_size: int = BATCH_SIZE_S1_QUERIES,
    chunksize: int = PILOT_CHUNKSIZE,
    filter_s1_ids: Optional[Set[str]] = None,
) -> Tuple[List[Path], int]:
    """Stream-normalize and slice Source 1 queries into disk-backed batch files with atomic writes."""
    s1_batches_dir.mkdir(parents=True, exist_ok=True)
    batch_paths: List[Path] = []

    s1_file = Path(s1_path)
    if not s1_file.exists():
        raise FileNotFoundError(f"Source 1 file not found at: {s1_file.resolve()}")

    reader = pd.read_csv(
        s1_file,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=chunksize,
    )

    curr_batch_idx = 0
    curr_batch_count = 0
    curr_batch_file = s1_batches_dir / f"s1_batch_{curr_batch_idx:04d}.tsv"
    curr_tmp_file = curr_batch_file.with_suffix(".tsv.tmp")
    f_out = open(curr_tmp_file, "w", encoding="utf-8", newline="")
    f_out.write("entity_id\tbusiness_name_norm\tbusiness_address_norm\tcountry_norm\n")
    total_s1 = 0

    try:
        for chunk in reader:
            chunk_norm = normalize_dataframe(chunk, inplace=True)
            if filter_s1_ids is not None:
                chunk_norm = chunk_norm[chunk_norm["entity_id"].isin(filter_s1_ids)]

            if not chunk_norm.empty:
                eids = chunk_norm["entity_id"].tolist()
                names = chunk_norm["business_name_norm"].tolist()
                addrs = chunk_norm["business_address_norm"].tolist()
                ctys = chunk_norm["country_norm"].tolist() if "country_norm" in chunk_norm.columns else [""] * len(chunk_norm)

                for eid, name, addr, cty in zip(eids, names, addrs, ctys):
                    if curr_batch_count >= batch_size:
                        f_out.close()
                        os.replace(curr_tmp_file, curr_batch_file)
                        batch_paths.append(curr_batch_file)

                        curr_batch_idx += 1
                        curr_batch_count = 0
                        curr_batch_file = s1_batches_dir / f"s1_batch_{curr_batch_idx:04d}.tsv"
                        curr_tmp_file = curr_batch_file.with_suffix(".tsv.tmp")
                        f_out = open(curr_tmp_file, "w", encoding="utf-8", newline="")
                        f_out.write("entity_id\tbusiness_name_norm\tbusiness_address_norm\tcountry_norm\n")

                    f_out.write(f"{eid}\t{name}\t{addr}\t{cty}\n")
                    curr_batch_count += 1
                    total_s1 += 1
    finally:
        reader.close()
        f_out.close()

    if curr_batch_count > 0:
        os.replace(curr_tmp_file, curr_batch_file)
        batch_paths.append(curr_batch_file)
    elif curr_tmp_file.exists():
        try:
            curr_tmp_file.unlink()
        except Exception:
            pass

    print(f"Source 1 query partitioning complete: {total_s1:,} queries in {len(batch_paths)} batch files.", flush=True)
    return batch_paths, total_s1


def run_disk_backed_retrieval_pipeline(
    s1_path: Union[Path, str],
    s2_path: Union[Path, str],
    s3_path: Union[Path, str],
    partitions_dir: Path,
    manifest_path: Path,
    mode: str = "eval_val",
    split_file: Optional[Union[Path, str]] = None,
    filter_s1_ids: Optional[Set[str]] = None,
    sample_s1: Optional[int] = None,
    gt_lookup: Optional[Dict[str, Set[str]]] = None,
    batch_size: int = BATCH_SIZE_S1_QUERIES,
    shard_size: int = SHARD_SIZE_TARGETS,
    chunksize: int = PILOT_CHUNKSIZE,
    max_candidates: int = BLOCKING_MAX_CANDIDATES_PER_S1,
    per_shard_candidate_cap: int = BLOCKING_PER_SHARD_CANDIDATE_CAP,
    resume: bool = True,
) -> Dict[str, Any]:
    """Execute memory-bounded, disk-backed single-shard-resident retrieval with atomic writes."""
    partitions_dir = Path(partitions_dir)
    partitions_dir.mkdir(parents=True, exist_ok=True)
    target_shards_dir = partitions_dir / "target_shards"
    s1_batches_dir = partitions_dir / "s1_batches"
    intermediate_dir = partitions_dir / "intermediate"
    merged_batches_dir = partitions_dir / "merged_batches"

    target_shards_dir.mkdir(parents=True, exist_ok=True)
    s1_batches_dir.mkdir(parents=True, exist_ok=True)
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    merged_batches_dir.mkdir(parents=True, exist_ok=True)

    manifest_file = Path(manifest_path)
    t_start = time.time()

    # --------------------------------------------------------------------------
    # 1. Compute & Validate Manifest Fingerprint
    # --------------------------------------------------------------------------
    current_fingerprint = compute_manifest_fingerprint(
        mode=mode,
        input_files=[s1_path, s2_path, s3_path],
        split_file=split_file,
        filter_s1_ids=filter_s1_ids,
        sample_s1=sample_s1,
        shard_size=shard_size,
        batch_size=batch_size,
        max_candidates=max_candidates,
        per_shard_cap=per_shard_candidate_cap,
    )

    manifest_data: Dict[str, Any] = {
        "fingerprint": current_fingerprint,
        "mode": mode,
        "completed_batches": [],
        "total_s1_processed": 0,
        "total_candidates_generated": 0,
        "total_truncations": 0,
        "empty_candidate_queries": 0,
        "total_shard_candidates_pruned": 0,
        "measured_shard_ram_mb": [],
        "accumulated_metrics": {
            "total_gt_pairs": 0,
            "total_s2_gt": 0,
            "total_s3_gt": 0,
            "found_gt_pairs": 0,
            "found_s2_gt": 0,
            "found_s3_gt": 0,
            "s1_with_matches": 0,
            "s1_full_covered": 0,
        },
    }

    if resume and manifest_file.exists():
        try:
            with open(manifest_file, "r", encoding="utf-8") as f:
                saved = json.load(f)
            saved_fp = saved.get("fingerprint")
            if saved_fp == current_fingerprint:
                manifest_data.update(saved)
                print(f"[RESUME] Manifest fingerprint matched. Resuming from {len(manifest_data['completed_batches'])} completed batches.", flush=True)
            else:
                print(f"[STALE MANIFEST] Manifest fingerprint mismatch (mode/config/inputs changed). Rebuilding all partition directories.", flush=True)
                for d in [target_shards_dir, s1_batches_dir, intermediate_dir, merged_batches_dir]:
                    shutil.rmtree(d, ignore_errors=True)
                    d.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"[WARNING] Could not read manifest '{manifest_file}': {e}. Starting fresh.", flush=True)
            for d in [target_shards_dir, s1_batches_dir, intermediate_dir, merged_batches_dir]:
                shutil.rmtree(d, ignore_errors=True)
                d.mkdir(parents=True, exist_ok=True)
    elif not resume:
        for d in [target_shards_dir, s1_batches_dir, intermediate_dir, merged_batches_dir]:
            shutil.rmtree(d, ignore_errors=True)
            d.mkdir(parents=True, exist_ok=True)

    completed_batches: Set[int] = set(manifest_data["completed_batches"])

    # Clean any orphan temporary files
    for tmp_f in partitions_dir.rglob("*.tmp"):
        try:
            tmp_f.unlink()
        except Exception:
            pass

    # --------------------------------------------------------------------------
    # 2. Disk-backed Target and S1 Partitioning
    # --------------------------------------------------------------------------
    existing_target_shards = sorted(target_shards_dir.glob("target_shard_*.tsv"))
    if not existing_target_shards:
        target_shard_paths = partition_target_sources(
            s2_path=s2_path,
            s3_path=s3_path,
            target_shards_dir=target_shards_dir,
            shard_size=shard_size,
            chunksize=chunksize,
        )
    else:
        target_shard_paths = existing_target_shards
        print(f"Using {len(target_shard_paths)} existing target shards in '{target_shards_dir.name}'.", flush=True)

    existing_s1_batches = sorted(s1_batches_dir.glob("s1_batch_*.tsv"))
    if not existing_s1_batches:
        s1_batch_paths, total_s1 = partition_s1_queries(
            s1_path=s1_path,
            s1_batches_dir=s1_batches_dir,
            batch_size=batch_size,
            chunksize=chunksize,
            filter_s1_ids=filter_s1_ids,
        )
    else:
        s1_batch_paths = existing_s1_batches
        print(f"Using {len(s1_batch_paths)} existing S1 batch files in '{s1_batches_dir.name}'.", flush=True)

    num_target_shards = len(target_shard_paths)
    num_s1_batches = len(s1_batch_paths)

    # --------------------------------------------------------------------------
    # 3. Shard-Major Retrieval: Exactly ONE Index Shard in RAM at a Time
    # --------------------------------------------------------------------------
    measured_ram_list: List[float] = []
    total_shard_pruned = manifest_data.get("total_shard_candidates_pruned", 0)

    for shard_idx, shard_path in enumerate(target_shard_paths):
        all_batches_done = all(b in completed_batches for b in range(num_s1_batches))
        if all_batches_done:
            print(f"All {num_s1_batches} batches already completed. Skipping target shard {shard_idx:04d}.", flush=True)
            continue

        print(f"\n[SHARD {shard_idx + 1}/{num_target_shards}] Building in-memory index for '{shard_path.name}'...", flush=True)
        t_shard_load = time.time()
        shard_df = pd.read_csv(shard_path, sep="\t", dtype=str, keep_default_na=False)
        shard_index = TargetIndexShard(shard_id=shard_idx)
        shard_index.build_from_dataframe(shard_df)
        shard_ram_mb = shard_index.measured_rss_peak_mb or (shard_index.measured_ram_bytes / (1024 * 1024))
        measured_ram_list.append(round(shard_ram_mb, 2))
        print(f"  -> Shard {shard_idx:04d} ready ({len(shard_df):,} targets, measured peak: {shard_ram_mb:.2f} MB) in {time.time() - t_shard_load:.2f}s.", flush=True)
        del shard_df
        gc.collect()

        # Stream all S1 query batches through this single resident shard
        for batch_idx, batch_path in enumerate(s1_batch_paths):
            if batch_idx in completed_batches:
                continue

            inter_file = intermediate_dir / f"shard_{shard_idx:04d}_batch_{batch_idx:04d}.tsv"
            if inter_file.exists():
                continue

            inter_tmp = inter_file.with_suffix(".tsv.tmp")
            s1_batch_df = pd.read_csv(batch_path, sep="\t", dtype=str, keep_default_na=False)
            s1_records = s1_batch_df.to_dict(orient="records")

            with open(inter_tmp, "w", encoding="utf-8", newline="") as f_inter:
                f_inter.write("s1_id\tcandidate_id\tsource_dataset\trule_name\tscore\n")
                for rec in s1_records:
                    s1_id = rec["entity_id"]
                    name = rec.get("business_name_norm", "")
                    addr = rec.get("business_address_norm", "")
                    country = rec.get("country_norm", "")

                    cand_map, pruned = shard_index.query_record(
                        name=name,
                        addr=addr,
                        country=country,
                        max_candidates=max_candidates,
                        per_shard_cap=per_shard_candidate_cap,
                        return_pruned_count=True,
                    )
                    total_shard_pruned += pruned

                    for cid, (src, rule, score) in cand_map.items():
                        f_inter.write(f"{s1_id}\t{cid}\t{src}\t{rule}\t{score}\n")

            os.replace(inter_tmp, inter_file)
            del s1_batch_df
            del s1_records

        # Explicitly unload index shard from RAM
        print(f"  -> Unloading Shard {shard_idx:04d} from RAM (freeing memory)...", flush=True)
        del shard_index
        gc.collect()

    manifest_data["measured_shard_ram_mb"] = measured_ram_list
    manifest_data["total_shard_candidates_pruned"] = total_shard_pruned

    # --------------------------------------------------------------------------
    # 4. Batch-by-Batch Candidate Merger & Online Metric Accumulation
    # --------------------------------------------------------------------------
    print("\n" + "=" * 75, flush=True)
    print("MERGING INTERMEDIATE SHARDS & ACCUMULATING VALIDATION METRICS", flush=True)
    print("=" * 75, flush=True)

    cand_counts_sample: List[int] = []

    for batch_idx, batch_path in enumerate(s1_batch_paths):
        pairs_batch_file = merged_batches_dir / f"candidate_pairs_batch_{batch_idx:04d}.tsv"
        prov_batch_file = merged_batches_dir / f"candidate_provenance_batch_{batch_idx:04d}.tsv"

        if batch_idx in completed_batches and pairs_batch_file.exists() and prov_batch_file.exists():
            print(f"Batch {batch_idx:04d} already completed and recorded in manifest. Skipping merger.", flush=True)
            continue

        b_t0 = time.time()
        s1_batch_df = pd.read_csv(batch_path, sep="\t", dtype=str, keep_default_na=False)
        s1_records = s1_batch_df.to_dict(orient="records")

        # Collect all intermediate matches for this batch across all shards
        merged_candidates: Dict[str, Dict[str, Tuple[str, str, int]]] = {
            rec["entity_id"]: {} for rec in s1_records
        }

        inter_files = [
            intermediate_dir / f"shard_{s_idx:04d}_batch_{batch_idx:04d}.tsv"
            for s_idx in range(num_target_shards)
        ]

        for ifile in inter_files:
            if ifile.exists():
                with open(ifile, "r", encoding="utf-8") as f_in:
                    header = f_in.readline()
                    for line in f_in:
                        parts = line.rstrip("\r\n").split("\t")
                        if len(parts) >= 5:
                            s1_id, cid, src, rule, score_str = parts[0], parts[1], parts[2], parts[3], parts[4]
                            score = int(score_str)
                            if s1_id in merged_candidates:
                                cur_map = merged_candidates[s1_id]
                                if cid not in cur_map or score > cur_map[cid][2]:
                                    cur_map[cid] = (src, rule, score)

        batch_truncations = 0
        batch_empty = 0
        batch_pairs_count = 0

        pairs_tmp = pairs_batch_file.with_suffix(".tsv.tmp")
        prov_tmp = prov_batch_file.with_suffix(".tsv.tmp")

        with open(pairs_tmp, "w", encoding="utf-8", newline="") as f_pairs, \
             open(prov_tmp, "w", encoding="utf-8", newline="") as f_prov:

            f_pairs.write("source1_entity_id\tcandidate_entity_ids\n")
            f_prov.write("source1_entity_id\tcandidate_entity_id\tsource_dataset\tprovenance_rules\n")

            for rec in s1_records:
                s1_id = rec["entity_id"]
                cand_map = merged_candidates[s1_id]
                total_found = len(cand_map)
                is_truncated = total_found > max_candidates

                if is_truncated:
                    sorted_items = sorted(
                        cand_map.items(),
                        key=lambda item: (-item[1][2], item[0]),
                    )[:max_candidates]
                    cand_map = dict(sorted_items)
                    batch_truncations += 1

                num_cands = len(cand_map)
                cand_counts_sample.append(num_cands)
                batch_pairs_count += num_cands

                if num_cands == 0:
                    batch_empty += 1

                cand_ids = sorted(cand_map.keys())
                f_pairs.write(f"{s1_id}\t{','.join(cand_ids)}\n")

                for cid, (src, rule, _) in cand_map.items():
                    f_prov.write(f"{s1_id}\t{cid}\t{src}\t{rule}\n")

                # Update incremental recall metrics without double-counting
                if gt_lookup is not None and s1_id in gt_lookup:
                    true_set = gt_lookup[s1_id]
                    if true_set:
                        manifest_data["accumulated_metrics"]["s1_with_matches"] += 1
                        manifest_data["accumulated_metrics"]["total_gt_pairs"] += len(true_set)
                        s2_true = {m for m in true_set if m.startswith("S2-")}
                        s3_true = {m for m in true_set if m.startswith("S3-")}
                        manifest_data["accumulated_metrics"]["total_s2_gt"] += len(s2_true)
                        manifest_data["accumulated_metrics"]["total_s3_gt"] += len(s3_true)

                        found = true_set.intersection(cand_map.keys())
                        manifest_data["accumulated_metrics"]["found_gt_pairs"] += len(found)
                        manifest_data["accumulated_metrics"]["found_s2_gt"] += len(s2_true.intersection(cand_map.keys()))
                        manifest_data["accumulated_metrics"]["found_s3_gt"] += len(s3_true.intersection(cand_map.keys()))

                        if len(found) == len(true_set):
                            manifest_data["accumulated_metrics"]["s1_full_covered"] += 1

        # Atomically commit batch output files
        os.replace(pairs_tmp, pairs_batch_file)
        os.replace(prov_tmp, prov_batch_file)

        b_time = time.time() - b_t0
        throughput = len(s1_records) / b_time if b_time > 0 else 0.0

        # Update and save manifest atomically
        manifest_data["completed_batches"].append(batch_idx)
        manifest_data["total_s1_processed"] += len(s1_records)
        manifest_data["total_candidates_generated"] += batch_pairs_count
        manifest_data["total_truncations"] += batch_truncations
        manifest_data["empty_candidate_queries"] += batch_empty

        manifest_tmp = manifest_file.with_suffix(".json.tmp")
        with open(manifest_tmp, "w", encoding="utf-8") as f_m:
            json.dump(manifest_data, f_m, indent=2)
        os.replace(manifest_tmp, manifest_file)

        # Remove intermediate files for this batch to save disk space
        for ifile in inter_files:
            try:
                if ifile.exists():
                    ifile.unlink()
            except Exception:
                pass

        print(
            f"Batch {batch_idx:04d}: Merged {len(s1_records):,} S1 queries -> "
            f"{batch_pairs_count:,} candidate pairs ({throughput:.1f} q/s, "
            f"empty: {batch_empty}, truncated: {batch_truncations}) in {b_time:.2f}s.",
            flush=True,
        )
        del s1_batch_df
        del s1_records
        del merged_candidates
        gc.collect()

    total_time = time.time() - t_start

    # Compute validation recall summary
    rec_stats = manifest_data["accumulated_metrics"]
    tot_gt = rec_stats["total_gt_pairs"]
    val_pair_recall = rec_stats["found_gt_pairs"] / tot_gt if tot_gt > 0 else 0.0
    val_s2_rec = rec_stats["found_s2_gt"] / rec_stats["total_s2_gt"] if rec_stats["total_s2_gt"] > 0 else 0.0
    val_s3_rec = rec_stats["found_s3_gt"] / rec_stats["total_s3_gt"] if rec_stats["total_s3_gt"] > 0 else 0.0
    val_full_cov = rec_stats["s1_full_covered"] / rec_stats["s1_with_matches"] if rec_stats["s1_with_matches"] > 0 else 0.0

    cand_arr = np.array(cand_counts_sample) if cand_counts_sample else np.array([0])
    return {
        "total_s1_processed": manifest_data["total_s1_processed"],
        "total_candidate_pairs": manifest_data["total_candidates_generated"],
        "total_truncations": manifest_data["total_truncations"],
        "truncation_rate": manifest_data["total_truncations"] / max(manifest_data["total_s1_processed"], 1),
        "empty_candidate_count": manifest_data["empty_candidate_queries"],
        "empty_candidate_rate": manifest_data["empty_candidate_queries"] / max(manifest_data["total_s1_processed"], 1),
        "total_shard_candidates_pruned": manifest_data.get("total_shard_candidates_pruned", 0),
        "mean_candidates": float(np.mean(cand_arr)),
        "median_candidates": float(np.median(cand_arr)),
        "p90_candidates": float(np.percentile(cand_arr, 90)),
        "p95_candidates": float(np.percentile(cand_arr, 95)),
        "p99_candidates": float(np.percentile(cand_arr, 99)),
        "max_candidates": int(np.max(cand_arr)),
        "total_elapsed_seconds": round(total_time, 2),
        "overall_throughput_qps": round(manifest_data["total_s1_processed"] / total_time if total_time > 0 else 0.0, 1),
        "measured_shard_ram_mb": manifest_data["measured_shard_ram_mb"],
        "val_total_gt_pairs": tot_gt,
        "val_found_gt_pairs": rec_stats["found_gt_pairs"],
        "val_pair_recall": val_pair_recall,
        "val_s2_recall": val_s2_rec,
        "val_s3_recall": val_s3_rec,
        "val_full_s1_coverage": val_full_cov,
    }


def merge_candidate_partitions(
    partitions_dir: Union[Path, str],
    output_pairs_path: Union[Path, str],
    output_prov_path: Optional[Union[Path, str]] = None,
    clean_partitions: bool = False,
) -> Dict[str, Any]:
    """Stream-merge all batch candidate partition files into final candidate_pairs.tsv and candidate_provenance.tsv with atomic rename."""
    part_dir = Path(partitions_dir)
    merged_dir = part_dir / "merged_batches" if (part_dir / "merged_batches").exists() else part_dir
    pairs_dest = Path(output_pairs_path)
    prov_dest = Path(output_prov_path) if output_prov_path else None

    pairs_dest.parent.mkdir(parents=True, exist_ok=True)
    if prov_dest:
        prov_dest.parent.mkdir(parents=True, exist_ok=True)

    pair_part_files = sorted(merged_dir.glob("candidate_pairs_batch_*.tsv"))
    prov_part_files = sorted(merged_dir.glob("candidate_provenance_batch_*.tsv"))

    if not pair_part_files:
        raise FileNotFoundError(f"No candidate partition files found in '{merged_dir.resolve()}'.")

    print(f"Merging {len(pair_part_files)} candidate pair partition files into '{pairs_dest.name}'...", flush=True)
    total_pairs_rows = 0

    pairs_tmp = pairs_dest.with_suffix(".tsv.tmp")
    with open(pairs_tmp, "w", encoding="utf-8", newline="") as f_out:
        f_out.write("source1_entity_id\tcandidate_entity_ids\n")
        for pfile in pair_part_files:
            with open(pfile, "r", encoding="utf-8") as f_in:
                header = f_in.readline()
                for line in f_in:
                    f_out.write(line)
                    total_pairs_rows += 1
    os.replace(pairs_tmp, pairs_dest)

    total_prov_rows = 0
    if prov_dest and prov_part_files:
        print(f"Merging {len(prov_part_files)} provenance partition files into '{prov_dest.name}'...", flush=True)
        prov_tmp = prov_dest.with_suffix(".tsv.tmp")
        with open(prov_tmp, "w", encoding="utf-8", newline="") as f_out:
            f_out.write("source1_entity_id\tcandidate_entity_id\tsource_dataset\tprovenance_rules\n")
            for pfile in prov_part_files:
                with open(pfile, "r", encoding="utf-8") as f_in:
                    header = f_in.readline()
                    for line in f_in:
                        f_out.write(line)
                        total_prov_rows += 1
        os.replace(prov_tmp, prov_dest)

    complete_marker = pairs_dest.with_suffix(".complete")
    with open(complete_marker, "w", encoding="utf-8") as f_mark:
        f_mark.write(f"COMPLETED at {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n")
        f_mark.write(f"total_s1_rows={total_pairs_rows}\n")
        f_mark.write(f"total_provenance_rows={total_prov_rows}\n")

    if clean_partitions:
        print("Cleaning temporary partition directory...", flush=True)
        shutil.rmtree(part_dir, ignore_errors=True)

    print(f"[MERGE COMPLETE] Successfully wrote {total_pairs_rows:,} candidate rows to '{pairs_dest.resolve()}'.\n", flush=True)
    return {
        "output_pairs_path": str(pairs_dest.resolve()),
        "output_prov_path": str(prov_dest.resolve()) if prov_dest else None,
        "total_s1_rows": total_pairs_rows,
        "total_provenance_rows": total_prov_rows,
    }


# Backwards-compatible aliases for existing test setups
def build_sharded_target_index(
    s2_path: Union[Path, str],
    s3_path: Union[Path, str],
    shard_size: int = SHARD_SIZE_TARGETS,
    chunksize: int = PILOT_CHUNKSIZE,
) -> ShardedTargetIndex:
    """Build in-memory ShardedTargetIndex (for backward compatibility)."""
    sharded_index = ShardedTargetIndex(shard_size=shard_size)
    curr_shard = TargetIndexShard(shard_id=0)
    current_shard_count = 0
    total_indexed = 0

    for source_label, file_path in [("source2", Path(s2_path)), ("source3", Path(s3_path))]:
        if not file_path.exists():
            raise FileNotFoundError(f"Target file not found at: {file_path.resolve()}")

        reader = pd.read_csv(file_path, sep="\t", dtype=str, keep_default_na=False, chunksize=chunksize)
        try:
            for chunk in reader:
                chunk_norm = normalize_dataframe(chunk, inplace=True)
                added = curr_shard.add_records(df=chunk_norm, source_label=source_label)
                current_shard_count += added
                total_indexed += added

                if current_shard_count >= shard_size:
                    sharded_index.add_shard(curr_shard)
                    curr_shard = TargetIndexShard(shard_id=len(sharded_index.shards))
                    current_shard_count = 0
                    gc.collect()
        finally:
            reader.close()

    if len(curr_shard.entity_ids) > 0:
        sharded_index.add_shard(curr_shard)

    return sharded_index


def stream_process_s1_batches(
    s1_path: Union[Path, str],
    sharded_index: ShardedTargetIndex,
    partitions_dir: Path = PARTITIONS_DIR,
    manifest_path: Path = MANIFEST_PATH,
    filter_s1_ids: Optional[Set[str]] = None,
    gt_lookup: Optional[Dict[str, Set[str]]] = None,
    batch_size: int = BATCH_SIZE_S1_QUERIES,
    chunksize: int = PILOT_CHUNKSIZE,
    max_candidates: int = BLOCKING_MAX_CANDIDATES_PER_S1,
    resume: bool = True,
) -> Dict[str, Any]:
    """In-memory streaming query runner (provided for backward compatibility)."""
    partitions_dir = Path(partitions_dir)
    partitions_dir.mkdir(parents=True, exist_ok=True)
    manifest_file = Path(manifest_path)

    completed_batches: Set[int] = set()
    manifest_data: Dict[str, Any] = {
        "completed_batches": [],
        "total_s1_processed": 0,
        "total_candidates_generated": 0,
        "total_truncations": 0,
        "empty_candidate_queries": 0,
        "accumulated_metrics": {
            "total_gt_pairs": 0,
            "total_s2_gt": 0,
            "total_s3_gt": 0,
            "found_gt_pairs": 0,
            "found_s2_gt": 0,
            "found_s3_gt": 0,
            "s1_with_matches": 0,
            "s1_full_covered": 0,
        },
    }

    if resume and manifest_file.exists():
        try:
            with open(manifest_file, "r", encoding="utf-8") as f:
                saved = json.load(f)
                manifest_data.update(saved)
                completed_batches = set(manifest_data.get("completed_batches", []))
        except Exception:
            pass

    s1_file = Path(s1_path)
    reader = pd.read_csv(s1_file, sep="\t", dtype=str, keep_default_na=False, chunksize=chunksize)

    batch_buffer: List[Dict[str, str]] = []
    batch_idx = 0
    t_start = time.time()
    cand_counts_sample: List[int] = []

    def process_current_batch(b_idx: int, s1_records: List[Dict[str, str]]) -> None:
        nonlocal manifest_data
        pairs_file = partitions_dir / f"candidate_pairs_batch_{b_idx:04d}.tsv"
        prov_file = partitions_dir / f"candidate_provenance_batch_{b_idx:04d}.tsv"

        if b_idx in completed_batches and pairs_file.exists() and prov_file.exists():
            return

        batch_truncations = 0
        batch_empty = 0
        batch_pairs_count = 0

        with open(pairs_file, "w", encoding="utf-8", newline="") as f_pairs, \
             open(prov_file, "w", encoding="utf-8", newline="") as f_prov:

            f_pairs.write("source1_entity_id\tcandidate_entity_ids\n")
            f_prov.write("source1_entity_id\tcandidate_entity_id\tsource_dataset\tprovenance_rules\n")

            for rec in s1_records:
                s1_id = rec["entity_id"]
                name = rec.get("business_name_norm", "")
                addr = rec.get("business_address_norm", "")
                country = rec.get("country_norm", "")

                cand_set, prov_records, is_truncated = sharded_index.query_record(
                    name=name,
                    addr=addr,
                    country=country,
                    max_candidates=max_candidates,
                )

                num_cands = len(cand_set)
                cand_counts_sample.append(num_cands)
                batch_pairs_count += num_cands

                if num_cands == 0:
                    batch_empty += 1
                if is_truncated:
                    batch_truncations += 1

                sorted_cands = sorted(cand_set)
                f_pairs.write(f"{s1_id}\t{','.join(sorted_cands)}\n")

                for cid, src, rule in prov_records:
                    f_prov.write(f"{s1_id}\t{cid}\t{src}\t{rule}\n")

                if gt_lookup is not None and s1_id in gt_lookup:
                    true_set = gt_lookup[s1_id]
                    if true_set:
                        manifest_data["accumulated_metrics"]["s1_with_matches"] += 1
                        manifest_data["accumulated_metrics"]["total_gt_pairs"] += len(true_set)
                        s2_true = {m for m in true_set if m.startswith("S2-")}
                        s3_true = {m for m in true_set if m.startswith("S3-")}
                        manifest_data["accumulated_metrics"]["total_s2_gt"] += len(s2_true)
                        manifest_data["accumulated_metrics"]["total_s3_gt"] += len(s3_true)

                        found = true_set.intersection(cand_set)
                        manifest_data["accumulated_metrics"]["found_gt_pairs"] += len(found)
                        manifest_data["accumulated_metrics"]["found_s2_gt"] += len(s2_true.intersection(cand_set))
                        manifest_data["accumulated_metrics"]["found_s3_gt"] += len(s3_true.intersection(cand_set))

                        if len(found) == len(true_set):
                            manifest_data["accumulated_metrics"]["s1_full_covered"] += 1

        manifest_data["completed_batches"].append(b_idx)
        manifest_data["total_s1_processed"] += len(s1_records)
        manifest_data["total_candidates_generated"] += batch_pairs_count
        manifest_data["total_truncations"] += batch_truncations
        manifest_data["empty_candidate_queries"] += batch_empty

        with open(manifest_file, "w", encoding="utf-8") as f_m:
            json.dump(manifest_data, f_m, indent=2)

    try:
        for chunk in reader:
            chunk_norm = normalize_dataframe(chunk, inplace=True)
            if filter_s1_ids is not None:
                chunk_norm = chunk_norm[chunk_norm["entity_id"].isin(filter_s1_ids)]

            if not chunk_norm.empty:
                records = chunk_norm.to_dict(orient="records")
                batch_buffer.extend(records)

                while len(batch_buffer) >= batch_size:
                    current_batch = batch_buffer[:batch_size]
                    batch_buffer = batch_buffer[batch_size:]
                    process_current_batch(batch_idx, current_batch)
                    batch_idx += 1

        if batch_buffer:
            process_current_batch(batch_idx, batch_buffer)
            batch_idx += 1
    finally:
        reader.close()

    total_time = time.time() - t_start
    rec_stats = manifest_data["accumulated_metrics"]
    tot_gt = rec_stats["total_gt_pairs"]
    val_pair_recall = rec_stats["found_gt_pairs"] / tot_gt if tot_gt > 0 else 0.0
    val_s2_rec = rec_stats["found_s2_gt"] / rec_stats["total_s2_gt"] if rec_stats["total_s2_gt"] > 0 else 0.0
    val_s3_rec = rec_stats["found_s3_gt"] / rec_stats["total_s3_gt"] if rec_stats["total_s3_gt"] > 0 else 0.0
    val_full_cov = rec_stats["s1_full_covered"] / rec_stats["s1_with_matches"] if rec_stats["s1_with_matches"] > 0 else 0.0

    cand_arr = np.array(cand_counts_sample) if cand_counts_sample else np.array([0])
    return {
        "total_s1_processed": manifest_data["total_s1_processed"],
        "total_candidate_pairs": manifest_data["total_candidates_generated"],
        "total_truncations": manifest_data["total_truncations"],
        "truncation_rate": manifest_data["total_truncations"] / max(manifest_data["total_s1_processed"], 1),
        "empty_candidate_count": manifest_data["empty_candidate_queries"],
        "empty_candidate_rate": manifest_data["empty_candidate_queries"] / max(manifest_data["total_s1_processed"], 1),
        "mean_candidates": float(np.mean(cand_arr)),
        "median_candidates": float(np.median(cand_arr)),
        "p90_candidates": float(np.percentile(cand_arr, 90)),
        "p95_candidates": float(np.percentile(cand_arr, 95)),
        "p99_candidates": float(np.percentile(cand_arr, 99)),
        "max_candidates": int(np.max(cand_arr)),
        "total_elapsed_seconds": round(total_time, 2),
        "overall_throughput_qps": round(manifest_data["total_s1_processed"] / total_time if total_time > 0 else 0.0, 1),
        "val_total_gt_pairs": tot_gt,
        "val_found_gt_pairs": rec_stats["found_gt_pairs"],
        "val_pair_recall": val_pair_recall,
        "val_s2_recall": val_s2_rec,
        "val_s3_recall": val_s3_rec,
        "val_full_s1_coverage": val_full_cov,
    }

