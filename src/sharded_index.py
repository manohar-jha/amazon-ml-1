"""Compact sharded inverted index and MinHash LSH engine for large-scale target retrieval.

Provides memory-bounded indexing of Source 2 and Source 3 target entities by
partitioning the target space into compact, independently queryable index shards.
Ensures strictly one bounded shard is resident in RAM at a time during retrieval.
"""

import gc
import os
import re
import sys
import tracemalloc
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd

from src.config import (
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
    COMMON_LEGAL_TERMS,
    LSH_MAX_BUCKET_SIZE,
    LSH_MAX_CANDIDATES,
    LSH_NUM_BANDS,
    LSH_NUM_PERMUTATIONS,
    LSH_SHINGLE_N,
    RAM_SAFETY_MARGIN,
    SHARD_SIZE_TARGETS,
)
from src.lsh import MinHashLSH
from src.normalize import normalize_dataframe


def get_process_rss_bytes() -> Optional[int]:
    """Return the current process Resident Set Size (RSS) in bytes using OS-level metrics."""
    # 1. Try psutil
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss
    except Exception:
        pass

    # 2. Try resource (Linux / macOS)
    try:
        import resource
        ru = resource.getrusage(resource.RUSAGE_SELF)
        if sys.platform == "darwin":
            return int(ru.ru_maxrss)
        else:
            return int(ru.ru_maxrss * 1024)
    except Exception:
        pass

    # 3. Try Windows ctypes
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            if ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
                return int(counters.WorkingSetSize)
    except Exception:
        pass

    return None


def get_core_name(name: Optional[str]) -> str:
    """Extract core business name by stripping common corporate and legal suffixes."""
    if not name:
        return ""
    words = [w for w in name.split() if w not in COMMON_LEGAL_TERMS]
    return " ".join(words) if words else name


def get_compact_name(name: Optional[str]) -> str:
    """Return whitespace-stripped representation of the core name."""
    if not name:
        return ""
    core = get_core_name(name)
    return core.replace(" ", "") if core else name.replace(" ", "")


def _clean_num(token: str) -> str:
    """Strip leading zeros from numeric tokens while preserving zero."""
    stripped = token.lstrip("0")
    return stripped if stripped else "0"


def extract_address_keys(addr: Optional[str]) -> List[str]:
    """Extract distinctive address signals (numbers and landmark words)."""
    if not addr:
        return []

    tokens = addr.split()
    nums = [_clean_num(t) for t in tokens if re.search(r"\d", t)]
    words = [
        t for t in tokens
        if len(t) >= 4 and not re.search(r"\d", t) and t not in {
            "road", "street", "avenue", "lane", "drive", "floor", "suite",
            "block", "near", "behind", "opp", "opposite", "post", "dist",
            "district", "state", "nagar", "colony", "bhavan", "house",
            "north", "south", "east", "west", "building", "unit",
        }
    ]

    keys: List[str] = []
    for num in nums[:2]:
        for w in words[:2]:
            keys.append(f"a_nw:{num}_{w}")

    if len(nums) >= 2:
        keys.append(f"a_nn:{nums[0]}_{nums[1]}")

    if not nums and len(words) >= 2:
        keys.append(f"a_ww:{words[0]}_{words[1]}")

    return keys


class TargetIndexShard:
    """Compact in-memory index shard for a single bounded slice of target records.
    
    Constructs deterministic, order-independent inverted indices and MinHash LSH tables.
    """

    def __init__(self, shard_id: int):
        self.shard_id = shard_id
        self.entity_ids: List[str] = []
        self.source_labels: List[str] = []  # 'source2' or 'source3'
        self.country_labels: List[str] = []

        # Inverted index tables mapping key -> list of integer entity indices
        self.exact_name_idx: Dict[str, List[int]] = defaultdict(list)
        self.core_name_idx: Dict[str, List[int]] = defaultdict(list)
        self.compact_name_idx: Dict[str, List[int]] = defaultdict(list)
        self.token_idx: Dict[str, List[int]] = defaultdict(list)
        self.pair_idx: Dict[str, List[int]] = defaultdict(list)
        self.addr_idx: Dict[str, List[int]] = defaultdict(list)
        self.prefix_idx: Dict[str, List[int]] = defaultdict(list)

        self.token_freq: Counter = Counter()
        self.addr_key_freq: Counter = Counter()
        self.pair_key_freq: Counter = Counter()

        self.lsh = MinHashLSH(
            num_permutations=LSH_NUM_PERMUTATIONS,
            num_bands=LSH_NUM_BANDS,
            shingle_n=LSH_SHINGLE_N,
        )
        self.measured_ram_bytes: int = 0
        self.measured_rss_peak_mb: float = 0.0

    def build_from_dataframe(
        self,
        df: pd.DataFrame,
        source_label: Optional[str] = None,
        max_token_freq: int = BLOCKING_MAX_TOKEN_DOC_FREQ,
        max_posting_len: int = BLOCKING_MAX_POSTING_LEN,
        max_pair_freq: int = BLOCKING_MAX_PAIR_TOKEN_FREQ,
        max_addr_freq: int = BLOCKING_MAX_ADDR_KEY_FREQ,
        max_bucket_size: int = BLOCKING_SAFETY_NET_MAX_BUCKET,
    ) -> int:
        """Build deterministic, order-independent index structures from target records.
        
        Pass 1: Computes global document frequencies across the entire shard.
        Pass 2: Populates postings filtered against frequency caps deterministically.
        """
        tracemalloc_started = False
        if not tracemalloc.is_tracing():
            tracemalloc.start()
            tracemalloc_started = True
        t0_heap = tracemalloc.get_traced_memory()[0]

        if "business_name_norm" not in df.columns:
            df = normalize_dataframe(df.copy())

        eids = df["entity_id"].astype(str).tolist()
        names = df["business_name_norm"].astype(str).tolist()
        addrs = df["business_address_norm"].astype(str).tolist()
        countries = df["country_norm"].astype(str).tolist() if "country_norm" in df.columns else [""] * len(df)

        if "source_dataset" in df.columns:
            sources = df["source_dataset"].astype(str).tolist()
        elif source_label:
            sources = [source_label] * len(df)
        else:
            sources = ["source2" if eid.startswith("S2-") else "source3" for eid in eids]

        self.entity_ids = eids
        self.source_labels = sources
        self.country_labels = countries
        n_records = len(eids)

        # ----------------------------------------------------------------------
        # Pass 1: Global Frequency Counting for Order-Independent Capping
        # ----------------------------------------------------------------------
        for name in names:
            if name:
                tokens = [t for t in name.split() if len(t) >= BLOCKING_MIN_TOKEN_LEN]
                self.token_freq.update(set(tokens))

                # Pair frequencies
                if len(tokens) >= 2:
                    sorted_toks = sorted(set(tokens))
                    for t1_idx in range(min(3, len(sorted_toks))):
                        for t2_idx in range(t1_idx + 1, min(4, len(sorted_toks))):
                            self.pair_key_freq[f"{sorted_toks[t1_idx]}_{sorted_toks[t2_idx]}"] += 1

        for addr in addrs:
            if addr:
                self.addr_key_freq.update(extract_address_keys(addr))

        # ----------------------------------------------------------------------
        # Pass 2: Populate Inverted Postings and MinHash LSH
        # ----------------------------------------------------------------------
        for curr_idx in range(n_records):
            eid = eids[curr_idx]
            name = names[curr_idx]
            addr = addrs[curr_idx]
            cty = countries[curr_idx]

            if name:
                # Rule 1: Exact Name
                if len(self.exact_name_idx[name]) < max_posting_len:
                    self.exact_name_idx[name].append(curr_idx)

                # Rule 1b: Core & Compact Names
                core = get_core_name(name)
                if core and core != name and len(self.core_name_idx[core]) < max_posting_len:
                    self.core_name_idx[core].append(curr_idx)

                compact = get_compact_name(name)
                if compact and len(compact) >= 5 and len(self.compact_name_idx[compact]) < max_posting_len:
                    self.compact_name_idx[compact].append(curr_idx)

                # Rule 2: Inverted Tokens (Deterministic Order-Independent Check)
                tokens = [t for t in name.split() if len(t) >= BLOCKING_MIN_TOKEN_LEN]
                for tok in tokens:
                    if self.token_freq[tok] <= max_token_freq:
                        if len(self.token_idx[tok]) < max_posting_len:
                            self.token_idx[tok].append(curr_idx)

                # Rule 3: 2-Token Pair Indexing
                if len(tokens) >= 2:
                    sorted_toks = sorted(set(tokens))
                    for t1_idx in range(min(3, len(sorted_toks))):
                        for t2_idx in range(t1_idx + 1, min(4, len(sorted_toks))):
                            pair_key = f"{sorted_toks[t1_idx]}_{sorted_toks[t2_idx]}"
                            if self.pair_key_freq[pair_key] <= max_pair_freq:
                                if len(self.pair_idx[pair_key]) < max_pair_freq:
                                    self.pair_idx[pair_key].append(curr_idx)

                # Rule 4: MinHash LSH
                self.lsh.index_entity(entity_id=eid, text=name, country=cty, max_bucket_size=LSH_MAX_BUCKET_SIZE)

                # Rule 6: Prefix Safety Net
                if len(name) >= 4:
                    pfx = name[:4]
                    if len(self.prefix_idx[pfx]) < max_bucket_size:
                        self.prefix_idx[pfx].append(curr_idx)

            if addr:
                # Rule 5: Address Number & Landmark Keys
                for a_key in extract_address_keys(addr):
                    if self.addr_key_freq[a_key] <= max_addr_freq:
                        if len(self.addr_idx[a_key]) < max_posting_len:
                            self.addr_idx[a_key].append(curr_idx)

        t1_heap = tracemalloc.get_traced_memory()[0]
        self.measured_ram_bytes = max(0, t1_heap - t0_heap)
        if tracemalloc_started:
            tracemalloc.stop()

        rss_b = get_process_rss_bytes()
        if rss_b is not None:
            self.measured_rss_peak_mb = round(rss_b / (1024 * 1024), 2)

        return n_records

    def add_records(
        self,
        df: pd.DataFrame,
        source_label: str,
        max_token_freq: int = BLOCKING_MAX_TOKEN_DOC_FREQ,
        max_posting_len: int = BLOCKING_MAX_POSTING_LEN,
        max_pair_freq: int = BLOCKING_MAX_PAIR_TOKEN_FREQ,
        max_addr_freq: int = BLOCKING_MAX_ADDR_KEY_FREQ,
        max_bucket_size: int = BLOCKING_SAFETY_NET_MAX_BUCKET,
    ) -> int:
        """Compatibility wrapper for incremental record addition."""
        return self.build_from_dataframe(
            df=df,
            source_label=source_label,
            max_token_freq=max_token_freq,
            max_posting_len=max_posting_len,
            max_pair_freq=max_pair_freq,
            max_addr_freq=max_addr_freq,
            max_bucket_size=max_bucket_size,
        )

    def query_record(
        self,
        name: str,
        addr: str,
        country: str,
        max_candidates: int = BLOCKING_MAX_CANDIDATES_PER_S1,
        per_shard_cap: Optional[int] = BLOCKING_PER_SHARD_CANDIDATE_CAP,
        return_pruned_count: bool = False,
    ) -> Union[Dict[str, Tuple[str, str, int]], Tuple[Dict[str, Tuple[str, str, int]], int]]:
        """Query this shard for candidate matches against a single S1 record.

        Args:
            name: Normalized business name.
            addr: Normalized business address.
            country: Normalized country code.
            max_candidates: Upper bound requested by caller.
            per_shard_cap: Max candidates retained from this single shard before writing intermediate files.
            return_pruned_count: Whether to return the count of pruned candidates.

        Returns:
            Dict[str, Tuple[str, str, int]] or Tuple[Dict, int]: candidate matches and optional pruned count.
        """
        cand_map: Dict[str, Tuple[str, str, int]] = {}

        def add_cand(idx: int, rule: str, score: int) -> None:
            cid = self.entity_ids[idx]
            src = self.source_labels[idx]
            if cid not in cand_map or score > cand_map[cid][2]:
                cand_map[cid] = (src, rule, score)

        # 1. Exact Name (Score: 100)
        if name and name in self.exact_name_idx:
            for idx in self.exact_name_idx[name]:
                add_cand(idx, "exact_name", 100)

        # 1b. Core Name (Score: 90) & Compact Name (Score: 85)
        core = get_core_name(name) if name else ""
        if core and core in self.core_name_idx:
            for idx in self.core_name_idx[core]:
                add_cand(idx, "core_name", 90)

        compact = get_compact_name(name) if name else ""
        if compact and compact in self.compact_name_idx:
            for idx in self.compact_name_idx[compact]:
                add_cand(idx, "compact_name", 85)

        # 2. Address Numbers and Keys (Score: 80)
        if addr:
            for a_key in extract_address_keys(addr):
                if a_key in self.addr_idx:
                    for idx in self.addr_idx[a_key]:
                        add_cand(idx, "address_key", 80)

        # 3. Rare Token Indexing (Score: 70)
        if name:
            tokens = [t for t in name.split() if len(t) >= BLOCKING_MIN_TOKEN_LEN]
            scored_tokens = sorted(tokens, key=lambda t: self.token_freq.get(t, 999999))
            for tok in scored_tokens[:BLOCKING_MAX_RARE_TOKENS_PER_S1]:
                if tok in self.token_idx and self.token_freq.get(tok, 999999) <= BLOCKING_MAX_TOKEN_DOC_FREQ:
                    for idx in self.token_idx[tok]:
                        add_cand(idx, "rare_token", 70)

        # 4. Token Pair Co-occurrence (Score: 60)
        if name:
            tokens = sorted(set(name.split()))
            for t1_idx in range(min(3, len(tokens))):
                for t2_idx in range(t1_idx + 1, min(4, len(tokens))):
                    pair_key = f"{tokens[t1_idx]}_{tokens[t2_idx]}"
                    if pair_key in self.pair_idx:
                        for idx in self.pair_idx[pair_key]:
                            add_cand(idx, "token_pair", 60)

        # 5. MinHash LSH Bands (Score: 50)
        if name:
            lsh_cands = self.lsh.query_candidates(
                text=name,
                country=country,
                max_candidates=LSH_MAX_CANDIDATES,
                soft_country_fallback=BLOCKING_SOFT_COUNTRY_MODE,
            )
            for cid in lsh_cands:
                if cid not in cand_map:
                    src = "source2" if cid.startswith("S2-") else "source3"
                    cand_map[cid] = (src, "minhash_lsh", 50)

        # 6. Prefix Safety Net (Score: 40)
        if name and len(cand_map) < BLOCKING_FUZZY_MIN_CANDS:
            pfx = name[:4] if len(name) >= 4 else name
            if pfx in self.prefix_idx:
                for idx in self.prefix_idx[pfx][:BLOCKING_SAFETY_NET_MAX_BUCKET]:
                    add_cand(idx, "prefix_safety_net", 40)

        # Enforce deterministic per-shard cap to bound intermediate volume
        pruned_count = 0
        effective_cap = per_shard_cap if per_shard_cap is not None else max_candidates
        if len(cand_map) > effective_cap:
            pruned_count = len(cand_map) - effective_cap
            sorted_items = sorted(
                cand_map.items(),
                key=lambda item: (-item[1][2], item[0]),
            )[:effective_cap]
            cand_map = dict(sorted_items)

        if return_pruned_count:
            return cand_map, pruned_count
        return cand_map


def benchmark_shard_memory_rss(
    sample_df: pd.DataFrame,
    query_batch_df: Optional[pd.DataFrame] = None,
    shard_size_targets: int = SHARD_SIZE_TARGETS,
    safety_margin: float = RAM_SAFETY_MARGIN,
) -> Dict[str, Any]:
    """Measure provisional OS-level RSS scaling from a representative target slice and query batch.
    
    Includes DataFrame loading, postings, MinHash LSH, and query buffers.
    Note: This returns a provisional extrapolation estimate from a sample; full-shard peak
    RSS is recorded per shard dynamically during execution.
    """
    gc.collect()
    rss_before = get_process_rss_bytes() or 0

    shard = TargetIndexShard(shard_id=0)
    shard.build_from_dataframe(sample_df)

    # Query with representative batch if provided
    if query_batch_df is not None:
        recs = query_batch_df.to_dict(orient="records")
        for r in recs[:200]:
            shard.query_record(
                name=r.get("business_name_norm", r.get("business_name", "")),
                addr=r.get("business_address_norm", r.get("business_address", "")),
                country=r.get("country_norm", r.get("country", "")),
            )

    rss_after = get_process_rss_bytes() or rss_before
    rss_delta_bytes = max(1024 * 1024, rss_after - rss_before)
    sample_count = max(len(sample_df), 1)
    bytes_per_target = rss_delta_bytes / sample_count

    estimated_shard_rss_gb = (shard_size_targets * bytes_per_target * safety_margin) / (1024 ** 3)

    del shard
    gc.collect()

    return {
        "sample_size": sample_count,
        "base_rss_mb": round(rss_before / (1024 * 1024), 2),
        "peak_rss_mb": round(rss_after / (1024 * 1024), 2),
        "rss_delta_mb": round(rss_delta_bytes / (1024 * 1024), 2),
        "rss_bytes_per_target": round(bytes_per_target, 2),
        "safety_margin": safety_margin,
        "estimated_shard_rss_gb": round(estimated_shard_rss_gb, 3),
        "is_provisional_extrapolation": True,
    }


def measure_shard_memory(
    records_df: pd.DataFrame,
    sample_sizes: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Empirically measure index memory footprint across increasing shard sizes."""
    if sample_sizes is None:
        sample_sizes = [1000, 5000, 10000, 25000, 50000]

    measurements = []
    total_len = len(records_df)

    for sz in sample_sizes:
        if sz > total_len:
            continue
        slice_df = records_df.iloc[:sz]
        bench = benchmark_shard_memory_rss(sample_df=slice_df, shard_size_targets=sz)
        measurements.append({
            "target_count": sz,
            "measured_heap_mb": bench["rss_delta_mb"],
            "bytes_per_target": bench["rss_bytes_per_target"],
        })

    return {
        "measurements": measurements,
        "mean_bytes_per_target": round(float(np.mean([m["bytes_per_target"] for m in measurements])), 1) if measurements else 0.0,
    }


class ShardedTargetIndex:
    """Coordinator for partitioned target retrieval."""

    def __init__(self, shard_size: int = SHARD_SIZE_TARGETS):
        self.shard_size = shard_size
        self.total_target_records = 0
        self.shard_file_paths: List[Path] = []
        self.measured_shard_ram_bytes: List[int] = []
        self.shards: List[TargetIndexShard] = []  # For unit test backward compatibility only

    def add_shard(self, shard: TargetIndexShard) -> None:
        """Register a shard (used in in-memory test setups)."""
        self.shards.append(shard)
        self.total_target_records += len(shard.entity_ids)

    def query_record(
        self,
        name: str,
        addr: str,
        country: str,
        max_candidates: int = BLOCKING_MAX_CANDIDATES_PER_S1,
    ) -> Tuple[Set[str], List[Tuple[str, str, str]], bool]:
        """Query currently registered shards in memory (used in unit test verification)."""
        merged_cands: Dict[str, Tuple[str, str, int]] = {}

        for shard in self.shards:
            shard_cands = shard.query_record(name=name, addr=addr, country=country, max_candidates=max_candidates)
            for cid, (src, rule, score) in shard_cands.items():
                if cid not in merged_cands or score > merged_cands[cid][2]:
                    merged_cands[cid] = (src, rule, score)

        total_found = len(merged_cands)
        is_truncated = total_found > max_candidates

        if is_truncated:
            sorted_items = sorted(
                merged_cands.items(),
                key=lambda item: (-item[1][2], item[0]),
            )[:max_candidates]
            merged_cands = dict(sorted_items)

        cand_set = set(merged_cands.keys())
        prov_records = [
            (cid, src, rule)
            for cid, (src, rule, _) in merged_cands.items()
        ]

        return cand_set, prov_records, is_truncated
