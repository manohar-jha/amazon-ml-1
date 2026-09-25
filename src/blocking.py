"""Scalable multi-rule candidate generation engine for Entity Resolution.

Combines:
1. Inverted token indexing with posting caps and rare token prioritization.
2. Character 3-gram MinHash LSH for sublinear approximate name retrieval.
3. Bounded index-backed fuzzy fallback with soft country preference.
4. Address number and landmark key indexing.
5. Full candidate provenance tracking and validation recall evaluation.
"""

import gc
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
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
    BLOCKING_SAFETY_NET_MAX_BUCKET,
    BLOCKING_SOFT_COUNTRY_MODE,
    COMMON_LEGAL_TERMS,
    LSH_MAX_BUCKET_SIZE,
    LSH_MAX_CANDIDATES,
    LSH_NUM_BANDS,
    LSH_NUM_PERMUTATIONS,
    LSH_SHINGLE_N,
    TEST_CANDIDATES_PATH,
    TEST_PROVENANCE_PATH,
    TRAIN_CANDIDATES_PATH,
)
from src.data_loader import load_ground_truth, load_source_file, load_test_data, load_training_data
from src.lsh import MinHashLSH
from src.normalize import normalize_dataframe
from src.schemas import COL_CANDIDATE_IDS, COL_MATCHED_IDS, COL_SOURCE1_ID, VALID_TARGET_PREFIXES


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
    """Extract distinctive address signals (numbers and landmark words).

    Args:
        addr: Normalized address string.

    Returns:
        List[str]: Address composite keys.
    """
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


class MultiSourceCandidateIndex:
    """Multi-source candidate indexing engine with inverted indexes and MinHash LSH."""

    def __init__(self, country: Optional[str] = None):
        self.country = country
        self.exact_name_idx: Dict[str, List[str]] = defaultdict(list)
        self.core_name_idx: Dict[str, List[str]] = defaultdict(list)
        self.compact_name_idx: Dict[str, List[str]] = defaultdict(list)
        self.token_idx: Dict[str, List[str]] = defaultdict(list)
        self.pair_idx: Dict[str, List[str]] = defaultdict(list)
        self.addr_idx: Dict[str, List[str]] = defaultdict(list)
        self.prefix_idx: Dict[str, List[str]] = defaultdict(list)

        self.token_freq: Counter = Counter()
        self.addr_key_freq: Counter = Counter()

        # MinHash LSH instance for approximate name matching
        self.lsh = MinHashLSH(
            num_permutations=LSH_NUM_PERMUTATIONS,
            num_bands=LSH_NUM_BANDS,
            shingle_n=LSH_SHINGLE_N,
        )

        # Source mapping (ID -> 'source2' | 'source3')
        self.id_to_source: Dict[str, str] = {}

    def build_from_sources(
        self,
        s2_df: pd.DataFrame,
        s3_df: pd.DataFrame,
        max_token_freq: int = BLOCKING_MAX_TOKEN_DOC_FREQ,
        max_posting_len: int = BLOCKING_MAX_POSTING_LEN,
        max_pair_freq: int = BLOCKING_MAX_PAIR_TOKEN_FREQ,
        max_addr_freq: int = BLOCKING_MAX_ADDR_KEY_FREQ,
        max_bucket_size: int = BLOCKING_SAFETY_NET_MAX_BUCKET,
    ) -> None:
        """Build all inverted indexes and MinHash LSH from S2 and S3 DataFrames."""
        # 1. Compute document frequencies across S2 and S3
        for df in [s2_df, s3_df]:
            if "business_name_norm" in df.columns:
                for name in df["business_name_norm"]:
                    if name:
                        self.token_freq.update(set(name.split()))

            if "business_address_norm" in df.columns:
                for addr in df["business_address_norm"]:
                    if addr:
                        self.addr_key_freq.update(extract_address_keys(addr))

        # 2. Populate inverted indexes and MinHash LSH
        for source_label, df in [("source2", s2_df), ("source3", s3_df)]:
            country_series = df["country_norm"] if "country_norm" in df.columns else [""] * len(df)
            for eid, name, addr, cty in zip(
                df["entity_id"],
                df["business_name_norm"],
                df["business_address_norm"],
                country_series,
            ):
                self.id_to_source[eid] = source_label

                if name:
                    # Rule 1: Exact Name
                    if len(self.exact_name_idx[name]) < max_posting_len:
                        self.exact_name_idx[name].append(eid)

                    # Rule 1b: Core & Compact Names
                    core = get_core_name(name)
                    if core and core != name and len(self.core_name_idx[core]) < max_posting_len:
                        self.core_name_idx[core].append(eid)

                    compact = get_compact_name(name)
                    if compact and len(compact) >= 5 and len(self.compact_name_idx[compact]) < max_posting_len:
                        self.compact_name_idx[compact].append(eid)

                    # Rule 2: Rare Name Tokens (with posting cap)
                    tokens = [
                        t for t in set(name.split())
                        if len(t) >= BLOCKING_MIN_TOKEN_LEN and self.token_freq[t] <= max_token_freq
                    ]
                    for t in tokens:
                        if len(self.token_idx[t]) < max_posting_len:
                            self.token_idx[t].append(eid)

                    # Rule 3: 2-Token Combinations
                    all_toks = sorted([
                        t for t in set(name.split())
                        if len(t) >= BLOCKING_MIN_TOKEN_LEN and self.token_freq.get(t, 0) <= max_pair_freq
                    ])
                    for i in range(min(len(all_toks), 4)):
                        for j in range(i + 1, min(len(all_toks), 4)):
                            pair_key = f"{all_toks[i]}_{all_toks[j]}"
                            if len(self.pair_idx[pair_key]) < max_posting_len:
                                self.pair_idx[pair_key].append(eid)

                    # MinHash LSH indexing
                    self.lsh.index_entity(
                        entity_id=eid,
                        text=core if core else name,
                        country=cty if cty else (self.country or ""),
                        max_bucket_size=LSH_MAX_BUCKET_SIZE,
                    )

                    # Prefix safety net
                    if core and len(core) >= 4:
                        prefix_key = core[:4]
                        self.prefix_idx[prefix_key].append(eid)

                # Rule 4: Address Keys
                if addr:
                    for k in extract_address_keys(addr):
                        if self.addr_key_freq.get(k, 0) <= max_addr_freq:
                            if len(self.addr_idx[k]) < max_posting_len:
                                self.addr_idx[k].append(eid)

        # Filter prefix buckets that exceed safety threshold
        self.prefix_idx = {
            k: v for k, v in self.prefix_idx.items() if len(v) <= max_bucket_size
        }


def generate_candidates_with_provenance(
    name: str,
    addr: str,
    country: str,
    index: MultiSourceCandidateIndex,
    max_candidates: int = BLOCKING_MAX_CANDIDATES_PER_S1,
    soft_country_mode: bool = BLOCKING_SOFT_COUNTRY_MODE,
) -> Tuple[Set[str], Dict[str, Set[str]], bool]:
    """Generate candidate entity IDs and track rule provenance for an S1 record.

    Returns:
        Tuple[Set[str], Dict[str, Set[str]], bool]:
        - Set of candidate entity IDs.
        - Provenance mapping: candidate_id -> set of triggered rule names.
        - Truncated flag: True if candidates exceeded max_candidates.
    """
    candidates: Set[str] = set()
    provenance: Dict[str, Set[str]] = defaultdict(set)

    # 1. Exact Name & Core/Compact Name
    if name:
        for eid in index.exact_name_idx.get(name, []):
            candidates.add(eid)
            provenance[eid].add("exact_name")

        core = get_core_name(name)
        if core:
            for eid in index.exact_name_idx.get(core, []):
                candidates.add(eid)
                provenance[eid].add("core_name")
            for eid in index.core_name_idx.get(core, []):
                candidates.add(eid)
                provenance[eid].add("core_name")

            compact = get_compact_name(name)
            if compact and len(compact) >= 5:
                for eid in index.compact_name_idx.get(compact, []):
                    candidates.add(eid)
                    provenance[eid].add("compact_name")

    # 2. Rare Name Tokens (rarest first)
    if name:
        toks = [
            t for t in set(name.split())
            if len(t) >= BLOCKING_MIN_TOKEN_LEN and t in index.token_freq
        ]
        toks.sort(key=lambda t: index.token_freq[t])
        for t in toks[:BLOCKING_MAX_RARE_TOKENS_PER_S1]:
            if index.token_freq[t] <= BLOCKING_MAX_TOKEN_DOC_FREQ:
                for eid in index.token_idx.get(t, []):
                    candidates.add(eid)
                    provenance[eid].add("rare_token")

    # 3. 2-Token Combinations
    if name:
        all_toks = sorted([
            t for t in set(name.split())
            if len(t) >= BLOCKING_MIN_TOKEN_LEN and index.token_freq.get(t, 0) <= BLOCKING_MAX_PAIR_TOKEN_FREQ
        ])
        for i in range(min(len(all_toks), 4)):
            for j in range(i + 1, min(len(all_toks), 4)):
                pair_key = f"{all_toks[i]}_{all_toks[j]}"
                for eid in index.pair_idx.get(pair_key, []):
                    candidates.add(eid)
                    provenance[eid].add("token_pair")

    # 4. Character 3-gram MinHash LSH (Approximate Name Matches)
    if name:
        core_str = get_core_name(name)
        lsh_cands = index.lsh.query_candidates(
            text=core_str if core_str else name,
            country=country,
            max_candidates=LSH_MAX_CANDIDATES,
            soft_country_fallback=soft_country_mode,
        )
        for eid in lsh_cands:
            candidates.add(eid)
            provenance[eid].add("minhash_lsh")

    # 5. Address-Derived Signals
    if addr:
        for k in extract_address_keys(addr):
            if index.addr_key_freq.get(k, 0) <= BLOCKING_MAX_ADDR_KEY_FREQ:
                for eid in index.addr_idx.get(k, []):
                    candidates.add(eid)
                    provenance[eid].add("address_key")

    # 6. Bounded Fuzzy Fallback / Safety Net (for entities with few candidates)
    if len(candidates) < BLOCKING_FUZZY_MIN_CANDS and name:
        core = get_core_name(name)
        if core and len(core) >= 4:
            prefix_key = core[:4]
            for eid in index.prefix_idx.get(prefix_key, []):
                candidates.add(eid)
                provenance[eid].add("prefix_safety_net")

    is_truncated = len(candidates) > max_candidates
    if is_truncated:
        sorted_cands = sorted(candidates)[:max_candidates]
        final_cands = set(sorted_cands)
        filtered_provenance = {cid: provenance[cid] for cid in sorted_cands}
        return final_cands, filtered_provenance, True

    return candidates, provenance, False


def generate_candidate_union(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    max_candidates: int = BLOCKING_MAX_CANDIDATES_PER_S1,
    soft_country_mode: bool = BLOCKING_SOFT_COUNTRY_MODE,
    chunk_size: int = 50000,
) -> Tuple[Dict[str, Set[str]], List[Tuple[str, str, str, str]], Dict[str, Any]]:
    """Generate candidate union and pair-level provenance table across all S1 entities.

    Returns:
        Tuple containing:
        - candidates_map: S1 ID -> set of candidate IDs.
        - provenance_records: list of (s1_id, candidate_id, source_dataset, rule_provenance).
        - stats: summary execution statistics.
    """
    candidates_map: Dict[str, Set[str]] = {}
    provenance_records: List[Tuple[str, str, str, str]] = []
    total_truncations = 0
    candidate_counts: List[int] = []

    countries = s1_df["country_norm"].unique()

    for country in countries:
        s1_c = s1_df[s1_df["country_norm"] == country]
        s2_c = s2_df[s2_df["country_norm"] == country]
        s3_c = s3_df[s3_df["country_norm"] == country]

        if s1_c.empty:
            continue

        index = MultiSourceCandidateIndex(country=country)
        index.build_from_sources(s2_c, s3_c)

        for start_idx in range(0, len(s1_c), chunk_size):
            chunk = s1_c.iloc[start_idx : start_idx + chunk_size]
            for s1_id, name, addr in zip(
                chunk["entity_id"],
                chunk["business_name_norm"],
                chunk["business_address_norm"],
            ):
                cands, prov, truncated = generate_candidates_with_provenance(
                    name=name,
                    addr=addr,
                    country=country,
                    index=index,
                    max_candidates=max_candidates,
                    soft_country_mode=soft_country_mode,
                )
                candidates_map[s1_id] = cands
                candidate_counts.append(len(cands))
                if truncated:
                    total_truncations += 1

                for cid, rules in prov.items():
                    src = index.id_to_source.get(cid, "source2" if cid.startswith("S2-") else "source3")
                    rules_str = ",".join(sorted(rules))
                    provenance_records.append((s1_id, cid, src, rules_str))

        del index
        gc.collect()

    cand_arr = np.array(candidate_counts) if candidate_counts else np.array([0])
    stats = {
        "total_s1": len(s1_df),
        "total_candidate_pairs": len(provenance_records),
        "truncation_count": total_truncations,
        "truncation_rate": total_truncations / len(s1_df) if len(s1_df) else 0.0,
        "empty_candidate_count": int(np.sum(cand_arr == 0)),
        "empty_candidate_rate": float(np.mean(cand_arr == 0)),
        "mean_candidates": float(np.mean(cand_arr)),
        "median_candidates": float(np.median(cand_arr)),
        "p90_candidates": float(np.percentile(cand_arr, 90)),
        "p95_candidates": float(np.percentile(cand_arr, 95)),
        "p99_candidates": float(np.percentile(cand_arr, 99)),
        "max_candidates": int(np.max(cand_arr)),
    }

    return candidates_map, provenance_records, stats


def write_candidate_outputs(
    candidates_map: Dict[str, Set[str]],
    provenance_records: List[Tuple[str, str, str, str]],
    ordered_s1_ids: List[str],
    candidate_pairs_path: Union[Path, str] = TEST_CANDIDATES_PATH,
    provenance_path: Optional[Union[Path, str]] = TEST_PROVENANCE_PATH,
) -> None:
    """Serialize candidate pairs and pair-level provenance table to TSV.

    candidate_pairs.tsv schema: (source1_entity_id, candidate_entity_ids)
    candidate_provenance.tsv schema: (source1_entity_id, candidate_entity_id, source_dataset, provenance_rules)
    """
    pairs_file = Path(candidate_pairs_path)
    pairs_file.parent.mkdir(parents=True, exist_ok=True)

    # 1. Write candidate_pairs.tsv (1 row per S1 entity)
    with open(pairs_file, "w", encoding="utf-8", newline="") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in ordered_s1_ids:
            cands = sorted(candidates_map.get(s1_id, set()))
            f.write(f"{s1_id}\t{','.join(cands)}\n")

    # 2. Write pair-level provenance table
    if provenance_path:
        prov_file = Path(provenance_path)
        prov_file.parent.mkdir(parents=True, exist_ok=True)
        with open(prov_file, "w", encoding="utf-8", newline="") as f:
            f.write("source1_entity_id\tcandidate_entity_id\tsource_dataset\tprovenance_rules\n")
            for s1_id, cid, src, rules in provenance_records:
                f.write(f"{s1_id}\t{cid}\t{src}\t{rules}\n")


def evaluate_validation_recall(
    candidates_map: Dict[str, Set[str]],
    val_s1_ids: Set[str],
    gt_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Evaluate candidate recall strictly on the saved validation split."""
    val_gt = gt_df[gt_df["source1_entity_id"].isin(val_s1_ids)]
    val_cands = {s1_id: candidates_map.get(s1_id, set()) for s1_id in val_s1_ids}

    total_gt_pairs = 0
    total_s2_gt = 0
    total_s3_gt = 0
    found_gt_pairs = 0
    found_s2_gt = 0
    found_s3_gt = 0
    s1_full_covered = 0
    s1_with_matches = 0

    for _, row in val_gt.iterrows():
        s1_id = row["source1_entity_id"]
        m_str = row["matched_entity_ids"].strip()
        true_set = set(m_str.split(",")) if m_str else set()
        cands = val_cands.get(s1_id, set())

        if not true_set:
            continue

        s1_with_matches += 1
        total_gt_pairs += len(true_set)
        s2_matches = {m for m in true_set if m.startswith("S2-")}
        s3_matches = {m for m in true_set if m.startswith("S3-")}

        total_s2_gt += len(s2_matches)
        total_s3_gt += len(s3_matches)

        found = true_set.intersection(cands)
        found_gt_pairs += len(found)
        found_s2_gt += len(s2_matches.intersection(cands))
        found_s3_gt += len(s3_matches.intersection(cands))

        if len(found) == len(true_set):
            s1_full_covered += 1

    val_pair_recall = found_gt_pairs / total_gt_pairs if total_gt_pairs > 0 else 0.0
    val_s2_recall = found_s2_gt / total_s2_gt if total_s2_gt > 0 else 0.0
    val_s3_recall = found_s3_gt / total_s3_gt if total_s3_gt > 0 else 0.0
    val_full_s1_coverage = s1_full_covered / s1_with_matches if s1_with_matches > 0 else 0.0

    return {
        "val_total_gt_pairs": total_gt_pairs,
        "val_total_s2_gt": total_s2_gt,
        "val_total_s3_gt": total_s3_gt,
        "val_found_gt_pairs": found_gt_pairs,
        "val_found_s2_gt": found_s2_gt,
        "val_found_s3_gt": found_s3_gt,
        "val_pair_recall": val_pair_recall,
        "val_s2_recall": val_s2_recall,
        "val_s3_recall": val_s3_recall,
        "val_s1_with_matches": s1_with_matches,
        "val_s1_full_covered": s1_full_covered,
        "val_full_s1_coverage": val_full_s1_coverage,
    }


def validate_candidate_file(
    candidate_file_path: Union[Path, str],
    expected_s1_ids: Set[str],
    valid_target_ids: Optional[Set[str]] = None,
    is_test: bool = False,
) -> Dict[str, Any]:
    """Validate a candidate_pairs.tsv file against expected S1 and target IDs."""
    path = Path(candidate_file_path)
    if not path.exists():
        raise FileNotFoundError(f"Candidate file not found: {path}")

    seen_s1: Set[str] = set()
    rows_with_duplicate_cands = 0
    invalid_target_id_count = 0
    train_id_leaks = 0
    empty_cands_count = 0
    total_rows = 0

    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\r\n")
        expected_header = f"{COL_SOURCE1_ID}\t{COL_CANDIDATE_IDS}"
        if header != expected_header:
            raise ValueError(f"Header mismatch. Expected '{expected_header}', found '{header}'")

        for line in f:
            line_str = line.rstrip("\r\n")
            if not line_str:
                continue
            parts = line_str.split("\t")
            total_rows += 1
            s1_id = parts[0].strip()
            raw_cands = parts[1].strip() if len(parts) > 1 else ""

            seen_s1.add(s1_id)

            if not raw_cands:
                empty_cands_count += 1
                continue

            cands_list = [c.strip() for c in raw_cands.split(",") if c.strip()]
            if len(cands_list) != len(set(cands_list)):
                rows_with_duplicate_cands += 1

            for cid in cands_list:
                if not cid.startswith(VALID_TARGET_PREFIXES):
                    invalid_target_id_count += 1
                if valid_target_ids and cid not in valid_target_ids:
                    invalid_target_id_count += 1

    missing_s1_count = len(expected_s1_ids - seen_s1)

    return {
        "total_rows": total_rows,
        "unique_s1_ids": len(seen_s1),
        "missing_s1_count": missing_s1_count,
        "invalid_target_id_count": invalid_target_id_count,
        "rows_with_duplicate_cands": rows_with_duplicate_cands,
        "train_id_leaks": train_id_leaks,
        "empty_cands_count": empty_cands_count,
        "is_valid": (missing_s1_count == 0 and invalid_target_id_count == 0 and rows_with_duplicate_cands == 0),
    }


# Backward compatibility aliases
CountryBlockingIndex = MultiSourceCandidateIndex


def generate_candidates_for_record(
    name: str,
    addr: str,
    index: MultiSourceCandidateIndex,
    country: Optional[str] = None,
    max_candidates: int = BLOCKING_MAX_CANDIDATES_PER_S1,
    **kwargs,
) -> Set[str]:
    """Backward-compatible single-record candidate generator wrapper."""
    cands, _, _ = generate_candidates_with_provenance(
        name=name,
        addr=addr,
        country=country if country else (index.country or ""),
        index=index,
        max_candidates=max_candidates,
    )
    return cands


def generate_candidates(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    max_candidates: int = BLOCKING_MAX_CANDIDATES_PER_S1,
    **kwargs,
) -> Dict[str, Set[str]]:
    """Backward-compatible candidate generator wrapper returning candidates_map."""
    cands_map, _, _ = generate_candidate_union(
        s1_df=s1_df,
        s2_df=s2_df,
        s3_df=s3_df,
        max_candidates=max_candidates,
    )
    return cands_map


def stream_generate_and_save_candidates(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    output_path: Union[Path, str],
    max_candidates: int = BLOCKING_MAX_CANDIDATES_PER_S1,
    **kwargs,
) -> None:
    """Backward-compatible wrapper for candidate file serialization."""
    cands_map, prov_records, _ = generate_candidate_union(
        s1_df=s1_df,
        s2_df=s2_df,
        s3_df=s3_df,
        max_candidates=max_candidates,
    )
    write_candidate_outputs(
        candidates_map=cands_map,
        provenance_records=prov_records,
        ordered_s1_ids=s1_df["entity_id"].tolist(),
        candidate_pairs_path=output_path,
    )


def evaluate_blocking_recall(
    candidates_map: Dict[str, Set[str]],
    gt_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Backward-compatible evaluate recall function."""
    all_s1_ids = set(gt_df["source1_entity_id"])
    res = evaluate_validation_recall(candidates_map, all_s1_ids, gt_df)
    res["overall_pair_recall"] = res["val_pair_recall"]
    res["s2_pair_recall"] = res["val_s2_recall"]
    res["s3_pair_recall"] = res["val_s3_recall"]
    res["full_coverage_active_s1"] = res["val_full_s1_coverage"]
    res["total_gt_pairs"] = res["val_total_gt_pairs"]
    res["found_gt_pairs"] = res["val_found_gt_pairs"]
    res["empty_gt_total"] = len(all_s1_ids) - res["val_s1_with_matches"]
    res["empty_gt_with_cands"] = sum(1 for sid in all_s1_ids if sid not in gt_df[gt_df["matched_entity_ids"].str.strip() != ""]["source1_entity_id"].values and len(candidates_map.get(sid, set())) > 0)
    return res


from utils.validate_submission import validate_submission_file  # re-export

