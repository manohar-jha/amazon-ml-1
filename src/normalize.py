"""Data normalization module for the Amazon Entity Resolution Challenge.

This module provides high-performance, Unicode-safe text, business name, address,
and country normalization while strictly preserving original raw columns and
multilingual content (e.g. Devanagari script, French accented characters).
"""

import re
import sys
import time
import unicodedata
from typing import Any, Dict, Optional, Union
import pandas as pd

from src.config import (
    EXPECTED_ID_PREFIXES,
    TEST_SOURCE1_PATH,
    TEST_SOURCE2_PATH,
    TEST_SOURCE3_PATH,
    TRAIN_GROUND_TRUTH_PATH,
    TRAIN_SOURCE1_PATH,
    TRAIN_SOURCE2_PATH,
    TRAIN_SOURCE3_PATH,
)
from src.data_loader import load_ground_truth, load_source_file


# Precompute lookup table for Unicode punctuation, symbols, and control chars
# This maps codepoints to a single space, preserving letters (L*), numbers (N*),
# combining marks/matras (M*), and whitespace.
_UNICODE_PUNCT_SYM_TABLE: dict[int, str] = {
    i: " "
    for i in range(0x110000)
    if unicodedata.category(chr(i)).startswith(("P", "S", "C"))
    and chr(i) not in ("\t", "\n", "\r")
}

_WHITESPACE_REGEX = re.compile(r"\s+")


def normalize_text(
    data: Union[str, pd.Series, None],
) -> Union[str, pd.Series]:
    """Normalize text using Unicode-safe transformations.

    Applies NFKC normalization, Unicode casefolding, punctuation/symbol replacement,
    and whitespace collapsing while fully preserving multilingual alphabets (Devanagari,
    Latin accents), numeric tokens, and combining marks.

    Args:
        data: A single string, None, or a pandas Series of strings.

    Returns:
        Union[str, pd.Series]: Cleaned, normalized string or pandas Series.
    """
    if data is None:
        return ""

    if isinstance(data, str):
        if not data.strip():
            return ""
        norm_str = unicodedata.normalize("NFKC", data)
        norm_str = norm_str.casefold()
        norm_str = norm_str.translate(_UNICODE_PUNCT_SYM_TABLE)
        norm_str = _WHITESPACE_REGEX.sub(" ", norm_str).strip()
        return norm_str

    if isinstance(data, pd.Series):
        # Vectorized string pipeline for high throughput on large series
        series = data.fillna("").astype(str)
        normalized = (
            series.str.normalize("NFKC")
            .str.casefold()
            .str.translate(_UNICODE_PUNCT_SYM_TABLE)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
        return normalized

    # Fallback for non-string scalar types
    return normalize_text(str(data))


def normalize_business_name(
    data: Union[str, pd.Series, None],
) -> Union[str, pd.Series]:
    """Normalize business names without aggressively stripping legal suffixes.

    Args:
        data: Business name string, None, or pandas Series.

    Returns:
        Union[str, pd.Series]: Normalized business name(s).
    """
    return normalize_text(data)


def normalize_address(
    data: Union[str, pd.Series, None],
) -> Union[str, pd.Series]:
    """Normalize business addresses while preserving numbers and multilingual tokens.

    Args:
        data: Address string, None, or pandas Series.

    Returns:
        Union[str, pd.Series]: Normalized address(es).
    """
    return normalize_text(data)


def normalize_country(
    data: Union[str, pd.Series, None],
) -> Union[str, pd.Series]:
    """Normalize country field with lightweight formatting (strip + casefold).

    Does not hardcode or restrict country values, allowing US, India, France,
    and any arbitrary country to remain valid.

    Args:
        data: Country string, None, or pandas Series.

    Returns:
        Union[str, pd.Series]: Normalized country string(s).
    """
    if data is None:
        return ""

    if isinstance(data, str):
        return unicodedata.normalize("NFKC", data).strip().casefold()

    if isinstance(data, pd.Series):
        return data.fillna("").astype(str).str.normalize("NFKC").str.strip().str.casefold()

    return str(data).strip().casefold()


def normalize_dataframe(
    df: pd.DataFrame,
    inplace: bool = False,
) -> pd.DataFrame:
    """Add normalized columns to a source DataFrame without overwriting original columns.

    Original columns ('entity_id', 'business_name', 'business_address', 'country')
    remain untouched. Adds 'business_name_norm', 'business_address_norm', and
    'country_norm'.

    Args:
        df: Source DataFrame containing entity records.
        inplace: Whether to add columns directly to the input DataFrame.

    Returns:
        pd.DataFrame: DataFrame containing both original and normalized columns.
    """
    target_df = df if inplace else df.copy()

    if "business_name" in target_df.columns:
        target_df["business_name_norm"] = normalize_business_name(target_df["business_name"])
    else:
        target_df["business_name_norm"] = ""

    if "business_address" in target_df.columns:
        target_df["business_address_norm"] = normalize_address(target_df["business_address"])
    else:
        target_df["business_address_norm"] = ""

    if "country" in target_df.columns:
        target_df["country_norm"] = normalize_country(target_df["country"])
    else:
        target_df["country_norm"] = ""

    return target_df


def inspect_normalization(
    df_norm: pd.DataFrame,
    source_name: str,
    num_examples: int = 10,
) -> Dict[str, Any]:
    """Generate detailed metrics and before/after comparisons for a normalized DataFrame.

    Args:
        df_norm: Normalized DataFrame.
        source_name: Title/label of the dataset.
        num_examples: Number of before/after examples to generate.

    Returns:
        Dict[str, Any]: Detailed metrics dictionary.
    """
    total_rows = len(df_norm)
    missing_before = {
        "business_name": int(df_norm["business_name"].isna().sum()) if "business_name" in df_norm.columns else 0,
        "business_address": int(df_norm["business_address"].isna().sum()) if "business_address" in df_norm.columns else 0,
        "country": int(df_norm["country"].isna().sum()) if "country" in df_norm.columns else 0,
    }
    missing_after = {
        "business_name_norm": int(df_norm["business_name_norm"].isna().sum()),
        "business_address_norm": int(df_norm["business_address_norm"].isna().sum()),
        "country_norm": int(df_norm["country_norm"].isna().sum()),
    }
    empty_names = int((df_norm["business_name_norm"] == "").sum())
    empty_addresses = int((df_norm["business_address_norm"] == "").sum())
    country_dist = df_norm["country_norm"].value_counts().to_dict()

    # Select representative examples: include non-empty and multilingual rows
    sample_indices: list[int] = []
    
    # Priority 1: France or India (multilingual / accented)
    multilingual_mask = df_norm["country_norm"].isin(["france", "india"])
    if multilingual_mask.any():
        sample_indices.extend(df_norm[multilingual_mask].index[:6].tolist())

    # Priority 2: US
    us_mask = df_norm["country_norm"] == "us"
    if us_mask.any():
        sample_indices.extend(df_norm[us_mask].index[:4].tolist())

    # Fallback to head if not enough samples
    if len(sample_indices) < num_examples:
        for idx in df_norm.index[:num_examples]:
            if idx not in sample_indices:
                sample_indices.append(idx)

    sample_indices = sample_indices[:num_examples]

    name_examples = [
        (df_norm.loc[idx, "business_name"], df_norm.loc[idx, "business_name_norm"])
        for idx in sample_indices
        if "business_name" in df_norm.columns
    ]

    addr_examples = [
        (df_norm.loc[idx, "business_address"], df_norm.loc[idx, "business_address_norm"])
        for idx in sample_indices
        if "business_address" in df_norm.columns
    ]

    return {
        "source_name": source_name,
        "total_rows": total_rows,
        "missing_before": missing_before,
        "missing_after": missing_after,
        "empty_names": empty_names,
        "empty_addresses": empty_addresses,
        "country_dist": country_dist,
        "name_examples": name_examples,
        "addr_examples": addr_examples,
    }


def print_normalization_report(metrics: Dict[str, Any]) -> None:
    """Print clean formatted normalization report for a dataset."""
    print("=" * 70)
    print(metrics["source_name"])
    print("=" * len(metrics["source_name"]))
    print(f"Total Rows: {metrics['total_rows']:,}")
    print("Missing Values Before Normalization:")
    for col, cnt in metrics["missing_before"].items():
        print(f"  - {col}: {cnt:,}")
    print("Missing (NaN) Values After Normalization:")
    for col, cnt in metrics["missing_after"].items():
        print(f"  - {col}: {cnt:,}")
    print(f"Empty Normalized Names ('')     : {metrics['empty_names']:,}")
    print(f"Empty Normalized Addresses ('') : {metrics['empty_addresses']:,}")
    print("Country Distribution (country_norm):")
    for country, count in metrics["country_dist"].items():
        pct = (count / metrics["total_rows"]) * 100
        print(f"  - {country}: {count:,} ({pct:.2f}%)")

    print("\n--- 10 Representative Before/After Business Names ---")
    for i, (orig, norm) in enumerate(metrics["name_examples"], 1):
        orig_repr = repr(orig) if pd.notna(orig) else "<MISSING / NaN>"
        print(f"{i:2d}. [RAW]  {orig_repr}")
        print(f"    [NORM] {repr(norm)}")

    print("\n--- 10 Representative Before/After Business Addresses ---")
    for i, (orig, norm) in enumerate(metrics["addr_examples"], 1):
        orig_repr = repr(orig) if pd.notna(orig) else "<MISSING / NaN>"
        print(f"{i:2d}. [RAW]  {orig_repr}")
        print(f"    [NORM] {repr(norm)}")
    print()


def run_normalization_pipeline() -> None:
    """Execute Phase 2 normalization inspection across all 7 challenge datasets."""
    # Ensure stdout handles UTF-8 characters across all environments
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    sources_to_process = [
        ("TRAIN SOURCE 1", TRAIN_SOURCE1_PATH),
        ("TRAIN SOURCE 2", TRAIN_SOURCE2_PATH),
        ("TRAIN SOURCE 3", TRAIN_SOURCE3_PATH),
        ("TEST SOURCE 1", TEST_SOURCE1_PATH),
        ("TEST SOURCE 2", TEST_SOURCE2_PATH),
        ("TEST SOURCE 3", TEST_SOURCE3_PATH),
    ]

    print("\n" + "#" * 70)
    print("# AMAZON ENTITY RESOLUTION - PHASE 2 DATA NORMALIZATION INSPECTION")
    print("#" * 70 + "\n")

    overall_start = time.time()

    for name, path in sources_to_process:
        if not path.exists():
            print(f"[MISSING] {name}: File not found at '{path}'")
            continue

        t0 = time.time()
        # Load raw file without altering original schema
        df = load_source_file(path)
        load_time = time.time() - t0

        t1 = time.time()
        # Apply non-destructive normalization (adds *_norm columns)
        df_norm = normalize_dataframe(df, inplace=True)
        norm_time = time.time() - t1

        # Generate and print inspection report
        report = inspect_normalization(df_norm, name)
        print_normalization_report(report)
        print(f"Timing: Loaded in {load_time:.2f}s, Normalized in {norm_time:.2f}s\n")

        # Explicitly release memory
        del df, df_norm

    # Validate ground truth
    if TRAIN_GROUND_TRUTH_PATH.exists():
        gt_df = load_ground_truth(TRAIN_GROUND_TRUTH_PATH)
        print("=" * 70)
        print("TRAIN GROUND TRUTH VALIDATION")
        print("=" * 29)
        print(f"Total Rows: {len(gt_df):,}")
        empty_matches = int((gt_df["matched_entity_ids"].str.strip() == "").sum())
        non_empty_matches = int((gt_df["matched_entity_ids"].str.strip() != "").sum())
        print(f"Empty Match Lists     : {empty_matches:,}")
        print(f"Non-Empty Match Lists : {non_empty_matches:,}")
        print(f"Original Columns Intact: {list(gt_df.columns)}")
        print()
        del gt_df

    total_elapsed = time.time() - overall_start
    print(f"Phase 2 normalization inspection completed in {total_elapsed:.2f}s total.")


if __name__ == "__main__":
    run_normalization_pipeline()
