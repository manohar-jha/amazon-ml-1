"""Data loading and inspection module for the Amazon Entity Resolution Challenge.

This module provides reusable functions to load and validate TSV source files
and ground truth annotations without altering the original raw data.
"""

import hashlib
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple, Union
import pandas as pd

from src.config import (
    EXPECTED_ID_PREFIXES,
    GROUND_TRUTH_EXPECTED_COLUMNS,
    SOURCE_EXPECTED_COLUMNS,
    TEST_SOURCE1_PATH,
    TEST_SOURCE2_PATH,
    TEST_SOURCE3_PATH,
    TRAIN_GROUND_TRUTH_PATH,
    TRAIN_SOURCE1_PATH,
    TRAIN_SOURCE2_PATH,
    TRAIN_SOURCE3_PATH,
)


def load_source_file(
    path: Union[Path, str],
    expected_columns: Optional[list[str]] = None,
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """Load a single TSV source file and validate its schema.

    Args:
        path: Path to the TSV source file.
        expected_columns: List of column names expected in the file.
            Defaults to SOURCE_EXPECTED_COLUMNS.
        nrows: Optional maximum number of rows to load.

    Returns:
        pd.DataFrame: Original DataFrame loaded from the TSV file.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If required columns are missing from the file.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Source file not found at: {file_path.resolve()}")

    if expected_columns is None:
        expected_columns = SOURCE_EXPECTED_COLUMNS

    # Always use explicit tab delimiter and preserve raw string representation
    df = pd.read_csv(file_path, sep="\t", dtype=str, nrows=nrows)

    missing_cols = [col for col in expected_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"File '{file_path.name}' is missing expected columns: {missing_cols}. "
            f"Found columns: {list(df.columns)}"
        )

    return df


def load_ground_truth(
    path: Union[Path, str],
    expected_columns: Optional[list[str]] = None,
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """Load the ground truth TSV file and validate its schema.

    Args:
        path: Path to the TSV ground truth file.
        expected_columns: List of column names expected in the file.
            Defaults to GROUND_TRUTH_EXPECTED_COLUMNS.
        nrows: Optional maximum number of rows to load.

    Returns:
        pd.DataFrame: Ground truth DataFrame preserving original values.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If required columns are missing from the file.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Ground truth file not found at: {file_path.resolve()}")

    if expected_columns is None:
        expected_columns = GROUND_TRUTH_EXPECTED_COLUMNS

    # Always load with explicit tab delimiter and preserve raw values
    df = pd.read_csv(file_path, sep="\t", dtype=str, keep_default_na=False, nrows=nrows)

    missing_cols = [col for col in expected_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Ground truth file '{file_path.name}' is missing expected columns: {missing_cols}. "
            f"Found columns: {list(df.columns)}"
        )

    return df


def deterministic_id_hash(entity_id: str, seed: int = 42) -> str:
    """Generate deterministic SHA-256 hash string for an entity ID given a seed."""
    return hashlib.sha256(f"{seed}_{entity_id}".encode("utf-8")).hexdigest()


def scan_and_sample_source_file(
    path: Union[Path, str],
    scan_rows: int = 250000,
    sample_size: int = 5000,
    chunksize: int = 50000,
    seed: int = 42,
    filter_ids: Optional[Set[str]] = None,
    expected_columns: Optional[list[str]] = None,
) -> Tuple[pd.DataFrame, int]:
    """Stream-scan a source TSV up to scan_rows and deterministically sample rows.

    Memory bounded: Processed in chunks of chunksize without loading the full file.
    Deterministic: Selected rows are chosen by sorting IDs on deterministic hash.

    Args:
        path: Path to the TSV file.
        scan_rows: Maximum rows to inspect from the start of the file.
        sample_size: Target number of sampled rows to return.
        chunksize: Number of rows per streaming pandas chunk.
        seed: Random seed for deterministic hash computation.
        filter_ids: Optional set of entity IDs to restrict selection to (e.g. validation split).
        expected_columns: Expected columns in the TSV.

    Returns:
        Tuple[pd.DataFrame, int]: Sampled DataFrame and total number of rows scanned.

    Raises:
        FileNotFoundError: If the source TSV does not exist.
        ValueError: If filter_ids is provided and no matching entities were found within scan_rows.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Source file not found at: {file_path.resolve()}")

    if expected_columns is None:
        expected_columns = SOURCE_EXPECTED_COLUMNS

    total_scanned = 0
    candidate_chunks: list[pd.DataFrame] = []

    # Read in bounded chunks with explicit tab separator and string dtype
    reader = pd.read_csv(
        file_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=chunksize,
    )

    try:
        for chunk in reader:
            # Validate columns on first chunk
            if total_scanned == 0:
                missing_cols = [col for col in expected_columns if col not in chunk.columns]
                if missing_cols:
                    raise ValueError(
                        f"File '{file_path.name}' is missing expected columns: {missing_cols}. "
                        f"Found columns: {list(chunk.columns)}"
                    )

            remaining_to_scan = scan_rows - total_scanned
            if remaining_to_scan <= 0:
                break

            if len(chunk) > remaining_to_scan:
                chunk = chunk.iloc[:remaining_to_scan]

            total_scanned += len(chunk)

            # Apply ID filter if specified (e.g. validation split S1 IDs)
            if filter_ids is not None:
                filtered_chunk = chunk[chunk["entity_id"].isin(filter_ids)]
            else:
                filtered_chunk = chunk

            if not filtered_chunk.empty:
                # Keep memory bounded: prune chunk to top sample_size by deterministic hash
                if len(filtered_chunk) > sample_size:
                    filtered_chunk = filtered_chunk.copy()
                    filtered_chunk["_hash"] = filtered_chunk["entity_id"].apply(
                        lambda eid: hashlib.sha256(f"{seed}_{eid}".encode("utf-8")).hexdigest()
                    )
                    filtered_chunk = (
                        filtered_chunk.sort_values("_hash")
                        .head(sample_size)
                        .drop(columns=["_hash"])
                    )

                candidate_chunks.append(filtered_chunk)

                # Periodically consolidate candidate chunks to keep memory footprint minimal
                if len(candidate_chunks) > 5:
                    merged = pd.concat(candidate_chunks, ignore_index=True)
                    if len(merged) > sample_size:
                        merged["_hash"] = merged["entity_id"].apply(
                            lambda eid: hashlib.sha256(f"{seed}_{eid}".encode("utf-8")).hexdigest()
                        )
                        merged = (
                            merged.sort_values("_hash")
                            .head(sample_size)
                            .drop(columns=["_hash"])
                        )
                    candidate_chunks = [merged]

            if total_scanned >= scan_rows:
                break
    finally:
        reader.close()

    if not candidate_chunks:
        if filter_ids is not None:
            raise ValueError(
                f"No validation S1 IDs were found in the initial {total_scanned:,} scanned rows of '{file_path.name}'. "
                f"Try increasing --pilot-scan-rows (e.g. --pilot-scan-rows {max(scan_rows * 2, 500000)}) or adjusting --pilot-chunksize."
            )
        return pd.DataFrame(columns=expected_columns), total_scanned

    final_df = pd.concat(candidate_chunks, ignore_index=True)
    if len(final_df) > sample_size:
        final_df["_hash"] = final_df["entity_id"].apply(
            lambda eid: hashlib.sha256(f"{seed}_{eid}".encode("utf-8")).hexdigest()
        )
        final_df = (
            final_df.sort_values("_hash")
            .head(sample_size)
            .drop(columns=["_hash"])
            .reset_index(drop=True)
        )
    else:
        final_df = final_df.reset_index(drop=True)

    return final_df, total_scanned



def load_training_data(
    train_dir: Optional[Union[Path, str]] = None,
) -> Dict[str, pd.DataFrame]:
    """Load all training datasets and ground truth.

    Args:
        train_dir: Optional custom training directory path. Defaults to TRAIN_DIR.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary containing 'source1', 'source2',
        'source3', and 'ground_truth' DataFrames.
    """
    base_dir = Path(train_dir) if train_dir else TRAIN_DIR
    return {
        "source1": load_source_file(base_dir / "train_source1.tsv"),
        "source2": load_source_file(base_dir / "train_source2.tsv"),
        "source3": load_source_file(base_dir / "train_source3.tsv"),
        "ground_truth": load_ground_truth(base_dir / "train_ground_truth.tsv"),
    }


def load_test_data(
    test_dir: Optional[Union[Path, str]] = None,
) -> Dict[str, pd.DataFrame]:
    """Load all test source datasets.

    Args:
        test_dir: Optional custom test directory path. Defaults to TEST_DIR.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary containing 'source1', 'source2',
        and 'source3' DataFrames.
    """
    base_dir = Path(test_dir) if test_dir else TEST_DIR
    return {
        "source1": load_source_file(base_dir / "test_source1.tsv"),
        "source2": load_source_file(base_dir / "test_source2.tsv"),
        "source3": load_source_file(base_dir / "test_source3.tsv"),
    }


def inspect_source_dataframe(
    df: pd.DataFrame,
    expected_prefix: Optional[str] = None,
) -> Dict[str, Any]:
    """Extract summary metrics and validation information for a source DataFrame.

    Args:
        df: The source DataFrame to inspect.
        expected_prefix: Optional expected ID prefix (e.g. 'S1-').

    Returns:
        Dict[str, Any]: Dictionary containing statistical counts, missing values,
        ID uniqueness, prefix anomalies, and generic country counts.
    """
    num_rows, num_cols = df.shape
    missing_counts = df.isna().sum().to_dict()
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}

    # Entity ID inspection
    entity_id_series = df["entity_id"] if "entity_id" in df.columns else pd.Series(dtype=str)
    unique_ids = int(entity_id_series.nunique())
    duplicate_ids = int(entity_id_series.duplicated().sum())

    # ID Prefix anomaly checking
    prefix_anomalies: list[str] = []
    if expected_prefix and "entity_id" in df.columns:
        non_matching_mask = ~entity_id_series.str.startswith(expected_prefix, na=False)
        prefix_anomalies = entity_id_series[non_matching_mask].dropna().tolist()

    # Country value counts (generic string/categorical, no hardcoded countries)
    country_counts: Dict[str, int] = {}
    if "country" in df.columns:
        country_counts = df["country"].fillna("<MISSING>").value_counts().to_dict()

    return {
        "num_rows": num_rows,
        "num_cols": num_cols,
        "columns": list(df.columns),
        "dtypes": dtypes,
        "missing_values": missing_counts,
        "unique_entity_ids": unique_ids,
        "duplicate_entity_ids": duplicate_ids,
        "expected_prefix": expected_prefix,
        "prefix_anomaly_count": len(prefix_anomalies),
        "prefix_anomaly_samples": prefix_anomalies[:5],
        "country_counts": country_counts,
    }


def inspect_ground_truth_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """Extract summary metrics and validation information for a ground truth DataFrame.

    Args:
        df: The ground truth DataFrame to inspect.

    Returns:
        Dict[str, Any]: Dictionary containing counts, missing values, duplicate
        source1 IDs, and empty/non-empty match list counts.
    """
    num_rows, num_cols = df.shape
    missing_counts = df.isna().sum().to_dict()

    s1_series = df["source1_entity_id"] if "source1_entity_id" in df.columns else pd.Series(dtype=str)
    unique_s1_ids = int(s1_series.nunique())
    duplicate_s1_ids = int(s1_series.duplicated().sum())

    empty_matches = 0
    non_empty_matches = 0

    if "matched_entity_ids" in df.columns:
        # Check empty strings, whitespace-only, or NaN
        matches_series = df["matched_entity_ids"].fillna("").astype(str).str.strip()
        empty_matches = int((matches_series == "").sum())
        non_empty_matches = int((matches_series != "").sum())

    return {
        "num_rows": num_rows,
        "num_cols": num_cols,
        "columns": list(df.columns),
        "missing_values": missing_counts,
        "unique_source1_ids": unique_s1_ids,
        "duplicate_source1_ids": duplicate_s1_ids,
        "empty_matches_count": empty_matches,
        "non_empty_matches_count": non_empty_matches,
    }


def print_source_summary(
    df: pd.DataFrame,
    title: str,
    expected_prefix: Optional[str] = None,
    sample_size: int = 3,
) -> None:
    """Print formatted inspection summary for a source DataFrame.

    Args:
        df: Source DataFrame.
        title: Header title to display.
        expected_prefix: Expected entity ID prefix string (e.g., 'S1-').
        sample_size: Number of sample rows to display.
    """
    metrics = inspect_source_dataframe(df, expected_prefix=expected_prefix)

    print("=" * 60)
    print(title)
    print("=" * len(title))
    print(f"Rows: {metrics['num_rows']:,}")
    print(f"Columns: {metrics['num_cols']} ({', '.join(metrics['columns'])})")
    print(f"Data types: {metrics['dtypes']}")
    print("Missing values:")
    for col, count in metrics["missing_values"].items():
        print(f"  - {col}: {count:,}")
    print(f"Unique entity_ids: {metrics['unique_entity_ids']:,}")
    print(f"Duplicate entity_ids: {metrics['duplicate_entity_ids']:,}")

    if expected_prefix:
        if metrics["prefix_anomaly_count"] == 0:
            print(f"Prefix validation: All IDs match expected prefix '{expected_prefix}'")
        else:
            print(
                f"Prefix validation: Found {metrics['prefix_anomaly_count']:,} IDs "
                f"not starting with '{expected_prefix}'."
            )
            print(f"  Example anomalies: {metrics['prefix_anomaly_samples']}")

    print("Countries:")
    for country, count in metrics["country_counts"].items():
        print(f"  - {country}: {count:,}")

    print(f"\nSample head ({min(sample_size, len(df))} rows):")
    print(df.head(sample_size).to_string(index=False))
    print()


def print_ground_truth_summary(
    df: pd.DataFrame,
    title: str = "TRAIN GROUND TRUTH",
    sample_size: int = 3,
) -> None:
    """Print formatted inspection summary for the ground truth DataFrame.

    Args:
        df: Ground truth DataFrame.
        title: Header title to display.
        sample_size: Number of sample rows to display.
    """
    metrics = inspect_ground_truth_dataframe(df)

    print("=" * 60)
    print(title)
    print("=" * len(title))
    print(f"Rows: {metrics['num_rows']:,}")
    print(f"Columns: {metrics['num_cols']} ({', '.join(metrics['columns'])})")
    print("Missing values:")
    for col, count in metrics["missing_values"].items():
        print(f"  - {col}: {count:,}")
    print(f"Unique source1_entity_ids: {metrics['unique_source1_ids']:,}")
    print(f"Duplicate source1_entity_ids: {metrics['duplicate_source1_ids']:,}")
    print(f"Empty match lists: {metrics['empty_matches_count']:,}")
    print(f"Non-empty match lists: {metrics['non_empty_matches_count']:,}")

    print(f"\nSample head ({min(sample_size, len(df))} rows):")
    print(df.head(sample_size).to_string(index=False))
    print()


def run_inspection() -> None:
    """Run full Phase 1 inspection for all training and test files."""
    files_to_check = [
        ("TRAIN SOURCE 1", TRAIN_SOURCE1_PATH, EXPECTED_ID_PREFIXES.get("source1"), "source"),
        ("TRAIN SOURCE 2", TRAIN_SOURCE2_PATH, EXPECTED_ID_PREFIXES.get("source2"), "source"),
        ("TRAIN SOURCE 3", TRAIN_SOURCE3_PATH, EXPECTED_ID_PREFIXES.get("source3"), "source"),
        ("TRAIN GROUND TRUTH", TRAIN_GROUND_TRUTH_PATH, None, "ground_truth"),
        ("TEST SOURCE 1", TEST_SOURCE1_PATH, EXPECTED_ID_PREFIXES.get("source1"), "source"),
        ("TEST SOURCE 2", TEST_SOURCE2_PATH, EXPECTED_ID_PREFIXES.get("source2"), "source"),
        ("TEST SOURCE 3", TEST_SOURCE3_PATH, EXPECTED_ID_PREFIXES.get("source3"), "source"),
    ]

    print("\n" + "#" * 60)
    print("# AMAZON ENTITY RESOLUTION - PHASE 1 DATA INSPECTION")
    print("#" * 60 + "\n")

    missing_files: list[Path] = []

    for title, path, prefix, file_type in files_to_check:
        if not path.exists():
            missing_files.append(path)
            print(f"[MISSING] {title}: Expected file not found at '{path}'")
            continue

        try:
            if file_type == "source":
                df = load_source_file(path)
                print_source_summary(df, title, expected_prefix=prefix)
            elif file_type == "ground_truth":
                df = load_ground_truth(path)
                print_ground_truth_summary(df, title)
        except Exception as e:
            print(f"[ERROR] Failed to load/inspect {title} ({path}): {e}")

    if missing_files:
        print("-" * 60)
        print(f"Note: {len(missing_files)} expected file(s) are not present yet.")
        print("Please place the TSV files into their respective directories:")
        print("  - data/train/ (train_source1.tsv, train_source2.tsv, train_source3.tsv, train_ground_truth.tsv)")
        print("  - data/test/  (test_source1.tsv, test_source2.tsv, test_source3.tsv)")
        print("-" * 60)


if __name__ == "__main__":
    run_inspection()
