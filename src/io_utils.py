"""I/O utilities for robust TSV reading, writing, and ID list serialization.

Enforces:
- Explicit tab delimiter (sep='\\t') across all files.
- Addresses are NEVER parsed by splitting on commas (commas are only for ID list columns).
- ID list parsing/serialization: deterministic sorting, unique IDs, empty string for no-matches.
- Strict column schema validation.
"""

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Union
import pandas as pd

from src.schemas import (
    COL_CANDIDATE_IDS,
    COL_ENTITY_ID,
    COL_MATCHED_IDS,
    COL_SOURCE1_ID,
    PREFIX_SOURCE1,
    PREFIX_SOURCE2,
    PREFIX_SOURCE3,
    VALID_TARGET_PREFIXES,
)


def parse_id_list(raw_value: Optional[Union[str, float]]) -> List[str]:
    """Parse comma-separated ID string into a clean list of unique IDs.

    Used ONLY for matched_entity_ids or candidate_entity_ids columns.
    Never use this on raw business addresses or text fields.

    Args:
        raw_value: Raw string from TSV (e.g. 'S2-1,S3-2' or '' or NaN).

    Returns:
        List[str]: List of unique trimmed entity IDs in deterministic order.
    """
    if raw_value is None or pd.isna(raw_value):
        return []

    str_val = str(raw_value).strip()
    if not str_val:
        return []

    tokens = [t.strip() for t in str_val.split(",") if t.strip()]
    # Deduplicate while preserving deterministic order
    seen: Set[str] = set()
    unique_ids: List[str] = []
    for token in tokens:
        if token not in seen:
            seen.add(token)
            unique_ids.append(token)

    return unique_ids


def serialize_id_list(ids: Optional[Iterable[str]]) -> str:
    """Serialize an iterable of entity IDs into a deterministic comma-separated string.

    Empty collections serialize to '' (empty string).
    IDs are sorted deterministically for reproducibility.

    Args:
        ids: Iterable of entity IDs (e.g. ['S2-20', 'S3-10'] or empty).

    Returns:
        str: Comma-separated string with no trailing commas (e.g. 'S2-20,S3-10' or '').
    """
    if not ids:
        return ""

    # Clean, deduplicate, and sort
    clean_ids = sorted({str(i).strip() for i in ids if str(i).strip()})
    return ",".join(clean_ids)


def validate_id_prefix(entity_id: str, expected_prefix: Optional[str] = None) -> bool:
    """Check if an entity ID follows expected source prefix convention (S1-, S2-, S3-)."""
    if not entity_id or not isinstance(entity_id, str):
        return False
    if expected_prefix:
        return entity_id.startswith(expected_prefix)
    return entity_id.startswith((PREFIX_SOURCE1, PREFIX_SOURCE2, PREFIX_SOURCE3))


def read_tsv(
    path: Union[Path, str],
    expected_columns: Optional[List[str]] = None,
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """Read a TSV file with explicit tab separation and schema validation.

    Args:
        path: Path to the TSV file.
        expected_columns: Optional list of expected column names.
        nrows: Optional row limit for smoke testing.

    Returns:
        pd.DataFrame: DataFrame with all columns loaded as strings without NA coercion.

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If required columns are missing.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"TSV file not found at: {file_path.resolve()}")

    df = pd.read_csv(
        file_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        nrows=nrows,
    )

    if expected_columns:
        missing = [col for col in expected_columns if col not in df.columns]
        if missing:
            raise ValueError(
                f"File '{file_path.name}' is missing expected columns: {missing}. "
                f"Found: {list(df.columns)}"
            )

    return df


def write_tsv(
    df: pd.DataFrame,
    path: Union[Path, str],
    expected_columns: Optional[List[str]] = None,
) -> None:
    """Write DataFrame to TSV with UTF-8 encoding and schema check.

    Args:
        df: DataFrame to serialize.
        path: Destination TSV path.
        expected_columns: Optional columns to enforce.
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if expected_columns:
        missing = [col for col in expected_columns if col not in df.columns]
        if missing:
            raise ValueError(f"DataFrame is missing required output columns: {missing}")
        out_df = df[expected_columns]
    else:
        out_df = df

    out_df.to_csv(file_path, sep="\t", index=False, encoding="utf-8")


def write_id_map_to_tsv(
    id_map: Dict[str, Union[List[str], Set[str]]],
    ordered_s1_ids: List[str],
    path: Union[Path, str],
    id_column_name: str = COL_MATCHED_IDS,
) -> None:
    """Write an S1-to-target-ID mapping directly to a TSV file with guaranteed 1-row-per-S1 coverage.

    Args:
        id_map: Dict mapping source1_entity_id -> list/set of candidate or matched IDs.
        ordered_s1_ids: Complete list of all S1 IDs to guarantee exact coverage and ordering.
        path: Output file path.
        id_column_name: Second column name ('matched_entity_ids' or 'candidate_entity_ids').
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w", encoding="utf-8", newline="") as f:
        f.write(f"{COL_SOURCE1_ID}\t{id_column_name}\n")
        for s1_id in ordered_s1_ids:
            target_ids = id_map.get(s1_id, [])
            serialized = serialize_id_list(target_ids)
            f.write(f"{s1_id}\t{serialized}\n")
