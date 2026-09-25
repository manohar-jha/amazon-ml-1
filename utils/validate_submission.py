"""Submission and candidate pair validation utility for Amazon Entity Resolution Challenge.

Validates:
1. TSV formatting and explicit tab delimiter.
2. Exact column schema: (source1_entity_id, matched_entity_ids) or (source1_entity_id, candidate_entity_ids).
3. Exactly one row per test Source 1 ID (zero duplicates, zero missing).
4. No duplicate target IDs within any single row.
5. All predicted target IDs use valid S2-/S3- prefixes.
6. Containment constraint: Every ID in matching_results.tsv MUST exist in candidate_pairs.tsv.
7. Empty string for singletons / no-match entities.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io_utils import parse_id_list
from src.schemas import (
    COL_CANDIDATE_IDS,
    COL_MATCHED_IDS,
    COL_SOURCE1_ID,
    VALID_TARGET_PREFIXES,
)


def validate_submission_file(
    submission_path: Path,
    expected_s1_ids: Set[str],
    candidate_pairs_path: Optional[Path] = None,
    valid_target_ids: Optional[Set[str]] = None,
    is_candidate_file: bool = False,
) -> Tuple[bool, List[str]]:
    """Validate a submission or candidate pairs TSV file.

    Args:
        submission_path: Path to matching_results.tsv or candidate_pairs.tsv.
        expected_s1_ids: Set of expected test Source 1 entity IDs.
        candidate_pairs_path: Optional path to candidate_pairs.tsv to verify containment.
        valid_target_ids: Optional set of known valid test target IDs (S2/S3).
        is_candidate_file: Whether the file being checked is candidate_pairs.tsv.

    Returns:
        Tuple[bool, List[str]]: (is_valid, list of error messages).
    """
    errors: List[str] = []
    expected_second_col = COL_CANDIDATE_IDS if is_candidate_file else COL_MATCHED_IDS
    expected_header = f"{COL_SOURCE1_ID}\t{expected_second_col}"

    if not submission_path.exists():
        return False, [f"File not found: {submission_path.resolve()}"]

    seen_s1: Set[str] = set()
    predictions_map: Dict[str, Set[str]] = {}
    duplicate_targets_count = 0
    invalid_prefix_count = 0
    invalid_target_id_count = 0

    with open(submission_path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\r\n")
        if header != expected_header:
            errors.append(
                f"Header mismatch in '{submission_path.name}'. "
                f"Found: '{header}', Expected: '{expected_header}'"
            )

        for line_num, line in enumerate(f, 2):
            line_str = line.rstrip("\r\n")
            parts = line_str.split("\t")
            if len(parts) > 2:
                errors.append(f"Line {line_num} has {len(parts)} columns (expected 2 tab-separated columns).")
                continue

            s1_id = parts[0].strip()
            raw_targets = parts[1].strip() if len(parts) == 2 else ""

            if not s1_id:
                errors.append(f"Line {line_num} has empty source1_entity_id.")
                continue

            if s1_id in seen_s1:
                errors.append(f"Duplicate source1_entity_id '{s1_id}' at line {line_num}.")
            seen_s1.add(s1_id)

            parsed_targets = parse_id_list(raw_targets)

            # Check for duplicates in the raw string before parsing
            raw_tokens = [t.strip() for t in raw_targets.split(",") if t.strip()]
            if len(raw_tokens) != len(set(raw_tokens)):
                duplicate_targets_count += 1

            for tid in parsed_targets:
                if not tid.startswith(VALID_TARGET_PREFIXES):
                    invalid_prefix_count += 1
                if valid_target_ids and tid not in valid_target_ids:
                    invalid_target_id_count += 1

            predictions_map[s1_id] = set(parsed_targets)

    # Check S1 coverage
    missing_s1 = expected_s1_ids - seen_s1
    if missing_s1:
        errors.append(
            f"Missing {len(missing_s1):,} expected Source 1 IDs in submission. "
            f"Examples: {list(missing_s1)[:5]}"
        )

    unexpected_s1 = seen_s1 - expected_s1_ids
    if unexpected_s1:
        errors.append(
            f"Found {len(unexpected_s1):,} unexpected Source 1 IDs. "
            f"Examples: {list(unexpected_s1)[:5]}"
        )

    if duplicate_targets_count > 0:
        errors.append(f"Found {duplicate_targets_count:,} rows with duplicate target IDs in their list.")

    if invalid_prefix_count > 0:
        errors.append(f"Found {invalid_prefix_count:,} predicted target IDs with invalid prefixes (must start with S2- or S3-).")

    if invalid_target_id_count > 0:
        errors.append(f"Found {invalid_target_id_count:,} predicted target IDs not present in test source datasets.")

    # Check candidate containment constraint if candidate_pairs_path is provided
    if candidate_pairs_path and candidate_pairs_path.exists() and not is_candidate_file:
        containment_violations = 0
        candidate_map: Dict[str, Set[str]] = {}
        with open(candidate_pairs_path, "r", encoding="utf-8") as f_cand:
            f_cand.readline()  # skip header
            for line in f_cand:
                parts = line.rstrip("\r\n").split("\t")
                cid_s1 = parts[0].strip()
                cid_targets = set(parse_id_list(parts[1])) if len(parts) > 1 else set()
                candidate_map[cid_s1] = cid_targets

        for s1_id, predicted_set in predictions_map.items():
            allowed_cands = candidate_map.get(s1_id, set())
            unallowed = predicted_set - allowed_cands
            if unallowed:
                containment_violations += len(unallowed)

        if containment_violations > 0:
            errors.append(
                f"Candidate containment violation: {containment_violations:,} predicted IDs "
                f"were not present in '{candidate_pairs_path.name}' for their respective S1 entity."
            )

    is_valid = len(errors) == 0
    return is_valid, errors


def validate_submission_streaming(
    submission_path: Path,
    s1_source_path: Path,
    target_source_paths: Optional[List[Path]] = None,
    candidate_pairs_path: Optional[Path] = None,
    is_candidate_file: bool = False,
    sample_s1: Optional[int] = None,
    chunksize: int = 50000,
) -> Tuple[bool, List[str]]:
    """Stream-validate submission or candidate pairs against source TSVs without holding all data in RAM.

    Args:
        submission_path: Path to candidate_pairs.tsv or matching_results.tsv.
        s1_source_path: Path to test_source1.tsv.
        target_source_paths: Optional list of paths to test_source2.tsv and test_source3.tsv.
        candidate_pairs_path: Optional candidate pairs path to verify containment for predictions.
        is_candidate_file: True if validating candidate_pairs.tsv.
        sample_s1: Optional limit if running a sampled test run.
        chunksize: Streaming chunk size.

    Returns:
        Tuple[bool, List[str]]: (is_valid, list of error messages).
    """
    errors: List[str] = []
    submission_path = Path(submission_path)
    s1_source_path = Path(s1_source_path)

    if not submission_path.exists():
        return False, [f"Submission file not found: {submission_path.resolve()}"]
    if not s1_source_path.exists():
        return False, [f"Source 1 file not found: {s1_source_path.resolve()}"]

    expected_second_col = COL_CANDIDATE_IDS if is_candidate_file else COL_MATCHED_IDS
    expected_header = f"{COL_SOURCE1_ID}\t{expected_second_col}"

    # 1. Collect expected S1 IDs in a compact set
    expected_s1_ids: Set[str] = set()
    with open(s1_source_path, "r", encoding="utf-8") as f_s1:
        header = f_s1.readline().rstrip("\r\n").split("\t")
        id_idx = header.index(COL_SOURCE1_ID) if COL_SOURCE1_ID in header else (header.index("entity_id") if "entity_id" in header else 0)
        for line in f_s1:
            line_str = line.rstrip("\r\n")
            if line_str:
                expected_s1_ids.add(line_str.split("\t")[id_idx].strip())
                if sample_s1 is not None and len(expected_s1_ids) >= sample_s1:
                    break

    # 2. Build target ID set if target sources provided and small enough
    valid_target_ids: Optional[Set[str]] = None
    if target_source_paths:
        valid_target_ids = set()
        for tp in target_source_paths:
            tp_path = Path(tp)
            if tp_path.exists():
                with open(tp_path, "r", encoding="utf-8") as f_t:
                    t_header = f_t.readline().rstrip("\r\n").split("\t")
                    t_id_idx = t_header.index("entity_id") if "entity_id" in t_header else 0
                    for t_line in f_t:
                        t_str = t_line.rstrip("\r\n")
                        if t_str:
                            valid_target_ids.add(t_str.split("\t")[t_id_idx].strip())

    return validate_submission_file(
        submission_path=submission_path,
        expected_s1_ids=expected_s1_ids,
        candidate_pairs_path=candidate_pairs_path,
        valid_target_ids=valid_target_ids,
        is_candidate_file=is_candidate_file,
    )


def main() -> None:
    """CLI for submission validation."""
    parser = argparse.ArgumentParser(description="Validate ER submission and candidate files.")
    parser.add_argument("submission_file", type=str, help="Path to matching_results.tsv or candidate_pairs.tsv")
    parser.add_argument("--test-s1", type=str, required=True, help="Path to test_source1.tsv to verify S1 coverage")
    parser.add_argument("--candidates-file", type=str, default=None, help="Optional path to candidate_pairs.tsv to verify containment")
    parser.add_argument("--is-candidates", action="store_true", help="Set if validating candidate_pairs.tsv instead of matching_results.tsv")

    args = parser.parse_args()

    sub_path = Path(args.submission_file)
    s1_path = Path(args.test_s1)
    cand_path = Path(args.candidates_file) if args.candidates_file else None

    # Load expected S1 IDs
    with open(s1_path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\r\n").split("\t")
        id_col_idx = header.index(COL_SOURCE1_ID) if COL_SOURCE1_ID in header else (header.index(COL_ENTITY_ID) if COL_ENTITY_ID in header else 0)
        expected_s1 = {line.rstrip("\r\n").split("\t")[id_col_idx].strip() for line in f if line.strip()}

    print(f"Validating '{sub_path.name}' against {len(expected_s1):,} expected test S1 entities...")
    is_valid, errors = validate_submission_file(
        submission_path=sub_path,
        expected_s1_ids=expected_s1,
        candidate_pairs_path=cand_path,
        is_candidate_file=args.is_candidates,
    )

    if is_valid:
        print(f"[SUCCESS] '{sub_path.name}' is 100% VALID and meets all competition invariants.")
        sys.exit(0)
    else:
        print(f"[FAILED] Validation errors found in '{sub_path.name}':")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
