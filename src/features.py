"""Feature engineering module stub for candidate pair scoring.

Phase 0 Interface: Defines feature contracts, input/output schemas, and CLI stubs.
Implementation will occur in Phase 4.
"""

import argparse
import sys
from pathlib import Path
from typing import List


FEATURE_NAMES: List[str] = [
    # String similarity features
    "name_exact_match",
    "name_token_jaccard",
    "name_levenshtein_ratio",
    "name_jw_distance",
    "addr_exact_match",
    "addr_token_jaccard",
    "addr_number_match",
    # Length & character ratios
    "name_len_diff",
    "addr_len_diff",
    # TF-IDF / Embedding cosine similarities
    "name_tfidf_cosine",
    "addr_tfidf_cosine",
]


def main() -> None:
    """CLI stub for feature extraction."""
    parser = argparse.ArgumentParser(description="Extract pairwise features for candidate entity pairs.")
    parser.add_argument("--candidate-file", type=str, required=True, help="Path to candidate_pairs.tsv")
    parser.add_argument("--output-features", type=str, required=True, help="Destination parquet/tsv for features")
    args = parser.parse_args()

    print(f"[Phase 0 Foundation] Feature engineering CLI interface defined. Feature contract: {FEATURE_NAMES}")
    print(f"Candidates file: {args.candidate_file}")
    print(f"Output features: {args.output_features}")


if __name__ == "__main__":
    main()
