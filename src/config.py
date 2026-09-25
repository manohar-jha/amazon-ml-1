"""Central configuration module for the Amazon Entity Resolution Challenge.

Defines dynamic paths, schema definitions, and configurable hyperparameters for
normalization, MinHash LSH, inverted indexing, fuzzy fallback, and candidate generation.
"""

import os
from pathlib import Path

# Project root dynamically resolved based on this file's location
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# Core directories
DATA_DIR: Path = PROJECT_ROOT / "data"
TRAIN_DIR: Path = DATA_DIR / "train"
TEST_DIR: Path = DATA_DIR / "test"
OUTPUT_DIR: Path = PROJECT_ROOT / "output"
MODELS_DIR: Path = PROJECT_ROOT / "models"
NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"
SPLITS_DIR: Path = OUTPUT_DIR / "splits"

# Training file paths
TRAIN_SOURCE1_PATH: Path = TRAIN_DIR / "train_source1.tsv"
TRAIN_SOURCE2_PATH: Path = TRAIN_DIR / "train_source2.tsv"
TRAIN_SOURCE3_PATH: Path = TRAIN_DIR / "train_source3.tsv"
TRAIN_GROUND_TRUTH_PATH: Path = TRAIN_DIR / "train_ground_truth.tsv"

# Test file paths
TEST_SOURCE1_PATH: Path = TEST_DIR / "test_source1.tsv"
TEST_SOURCE2_PATH: Path = TEST_DIR / "test_source2.tsv"
TEST_SOURCE3_PATH: Path = TEST_DIR / "test_source3.tsv"

# Output candidate files
TRAIN_CANDIDATES_PATH: Path = OUTPUT_DIR / "train_candidate_pairs.tsv"
TEST_CANDIDATES_PATH: Path = OUTPUT_DIR / "candidate_pairs.tsv"
TEST_PROVENANCE_PATH: Path = OUTPUT_DIR / "candidate_provenance.tsv"

# Schema definitions
SOURCE_EXPECTED_COLUMNS: list[str] = [
    "entity_id",
    "business_name",
    "business_address",
    "country",
]

GROUND_TRUTH_EXPECTED_COLUMNS: list[str] = [
    "source1_entity_id",
    "matched_entity_ids",
]

# ID prefix conventions
EXPECTED_ID_PREFIXES: dict[str, str] = {
    "source1": "S1-",
    "source2": "S2-",
    "source3": "S3-",
}

# ==============================================================================
# Phase 1: Candidate Generation & Blocking Hyperparameters
# ==============================================================================

# 1. Inverted Token Index Hyperparameters
BLOCKING_MAX_TOKEN_DOC_FREQ: int = int(os.environ.get("BLOCKING_MAX_TOKEN_DOC_FREQ", "100"))
BLOCKING_MAX_POSTING_LEN: int = int(os.environ.get("BLOCKING_MAX_POSTING_LEN", "250"))
BLOCKING_MAX_RARE_TOKENS_PER_S1: int = int(os.environ.get("BLOCKING_MAX_RARE_TOKENS_PER_S1", "3"))
BLOCKING_MIN_TOKEN_LEN: int = int(os.environ.get("BLOCKING_MIN_TOKEN_LEN", "3"))
BLOCKING_MAX_PAIR_TOKEN_FREQ: int = int(os.environ.get("BLOCKING_MAX_PAIR_TOKEN_FREQ", "1000"))

# 2. MinHash LSH Hyperparameters
LSH_NUM_PERMUTATIONS: int = int(os.environ.get("LSH_NUM_PERMUTATIONS", "32"))
LSH_NUM_BANDS: int = int(os.environ.get("LSH_NUM_BANDS", "8"))
LSH_SHINGLE_N: int = int(os.environ.get("LSH_SHINGLE_N", "3"))
LSH_MAX_BUCKET_SIZE: int = int(os.environ.get("LSH_MAX_BUCKET_SIZE", "100"))
LSH_MAX_CANDIDATES: int = int(os.environ.get("LSH_MAX_CANDIDATES", "50"))

# 3. Address Blocking Hyperparameters
BLOCKING_MAX_ADDR_KEY_FREQ: int = int(os.environ.get("BLOCKING_MAX_ADDR_KEY_FREQ", "50"))

# 4. Bounded Fuzzy Fallback & Safety Net
BLOCKING_FUZZY_MIN_CANDS: int = int(os.environ.get("BLOCKING_FUZZY_MIN_CANDS", "3"))
BLOCKING_FUZZY_TOP_K: int = int(os.environ.get("BLOCKING_FUZZY_TOP_K", "25"))
BLOCKING_SAFETY_NET_MAX_BUCKET: int = int(os.environ.get("BLOCKING_SAFETY_NET_MAX_BUCKET", "30"))

# 5. Overall Per-Entity Candidate Limit
BLOCKING_MAX_CANDIDATES_PER_S1: int = int(os.environ.get("BLOCKING_MAX_CANDIDATES_PER_S1", "250"))
BLOCKING_SOFT_COUNTRY_MODE: bool = os.environ.get("BLOCKING_SOFT_COUNTRY_MODE", "true").lower() == "true"

# Common corporate / legal terms used for core name derivation
COMMON_LEGAL_TERMS: set[str] = {
    "llc", "inc", "ltd", "limited", "pvt", "private", "corp", "corporation",
    "co", "company", "services", "enterprises", "holdings", "group",
    "llp", "pllc", "sarl", "sas", "sa", "gmbh", "bv", "spa", "consulting",
    "technologies", "technology", "international", "associates", "solutions",
    "industries", "ventures", "management", "development", "trading",
}
