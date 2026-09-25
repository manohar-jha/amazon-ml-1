"""Data schemas and type contracts for the Entity Resolution pipeline.

Defines schemas and column invariants for raw sources, normalized data,
ground truth, candidate pairs, feature/score tables, and final submissions.
"""

from dataclasses import dataclass
from typing import List, Optional, Set


# Standard column name constants
COL_ENTITY_ID = "entity_id"
COL_BUSINESS_NAME = "business_name"
COL_BUSINESS_ADDRESS = "business_address"
COL_COUNTRY = "country"

COL_BUSINESS_NAME_NORM = "business_name_norm"
COL_BUSINESS_ADDRESS_NORM = "business_address_norm"
COL_COUNTRY_NORM = "country_norm"

COL_SOURCE1_ID = "source1_entity_id"
COL_MATCHED_IDS = "matched_entity_ids"
COL_CANDIDATE_IDS = "candidate_entity_ids"
COL_CANDIDATE_ID = "candidate_entity_id"
COL_SCORE = "match_score"
COL_LABEL = "is_match"

# Expected column schemas
SOURCE_COLUMNS: List[str] = [
    COL_ENTITY_ID,
    COL_BUSINESS_NAME,
    COL_BUSINESS_ADDRESS,
    COL_COUNTRY,
]

NORMALIZED_SOURCE_COLUMNS: List[str] = [
    COL_ENTITY_ID,
    COL_BUSINESS_NAME,
    COL_BUSINESS_ADDRESS,
    COL_COUNTRY,
    COL_BUSINESS_NAME_NORM,
    COL_BUSINESS_ADDRESS_NORM,
    COL_COUNTRY_NORM,
]

GROUND_TRUTH_COLUMNS: List[str] = [
    COL_SOURCE1_ID,
    COL_MATCHED_IDS,
]

CANDIDATE_PAIRS_COLUMNS: List[str] = [
    COL_SOURCE1_ID,
    COL_CANDIDATE_IDS,
]

PAIR_SCORE_COLUMNS: List[str] = [
    COL_SOURCE1_ID,
    COL_CANDIDATE_ID,
    COL_SCORE,
]

SUBMISSION_COLUMNS: List[str] = [
    COL_SOURCE1_ID,
    COL_MATCHED_IDS,
]

# ID prefix constants
PREFIX_SOURCE1 = "S1-"
PREFIX_SOURCE2 = "S2-"
PREFIX_SOURCE3 = "S3-"
VALID_TARGET_PREFIXES = (PREFIX_SOURCE2, PREFIX_SOURCE3)


@dataclass(frozen=True)
class SourceRecord:
    """Individual entity record from Source 1, 2, or 3."""
    entity_id: str
    business_name: str
    business_address: str
    country: str


@dataclass(frozen=True)
class CandidatePair:
    """Candidate match list for an S1 entity."""
    source1_entity_id: str
    candidate_entity_ids: List[str]


@dataclass(frozen=True)
class SubmissionPrediction:
    """Predicted matches for an S1 entity."""
    source1_entity_id: str
    matched_entity_ids: List[str]
