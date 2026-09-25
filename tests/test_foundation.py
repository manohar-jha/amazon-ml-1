"""Synthetic unit tests for Phase 0 foundation: schemas, I/O, scoring, splits, and validation."""

import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from src.evaluate import calculate_entity_metrics, evaluate_predictions
from src.io_utils import (
    parse_id_list,
    read_tsv,
    serialize_id_list,
    validate_id_prefix,
    write_id_map_to_tsv,
    write_tsv,
)
from src.schemas import (
    COL_BUSINESS_ADDRESS,
    COL_BUSINESS_NAME,
    COL_COUNTRY,
    COL_ENTITY_ID,
    COL_MATCHED_IDS,
    COL_SOURCE1_ID,
    SOURCE_COLUMNS,
)
from src.split_utils import create_entity_splits, load_split_ids, save_split_ids
from utils.validate_submission import validate_submission_file


class TestFoundationContracts(unittest.TestCase):
    """Test suite verifying Phase 0 contracts with tiny synthetic data."""

    def test_id_list_parsing_and_deterministic_serialization(self):
        """Verify comma list parsing and deterministic sorted serialization."""
        # 1. Parsing comma lists
        self.assertEqual(parse_id_list("S2-10, S3-5, S2-10"), ["S2-10", "S3-5"])
        self.assertEqual(parse_id_list(""), [])
        self.assertEqual(parse_id_list(None), [])
        self.assertEqual(parse_id_list(np.nan), [])

        # 2. Serialization: sorted, unique, empty string for singletons
        self.assertEqual(serialize_id_list(["S3-2", "S2-1", "S3-2"]), "S2-1,S3-2")
        self.assertEqual(serialize_id_list([]), "")
        self.assertEqual(serialize_id_list(None), "")

    def test_raw_address_commas_are_not_split(self):
        """Verify that raw addresses containing commas are preserved completely without splitting."""
        synthetic_addr = "Plot 12, Floor 3, MG Road, Bangalore, Karnataka"
        df_synthetic = pd.DataFrame([{
            COL_ENTITY_ID: "S1-999",
            COL_BUSINESS_NAME: "Alpha, Beta & Gamma LLC",
            COL_BUSINESS_ADDRESS: synthetic_addr,
            COL_COUNTRY: "India",
        }])

        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "test_source.tsv"
            write_tsv(df_synthetic, file_path, expected_columns=SOURCE_COLUMNS)

            # Read back
            loaded_df = read_tsv(file_path, expected_columns=SOURCE_COLUMNS)
            loaded_addr = loaded_df[COL_BUSINESS_ADDRESS].iloc[0]
            self.assertEqual(loaded_addr, synthetic_addr)
            # Ensure name with commas was also preserved intact
            self.assertEqual(loaded_df[COL_BUSINESS_NAME].iloc[0], "Alpha, Beta & Gamma LLC")

    def test_open_country_handling(self):
        """Verify open country handling supports unseen countries (France, Germany, Japan)."""
        countries = ["US", "India", "France", "Germany", "Japan", "Brazil"]
        df_countries = pd.DataFrame([
            {COL_ENTITY_ID: f"S1-{i}", COL_BUSINESS_NAME: f"Biz {c}", COL_BUSINESS_ADDRESS: "123 St", COL_COUNTRY: c}
            for i, c in enumerate(countries)
        ])

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "countries.tsv"
            write_tsv(df_countries, path, expected_columns=SOURCE_COLUMNS)
            loaded = read_tsv(path, expected_columns=SOURCE_COLUMNS)
            self.assertEqual(set(loaded[COL_COUNTRY]), set(countries))

    def test_entity_macro_f05_calculation(self):
        """Verify Entity-Macro F0.5 scoring with exact precision weighting and singleton handling."""
        # Case 1: True singleton correctly predicted as empty -> F0.5 = 1.0
        p, r, f = calculate_entity_metrics(set(), set(), beta=0.5)
        self.assertEqual((p, r, f), (1.0, 1.0, 1.0))

        # Case 2: True singleton incorrectly predicted with candidate -> F0.5 = 0.0
        p, r, f = calculate_entity_metrics(set(), {"S2-1"}, beta=0.5)
        self.assertEqual((p, r, f), (0.0, 1.0, 0.0))

        # Case 3: True match missed (predicted empty) -> F0.5 = 0.0
        p, r, f = calculate_entity_metrics({"S2-1"}, set(), beta=0.5)
        self.assertEqual((p, r, f), (1.0, 0.0, 0.0))

        # Case 4: 1 TP, 1 FP (2 predicted, 1 true) -> Precision = 0.5, Recall = 1.0
        # F0.5 = (1.25 * 0.5 * 1.0) / (0.25 * 0.5 + 1.0) = 0.625 / 1.125 = 0.5555...
        p, r, f = calculate_entity_metrics({"S2-1"}, {"S2-1", "S3-2"}, beta=0.5)
        self.assertAlmostEqual(p, 0.5)
        self.assertAlmostEqual(r, 1.0)
        self.assertAlmostEqual(f, 0.625 / 1.125)

        # Batch evaluation
        gt_df = pd.DataFrame([
            {COL_SOURCE1_ID: "S1-1", COL_MATCHED_IDS: "S2-1,S3-1"},
            {COL_SOURCE1_ID: "S1-2", COL_MATCHED_IDS: ""},  # true singleton
        ])
        pred_df = pd.DataFrame([
            {COL_SOURCE1_ID: "S1-1", COL_MATCHED_IDS: "S2-1,S3-1"},  # 100% correct
            {COL_SOURCE1_ID: "S1-2", COL_MATCHED_IDS: ""},          # 100% correct
        ])
        metrics = evaluate_predictions(gt_df, pred_df, beta=0.5)
        self.assertEqual(metrics["entity_macro_f05"], 1.0)
        self.assertEqual(metrics["singleton_accuracy"], 1.0)

    def test_split_generation_and_zero_leakage(self):
        """Verify deterministic entity-level splitting with zero overlap between partitions."""
        synthetic_s1_ids = [f"S1-{i:05d}" for i in range(1000)]

        splits1 = create_entity_splits(synthetic_s1_ids, train_ratio=0.7, calib_ratio=0.15, val_ratio=0.15, seed=42)
        splits2 = create_entity_splits(synthetic_s1_ids, train_ratio=0.7, calib_ratio=0.15, val_ratio=0.15, seed=42)

        # 1. Determinism
        self.assertEqual(splits1["train"], splits2["train"])
        self.assertEqual(splits1["calibration"], splits2["calibration"])
        self.assertEqual(splits1["validation"], splits2["validation"])

        # 2. Mutually exclusive sets (Zero entity leakage)
        train_set = set(splits1["train"])
        calib_set = set(splits1["calibration"])
        val_set = set(splits1["validation"])

        self.assertEqual(len(train_set.intersection(calib_set)), 0)
        self.assertEqual(len(train_set.intersection(val_set)), 0)
        self.assertEqual(len(calib_set.intersection(val_set)), 0)
        self.assertEqual(train_set.union(calib_set).union(val_set), set(synthetic_s1_ids))

        # 3. Persistence and loading
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_split_ids(splits1, output_dir=tmp_dir)
            loaded_splits = load_split_ids(splits_dir=tmp_dir)
            self.assertEqual(loaded_splits["train"], train_set)
            self.assertEqual(loaded_splits["calibration"], calib_set)
            self.assertEqual(loaded_splits["validation"], val_set)

    def test_submission_invariants_and_candidate_containment(self):
        """Verify strict submission validation rules and candidate containment."""
        expected_s1 = {"S1-1", "S1-2", "S1-3"}
        valid_targets = {"S2-10", "S2-20", "S3-30", "S3-40"}

        with tempfile.TemporaryDirectory() as tmp_dir:
            cand_path = Path(tmp_dir) / "candidate_pairs.tsv"
            sub_path = Path(tmp_dir) / "matching_results.tsv"

            # 1. Write valid candidate pairs
            cand_map = {
                "S1-1": ["S2-10", "S3-30"],
                "S1-2": ["S2-20"],
                "S1-3": [],  # singleton
            }
            write_id_map_to_tsv(cand_map, ["S1-1", "S1-2", "S1-3"], cand_path, id_column_name="candidate_entity_ids")

            # 2. Write valid matching results (contained in candidates)
            pred_map = {
                "S1-1": ["S2-10"],
                "S1-2": ["S2-20"],
                "S1-3": [],
            }
            write_id_map_to_tsv(pred_map, ["S1-1", "S1-2", "S1-3"], sub_path, id_column_name="matched_entity_ids")

            is_valid, errors = validate_submission_file(
                submission_path=sub_path,
                expected_s1_ids=expected_s1,
                candidate_pairs_path=cand_path,
                valid_target_ids=valid_targets,
            )
            self.assertTrue(is_valid)
            self.assertEqual(len(errors), 0)

            # 3. Test violation: predicting ID not in candidate list
            invalid_pred_map = {
                "S1-1": ["S2-10", "S3-40"],  # S3-40 is not in S1-1 candidates!
                "S1-2": ["S2-20"],
                "S1-3": [],
            }
            invalid_sub_path = Path(tmp_dir) / "invalid_matching_results.tsv"
            write_id_map_to_tsv(invalid_pred_map, ["S1-1", "S1-2", "S1-3"], invalid_sub_path, id_column_name="matched_entity_ids")

            is_valid, errors = validate_submission_file(
                submission_path=invalid_sub_path,
                expected_s1_ids=expected_s1,
                candidate_pairs_path=cand_path,
                valid_target_ids=valid_targets,
            )
            self.assertFalse(is_valid)
            self.assertTrue(any("Candidate containment violation" in e for e in errors))


if __name__ == "__main__":
    unittest.main()
