"""Unit tests for Phase 3 blocking and candidate generation module."""

import tempfile
import unittest
from pathlib import Path
import pandas as pd

from src.blocking import (
    CountryBlockingIndex,
    evaluate_blocking_recall,
    extract_address_keys,
    generate_candidates,
    generate_candidates_for_record,
    get_compact_name,
    get_core_name,
    stream_generate_and_save_candidates,
    validate_candidate_file,
)


class TestBlocking(unittest.TestCase):
    """Unit test suite for blocking functions and candidate generation."""

    def test_get_core_and_compact_name(self):
        """Verify core and compact name extraction."""
        self.assertEqual(get_core_name("orelee s barbershop llc"), "orelee s barbershop")
        self.assertEqual(get_core_name("prime money inc"), "prime money")
        self.assertEqual(get_compact_name("prime money inc"), "primemoney")
        self.assertEqual(get_core_name("ram marketing pvt ltd"), "ram marketing")
        self.assertEqual(get_core_name(""), "")

    def test_extract_address_keys(self):
        """Verify address key extraction handles numbers and words."""
        keys = extract_address_keys("1795 westchester drive high point nc")
        self.assertTrue(any("1795" in k for k in keys))
        self.assertTrue(any("westchester" in k for k in keys))
        # Leading zero stripping
        keys_zero = extract_address_keys("001795 westchester dr")
        self.assertTrue(any("1795" in k for k in keys_zero))
        # Missing/empty address returns empty list
        self.assertEqual(extract_address_keys(""), [])
        self.assertEqual(extract_address_keys(None), [])

    def test_country_partitioning_and_s2_s3_retrieval(self):
        """Verify that candidates are partitioned by country and return both S2 and S3 IDs."""
        s2_df = pd.DataFrame({
            "entity_id": ["S2-1", "S2-2", "S2-3"],
            "business_name_norm": ["orelee s barbershop", "acme corp", "paris bistro sarl"],
            "business_address_norm": ["1795 westchester dr", "100 main st", "12 rue de paris"],
            "country_norm": ["us", "us", "france"],
        })
        s3_df = pd.DataFrame({
            "entity_id": ["S3-1", "S3-2", "S3-3"],
            "business_name_norm": ["orelee s barbershop llc", "beta tech", "paris bistro"],
            "business_address_norm": ["1795 westchester drive", "200 second ave", "12 rue de paris"],
            "country_norm": ["us", "us", "france"],
        })
        s1_df = pd.DataFrame({
            "entity_id": ["S1-1", "S1-2"],
            "business_name_norm": ["orelee s barbershop", "paris bistro"],
            "business_address_norm": ["1795 westchester drive", "12 rue de paris"],
            "country_norm": ["us", "france"],
        })

        candidates_map = generate_candidates(s1_df, s2_df, s3_df)

        # S1-1 (US) should match S2-1 and S3-1 (both US) and NOT any French entities
        self.assertIn("S1-1", candidates_map)
        s1_1_cands = candidates_map["S1-1"]
        self.assertIn("S2-1", s1_1_cands)
        self.assertIn("S3-1", s1_1_cands)
        self.assertNotIn("S2-3", s1_1_cands)
        self.assertNotIn("S3-3", s1_1_cands)

        # S1-2 (France) should match French entities S2-3 and S3-3
        self.assertIn("S1-2", candidates_map)
        s1_2_cands = candidates_map["S1-2"]
        self.assertIn("S2-3", s1_2_cands)
        self.assertIn("S3-3", s1_2_cands)
        self.assertNotIn("S2-1", s1_2_cands)

    def test_unseen_country_support(self):
        """Verify that arbitrary/unseen countries (e.g. Germany) work automatically without hardcoded lists."""
        s2_df = pd.DataFrame({
            "entity_id": ["S2-DE1"],
            "business_name_norm": ["berlin auto gmbh"],
            "business_address_norm": ["alexanderplatz 1"],
            "country_norm": ["germany"],
        })
        s3_df = pd.DataFrame({
            "entity_id": ["S3-DE1"],
            "business_name_norm": ["berlin auto"],
            "business_address_norm": ["alexanderplatz 1 berlin"],
            "country_norm": ["germany"],
        })
        s1_df = pd.DataFrame({
            "entity_id": ["S1-DE1"],
            "business_name_norm": ["berlin auto"],
            "business_address_norm": ["alexanderplatz 1"],
            "country_norm": ["germany"],
        })

        candidates_map = generate_candidates(s1_df, s2_df, s3_df)
        self.assertIn("S1-DE1", candidates_map)
        self.assertEqual(candidates_map["S1-DE1"], {"S2-DE1", "S3-DE1"})

    def test_duplicate_removal_and_empty_handling(self):
        """Verify that duplicate candidate IDs are not returned and empty records are handled safely."""
        s2_df = pd.DataFrame({
            "entity_id": ["S2-1", "S2-2"],
            "business_name_norm": ["alpha solutions", ""],
            "business_address_norm": ["100 main st", ""],
            "country_norm": ["us", "us"],
        })
        s3_df = pd.DataFrame({
            "entity_id": ["S3-1"],
            "business_name_norm": ["alpha solutions"],
            "business_address_norm": ["100 main street"],
            "country_norm": ["us"],
        })
        s1_df = pd.DataFrame({
            "entity_id": ["S1-1", "S1-2"],
            "business_name_norm": ["alpha solutions", ""],
            "business_address_norm": ["100 main st", ""],
            "country_norm": ["us", "us"],
        })

        cands = generate_candidates(s1_df, s2_df, s3_df)
        self.assertEqual(len(cands["S1-1"]), len(set(cands["S1-1"])))
        self.assertEqual(len(cands["S1-2"]), 0)

    def test_evaluate_blocking_recall(self):
        """Verify ground truth recall evaluation metrics."""
        candidates_map = {
            "S1-1": {"S2-1", "S3-1", "S2-99"},
            "S1-2": {"S2-2"},
            "S1-3": set(),
            "S1-4": {"S2-5"},  # Empty GT record with candidate
        }
        gt_df = pd.DataFrame({
            "source1_entity_id": ["S1-1", "S1-2", "S1-3", "S1-4"],
            "matched_entity_ids": ["S2-1,S3-1", "S2-2,S3-2", "S2-3", ""],
        })

        metrics = evaluate_blocking_recall(candidates_map, gt_df)

        # Total GT pairs: (S1-1: 2, S1-2: 2, S1-3: 1) = 5 pairs
        self.assertEqual(metrics["total_gt_pairs"], 5)
        # Found: S2-1, S3-1 (from S1-1) + S2-2 (from S1-2) = 3 pairs
        self.assertEqual(metrics["found_gt_pairs"], 3)
        self.assertAlmostEqual(metrics["overall_pair_recall"], 3 / 5)

        # Full coverage: S1-1 has 2/2 found = full. S1-2 has 1/2 = partial. S1-3 has 0/1 = none.
        self.assertAlmostEqual(metrics["full_coverage_active_s1"], 1 / 3)
        self.assertEqual(metrics["empty_gt_total"], 1)
        self.assertEqual(metrics["empty_gt_with_cands"], 1)

    def test_stream_and_validate_candidate_file(self):
        """Verify streaming candidate generation and TSV output validation."""
        s2_df = pd.DataFrame({
            "entity_id": ["S2-10", "S2-20"],
            "business_name_norm": ["zeta energy", "theta solar"],
            "business_address_norm": ["500 pine rd", "600 oak ave"],
            "country_norm": ["us", "us"],
        })
        s3_df = pd.DataFrame({
            "entity_id": ["S3-10", "S3-20"],
            "business_name_norm": ["zeta energy llc", "theta solar inc"],
            "business_address_norm": ["500 pine road", "600 oak ave"],
            "country_norm": ["us", "us"],
        })
        s1_df = pd.DataFrame({
            "entity_id": ["S1-10", "S1-20"],
            "business_name_norm": ["zeta energy", "theta solar"],
            "business_address_norm": ["500 pine rd", "600 oak ave"],
            "country_norm": ["us", "us"],
        })

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "candidate_pairs.tsv"
            stream_generate_and_save_candidates(
                s1_df=s1_df,
                s2_df=s2_df,
                s3_df=s3_df,
                output_path=out_file,
            )

            valid_target_ids = {"S2-10", "S2-20", "S3-10", "S3-20"}
            report = validate_candidate_file(
                candidate_file_path=out_file,
                expected_s1_ids={"S1-10", "S1-20"},
                valid_target_ids=valid_target_ids,
                is_test=True,
            )

            self.assertEqual(report["total_rows"], 2)
            self.assertEqual(report["unique_s1_ids"], 2)
            self.assertEqual(report["missing_s1_count"], 0)
            self.assertEqual(report["invalid_target_id_count"], 0)
            self.assertEqual(report["rows_with_duplicate_cands"], 0)
            self.assertEqual(report["train_id_leaks"], 0)


if __name__ == "__main__":
    unittest.main()
