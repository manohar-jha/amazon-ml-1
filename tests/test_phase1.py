"""Unit tests for Phase 1 Candidate Generation: MinHash LSH, bounded fuzzy, soft country, provenance."""

import tempfile
import unittest
from pathlib import Path
import pandas as pd

from src.blocking import (
    MultiSourceCandidateIndex,
    evaluate_validation_recall,
    generate_candidate_union,
    generate_candidates_with_provenance,
    write_candidate_outputs,
)
from src.lsh import MinHashLSH, extract_character_ngrams
from src.normalize import normalize_dataframe
from utils.validate_submission import validate_submission_file


class TestPhase1CandidateGeneration(unittest.TestCase):
    """Test suite for Phase 1 candidate generation components with synthetic fixtures."""

    def test_character_ngrams_extraction(self):
        """Verify character 3-gram extraction with boundary padding."""
        ngrams = extract_character_ngrams("orelee", n=3)
        self.assertIn("^or", ngrams)
        self.assertIn("ele", ngrams)
        self.assertIn("ee$", ngrams)

    def test_minhash_lsh_approximate_matching(self):
        """Verify MinHash LSH retrieves candidates with typos/approximate names in sublinear time."""
        lsh = MinHashLSH(num_permutations=32, num_bands=8, shingle_n=3)

        # Index target records
        lsh.index_entity("S2-100", "orelee barbershop", country="us")
        lsh.index_entity("S2-200", "acme manufacturing", country="us")
        lsh.index_entity("S3-300", "bistro de paris", country="france")

        # Query with typo: "orele barbershop" (missing 'e')
        cands = lsh.query_candidates("orele barbershop", country="us")
        self.assertIn("S2-100", cands)
        self.assertNotIn("S2-200", cands)

    def test_soft_country_preference_and_unseen_countries(self):
        """Verify soft country preference searches across countries if country is unseen or matches are sparse."""
        lsh = MinHashLSH(num_permutations=32, num_bands=8, shingle_n=3)
        lsh.index_entity("S2-FR1", "bistro de paris", country="france")

        # Query with unseen country 'germany' -> soft country fallback should retrieve S2-FR1
        cands = lsh.query_candidates("bistro de paris", country="germany", soft_country_fallback=True)
        self.assertIn("S2-FR1", cands)

    def test_posting_caps_and_truncation_tracking(self):
        """Verify that high-frequency postings are capped and candidate limits log truncation."""
        index = MultiSourceCandidateIndex(country="us")

        # Create synthetic data with 10 duplicate target records sharing 'general store'
        s2_df = pd.DataFrame([
            {
                "entity_id": f"S2-{i}",
                "business_name_norm": "general store",
                "business_address_norm": "100 main st",
                "country_norm": "us",
            }
            for i in range(10)
        ])
        s3_df = pd.DataFrame([
            {
                "entity_id": f"S3-{i}",
                "business_name_norm": "general store",
                "business_address_norm": "100 main st",
                "country_norm": "us",
            }
            for i in range(10)
        ])

        index.build_from_sources(s2_df, s3_df, max_posting_len=5)
        # Posting length should be capped at 5
        self.assertLessEqual(len(index.exact_name_idx["general store"]), 5)

        # Test truncation flag when max_candidates is set low (e.g. 3)
        cands, prov, truncated = generate_candidates_with_provenance(
            name="general store",
            addr="100 main st",
            country="us",
            index=index,
            max_candidates=3,
        )
        self.assertTrue(truncated)
        self.assertEqual(len(cands), 3)

    def test_pair_level_provenance_and_schema_validation(self):
        """Verify that pair-level provenance table and candidate pairs are generated and valid."""
        s1_df = pd.DataFrame([
            {"entity_id": "S1-1", "business_name_norm": "orelee barbershop", "business_address_norm": "1795 westchester dr", "country_norm": "us"},
            {"entity_id": "S1-2", "business_name_norm": "bistro de paris", "business_address_norm": "12 rue de paris", "country_norm": "france"},
            {"entity_id": "S1-3", "business_name_norm": "unknown store", "business_address_norm": "no addr", "country_norm": "us"},
        ])
        s2_df = pd.DataFrame([
            {"entity_id": "S2-10", "business_name_norm": "orelee barbershop llc", "business_address_norm": "1795 westchester drive", "country_norm": "us"},
            {"entity_id": "S2-20", "business_name_norm": "bistro de paris", "business_address_norm": "12 rue de paris", "country_norm": "france"},
        ])
        s3_df = pd.DataFrame([
            {"entity_id": "S3-10", "business_name_norm": "orelee barbershop", "business_address_norm": "1795 westchester dr", "country_norm": "us"},
        ])

        cands_map, prov_records, stats = generate_candidate_union(
            s1_df=s1_df,
            s2_df=s2_df,
            s3_df=s3_df,
            max_candidates=50,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            pairs_path = Path(tmp_dir) / "candidate_pairs.tsv"
            prov_path = Path(tmp_dir) / "candidate_provenance.tsv"

            write_candidate_outputs(
                candidates_map=cands_map,
                provenance_records=prov_records,
                ordered_s1_ids=s1_df["entity_id"].tolist(),
                candidate_pairs_path=pairs_path,
                provenance_path=prov_path,
            )

            # Check candidate_pairs.tsv validation
            all_target_ids = {"S2-10", "S2-20", "S3-10"}
            is_valid, errors = validate_submission_file(
                submission_path=pairs_path,
                expected_s1_ids=set(s1_df["entity_id"]),
                valid_target_ids=all_target_ids,
                is_candidate_file=True,
            )
            self.assertTrue(is_valid)
            self.assertEqual(len(errors), 0)

            # Check provenance table schema
            prov_df = pd.read_csv(prov_path, sep="\t", dtype=str)
            expected_prov_cols = ["source1_entity_id", "candidate_entity_id", "source_dataset", "provenance_rules"]
            self.assertEqual(list(prov_df.columns), expected_prov_cols)
            self.assertGreater(len(prov_df), 0)

    def test_validation_recall_evaluation(self):
        """Verify validation recall computation on saved validation split IDs."""
        cands_map = {
            "S1-1": {"S2-10", "S3-10"},
            "S1-2": {"S2-20"},
            "S1-3": set(),
        }
        val_s1_ids = {"S1-1", "S1-2", "S1-3"}
        gt_df = pd.DataFrame([
            {"source1_entity_id": "S1-1", "matched_entity_ids": "S2-10,S3-10"},
            {"source1_entity_id": "S1-2", "matched_entity_ids": "S2-20,S3-20"},
            {"source1_entity_id": "S1-3", "matched_entity_ids": ""},
        ])

        metrics = evaluate_validation_recall(cands_map, val_s1_ids, gt_df)
        self.assertEqual(metrics["val_total_gt_pairs"], 4)
        self.assertEqual(metrics["val_found_gt_pairs"], 3)
        self.assertAlmostEqual(metrics["val_pair_recall"], 3 / 4)
        self.assertAlmostEqual(metrics["val_full_s1_coverage"], 1 / 2)


if __name__ == "__main__":
    unittest.main()
