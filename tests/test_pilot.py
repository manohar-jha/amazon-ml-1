"""Unit tests for Phase 1 Bounded Pilot execution and deterministic sampling."""

import json
import tempfile
import unittest
from pathlib import Path
import pandas as pd

from kaggle_phase1 import run_pilot
from src.data_loader import deterministic_id_hash, scan_and_sample_source_file


class TestPhase1Pilot(unittest.TestCase):
    """Test suite for Phase 1 pilot mode and bounded data scanning."""

    def setUp(self):
        """Create a temporary directory with synthetic TSV files."""
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.tmp_dir.name)

        self.train_dir = self.base_path / "train"
        self.splits_dir = self.base_path / "splits"
        self.out_dir = self.base_path / "output"

        self.train_dir.mkdir(parents=True, exist_ok=True)
        self.splits_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # Create synthetic Source 1 (30 rows)
        s1_rows = [
            f"S1-{i:03d}\tBusiness Alpha {i}\t{100 + i} Main St\tus"
            for i in range(1, 31)
        ]
        s1_tsv = "entity_id\tbusiness_name\tbusiness_address\tcountry\n" + "\n".join(s1_rows) + "\n"
        with open(self.train_dir / "train_source1.tsv", "w", encoding="utf-8") as f:
            f.write(s1_tsv)

        # Create synthetic Source 2 (50 rows)
        s2_rows = [
            f"S2-{i:03d}\tBusiness Alpha {i} LLC\t{100 + i} Main Street\tus"
            for i in range(1, 51)
        ]
        s2_tsv = "entity_id\tbusiness_name\tbusiness_address\tcountry\n" + "\n".join(s2_rows) + "\n"
        with open(self.train_dir / "train_source2.tsv", "w", encoding="utf-8") as f:
            f.write(s2_tsv)

        # Create synthetic Source 3 (50 rows)
        s3_rows = [
            f"S3-{i:03d}\tBusiness Alpha {i} Corp\t{100 + i} Main St\tus"
            for i in range(1, 51)
        ]
        s3_tsv = "entity_id\tbusiness_name\tbusiness_address\tcountry\n" + "\n".join(s3_rows) + "\n"
        with open(self.train_dir / "train_source3.tsv", "w", encoding="utf-8") as f:
            f.write(s3_tsv)

        # Create synthetic validation split IDs (IDs 5 to 15)
        self.val_ids = {f"S1-{i:03d}" for i in range(5, 16)}
        with open(self.splits_dir / "validation_s1_ids.txt", "w", encoding="utf-8") as f:
            for vid in sorted(self.val_ids):
                f.write(f"{vid}\n")

    def tearDown(self):
        """Clean up temporary directory."""
        self.tmp_dir.cleanup()

    def test_deterministic_id_hash_consistency(self):
        """Verify deterministic hashing is reproducible and seed-dependent."""
        hash_1 = deterministic_id_hash("S1-001", seed=42)
        hash_2 = deterministic_id_hash("S1-001", seed=42)
        hash_diff_seed = deterministic_id_hash("S1-001", seed=99)
        hash_diff_id = deterministic_id_hash("S1-002", seed=42)

        self.assertEqual(hash_1, hash_2)
        self.assertNotEqual(hash_1, hash_diff_seed)
        self.assertNotEqual(hash_1, hash_diff_id)

    def test_scan_and_sample_scan_limit_and_chunksize(self):
        """Verify scan_rows bounds the number of rows read from disk."""
        df, total_scanned = scan_and_sample_source_file(
            path=self.train_dir / "train_source2.tsv",
            scan_rows=20,
            sample_size=10,
            chunksize=5,
            seed=42,
        )
        self.assertEqual(total_scanned, 20)
        self.assertEqual(len(df), 10)
        # All IDs must come from the first 20 rows
        for eid in df["entity_id"]:
            idx = int(eid.split("-")[1])
            self.assertLessEqual(idx, 20)

    def test_scan_and_sample_deterministic_selection(self):
        """Verify sampling is strictly deterministic across runs with identical seed."""
        df1, _ = scan_and_sample_source_file(
            path=self.train_dir / "train_source2.tsv",
            scan_rows=50,
            sample_size=10,
            chunksize=10,
            seed=42,
        )
        df2, _ = scan_and_sample_source_file(
            path=self.train_dir / "train_source2.tsv",
            scan_rows=50,
            sample_size=10,
            chunksize=10,
            seed=42,
        )
        self.assertEqual(list(df1["entity_id"]), list(df2["entity_id"]))

    def test_scan_and_sample_with_validation_filter(self):
        """Verify filtering restricts selected entities to validation split."""
        df, scanned = scan_and_sample_source_file(
            path=self.train_dir / "train_source1.tsv",
            scan_rows=30,
            sample_size=5,
            chunksize=10,
            seed=42,
            filter_ids=self.val_ids,
        )
        self.assertEqual(scanned, 30)
        self.assertEqual(len(df), 5)
        for eid in df["entity_id"]:
            self.assertIn(eid, self.val_ids)

    def test_scan_and_sample_filter_not_found_error(self):
        """Verify ValueError is raised when no validation IDs match in the scan window."""
        unmatched_ids = {"S1-998", "S1-999"}
        with self.assertRaises(ValueError) as ctx:
            scan_and_sample_source_file(
                path=self.train_dir / "train_source1.tsv",
                scan_rows=20,
                sample_size=5,
                chunksize=10,
                seed=42,
                filter_ids=unmatched_ids,
            )
        self.assertIn("No validation S1 IDs were found", str(ctx.exception))
        self.assertIn("--pilot-scan-rows", str(ctx.exception))

    def test_run_pilot_synthetic_end_to_end(self):
        """Verify full run_pilot pipeline creates valid TSVs and summary JSON."""
        run_pilot(
            train_dir=self.train_dir,
            splits_dir=self.splits_dir,
            out_dir=self.out_dir,
            scan_rows=30,
            chunksize=10,
            seed=42,
            sample_s1=5,
            sample_target=10,
            max_candidates=50,
        )

        # 1. Check summary JSON
        summary_path = self.out_dir / "pilot_summary.json"
        self.assertTrue(summary_path.exists())
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        self.assertEqual(summary["mode"], "pilot")
        self.assertIn("Candidate recall is NOT computed", summary["notice"])
        self.assertEqual(summary["sampled_counts"]["source1_validation_queries"], 5)
        self.assertEqual(summary["sampled_counts"]["source2_targets"], 10)
        self.assertEqual(summary["sampled_counts"]["source3_targets"], 10)
        self.assertTrue(summary["validation_schema_valid"])
        self.assertIn("query_throughput_qps", summary["timing_seconds"])

        # 2. Check candidate TSV outputs
        pairs_path = self.out_dir / "pilot_candidate_pairs.tsv"
        prov_path = self.out_dir / "pilot_candidate_provenance.tsv"
        self.assertTrue(pairs_path.exists())
        self.assertTrue(prov_path.exists())

        pairs_df = pd.read_csv(pairs_path, sep="\t", dtype=str, keep_default_na=False)
        self.assertEqual(len(pairs_df), 5)
        self.assertEqual(list(pairs_df.columns), ["source1_entity_id", "candidate_entity_ids"])


if __name__ == "__main__":
    unittest.main()
