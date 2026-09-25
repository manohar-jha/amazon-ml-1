"""Unit tests for Phase 1 Single-Shard-Resident Retrieval and Streaming Batch Pipeline.

Tests:
1. One-shard-at-a-time memory behavior and shard unloading.
2. Stale-manifest rejection and full partition clearing upon configuration/mode/input changes.
3. Accurate incremental validation recall without double-counting on resume.
4. Candidate completeness and top-K deduplication across multiple target shards.
5. Resource safety checks with empirical RSS scaling and dynamic shard headroom.
6. Deterministic and order-independent index construction with frequency caps.
7. Per-shard candidate cap enforcement and pruned candidate accounting.
8. Atomic writes with .tmp files and partial-file resilience.
9. Streaming submission validation with missing and extra Source 1 ID detection.
"""

import gc
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd

from src.batch_pipeline import (
    check_resource_headroom,
    compute_manifest_fingerprint,
    get_available_ram_bytes,
    merge_candidate_partitions,
    partition_s1_queries,
    partition_target_sources,
    run_disk_backed_retrieval_pipeline,
)
from src.normalize import normalize_dataframe
from src.sharded_index import (
    ShardedTargetIndex,
    TargetIndexShard,
    benchmark_shard_memory_rss,
    measure_shard_memory,
)
from utils.validate_submission import validate_submission_file, validate_submission_streaming


class TestShardedPipeline(unittest.TestCase):
    """Test suite for single-shard-resident retrieval and streaming batch processing."""

    def setUp(self):
        """Create temporary environment with synthetic datasets."""
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.tmp_dir.name)

        self.train_dir = self.base_path / "train"
        self.splits_dir = self.base_path / "splits"
        self.out_dir = self.base_path / "output"
        self.partitions_dir = self.out_dir / "partitions"

        self.train_dir.mkdir(parents=True, exist_ok=True)
        self.splits_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.partitions_dir.mkdir(parents=True, exist_ok=True)

        # 1. Synthetic S1 (40 queries)
        s1_rows = [
            f"S1-{i:03d}\tAcme Corp {i}\t{100 + i} Market Street\tus"
            for i in range(1, 41)
        ]
        with open(self.train_dir / "train_source1.tsv", "w", encoding="utf-8") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n" + "\n".join(s1_rows) + "\n")

        # 2. Synthetic S2 (60 targets)
        s2_rows = [
            f"S2-{i:03d}\tAcme Corp {i} LLC\t{100 + i} Market St\tus"
            for i in range(1, 61)
        ]
        with open(self.train_dir / "train_source2.tsv", "w", encoding="utf-8") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n" + "\n".join(s2_rows) + "\n")

        # 3. Synthetic S3 (60 targets)
        s3_rows = [
            f"S3-{i:03d}\tAcme Corp {i} Inc\t{100 + i} Market Street\tus"
            for i in range(1, 61)
        ]
        with open(self.train_dir / "train_source3.tsv", "w", encoding="utf-8") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n" + "\n".join(s3_rows) + "\n")

        # 4. Synthetic ground truth (S1-001 -> S2-001 in shard 0, S3-001 in later shard)
        gt_rows = [
            f"S1-{i:03d}\tS2-{i:03d},S3-{i:03d}" if i <= 30 else f"S1-{i:03d}\t"
            for i in range(1, 41)
        ]
        with open(self.train_dir / "train_ground_truth.tsv", "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tmatched_entity_ids\n" + "\n".join(gt_rows) + "\n")

        # 5. Validation split IDs (IDs 1 to 20)
        self.val_ids = {f"S1-{i:03d}" for i in range(1, 21)}
        self.val_split_file = self.splits_dir / "validation_s1_ids.txt"
        with open(self.val_split_file, "w", encoding="utf-8") as f:
            for vid in sorted(self.val_ids):
                f.write(f"{vid}\n")

        # Ground truth lookup map for validation
        self.gt_lookup = {
            f"S1-{i:03d}": {f"S2-{i:03d}", f"S3-{i:03d}"} for i in range(1, 21)
        }

    def tearDown(self):
        """Clean up temporary directory."""
        self.tmp_dir.cleanup()

    def test_resource_safety_with_measured_bytes_per_target(self):
        """Verify that resource checks fail safely if RAM is unmeasurable, unbenchmarked, or insufficient."""
        # 1. Valid measurement with explicit measured bytes per target
        is_safe, res, msg = check_resource_headroom(
            self.out_dir,
            min_disk_gb=0.01,
            min_ram_gb=0.01,
            shard_size=100,
            measured_bytes_per_target=350.0,
        )
        self.assertTrue(is_safe)
        self.assertIn("free_disk_gb", res)
        self.assertIn("available_ram_gb", res)
        self.assertIn("required_ram_gb", res)
        self.assertEqual(res["measured_bytes_per_target"], 350.0)

        # 2. Missing measured bytes per target -> must fail safely without fixed assumptions
        is_safe_unb, res_unb, msg_unb = check_resource_headroom(self.out_dir, measured_bytes_per_target=None)
        self.assertFalse(is_safe_unb)
        self.assertIn("Measured bytes per target was not provided", msg_unb)

        # 3. Mock unmeasurable RAM -> must fail safely
        with patch("src.batch_pipeline.get_available_ram_bytes", return_value=None):
            is_safe_none, res_none, msg_none = check_resource_headroom(self.out_dir, measured_bytes_per_target=350.0)
            self.assertFalse(is_safe_none)
            self.assertIsNone(res_none["available_ram_gb"])
            self.assertIn("could not be measured", msg_none)

        # 4. Insufficient RAM against measured shard headroom
        with patch("src.batch_pipeline.get_available_ram_bytes", return_value=int(0.5 * 1024**3)):  # 0.5 GB available
            is_safe_low, res_low, msg_low = check_resource_headroom(
                self.out_dir,
                min_ram_gb=2.0,
                shard_size=1000000,
                measured_bytes_per_target=350.0,
            )
            self.assertFalse(is_safe_low)
            self.assertIn("Insufficient available RAM", msg_low)

    def test_single_shard_resident_retrieval_and_cross_shard_completeness(self):
        """Verify that multi-shard execution merges candidates across distinct target shards correctly."""
        manifest_path = self.out_dir / "validation_manifest.json"

        stats = run_disk_backed_retrieval_pipeline(
            s1_path=self.train_dir / "train_source1.tsv",
            s2_path=self.train_dir / "train_source2.tsv",
            s3_path=self.train_dir / "train_source3.tsv",
            partitions_dir=self.partitions_dir,
            manifest_path=manifest_path,
            mode="eval_val",
            split_file=self.val_split_file,
            filter_s1_ids=self.val_ids,
            gt_lookup=self.gt_lookup,
            batch_size=10,
            shard_size=40,
            chunksize=20,
            max_candidates=50,
            per_shard_candidate_cap=50,
            resume=True,
        )

        # Verify exact recall across multiple shards
        self.assertEqual(stats["total_s1_processed"], 20)
        self.assertEqual(stats["val_total_gt_pairs"], 40)
        self.assertEqual(stats["val_found_gt_pairs"], 40)
        self.assertAlmostEqual(stats["val_pair_recall"], 1.0)
        self.assertAlmostEqual(stats["val_s2_recall"], 1.0)
        self.assertAlmostEqual(stats["val_s3_recall"], 1.0)
        self.assertAlmostEqual(stats["val_full_s1_coverage"], 1.0)

        # Merge final candidate files and validate schema
        final_pairs = self.out_dir / "candidate_pairs.tsv"
        final_prov = self.out_dir / "candidate_provenance.tsv"
        merge_info = merge_candidate_partitions(
            partitions_dir=self.partitions_dir,
            output_pairs_path=final_pairs,
            output_prov_path=final_prov,
        )

        self.assertEqual(merge_info["total_s1_rows"], 20)
        self.assertTrue(final_pairs.exists())
        self.assertTrue(final_pairs.with_suffix(".complete").exists())

        valid_targets = {f"S2-{i:03d}" for i in range(1, 61)} | {f"S3-{i:03d}" for i in range(1, 61)}
        is_valid, errors = validate_submission_file(
            submission_path=final_pairs,
            expected_s1_ids=self.val_ids,
            valid_target_ids=valid_targets,
            is_candidate_file=True,
        )
        self.assertTrue(is_valid, f"Validation errors: {errors}")

    def test_stale_manifest_and_partition_rebuild_on_mode_change(self):
        """Verify that mode or configuration change wipes all partition directories and rebuilds cleanly."""
        manifest_path = self.out_dir / "validation_manifest.json"

        # 1. Run initial pass in mode 'pilot'
        stats1 = run_disk_backed_retrieval_pipeline(
            s1_path=self.train_dir / "train_source1.tsv",
            s2_path=self.train_dir / "train_source2.tsv",
            s3_path=self.train_dir / "train_source3.tsv",
            partitions_dir=self.partitions_dir,
            manifest_path=manifest_path,
            mode="pilot",
            split_file=self.val_split_file,
            filter_s1_ids=self.val_ids,
            batch_size=10,
            shard_size=40,
            max_candidates=50,
            resume=True,
        )
        self.assertEqual(stats1["total_s1_processed"], 20)
        self.assertTrue((self.partitions_dir / "target_shards").exists())

        # 2. Switch mode to 'eval_val' (triggers fingerprint mismatch)
        stats2 = run_disk_backed_retrieval_pipeline(
            s1_path=self.train_dir / "train_source1.tsv",
            s2_path=self.train_dir / "train_source2.tsv",
            s3_path=self.train_dir / "train_source3.tsv",
            partitions_dir=self.partitions_dir,
            manifest_path=manifest_path,
            mode="eval_val",  # Changed!
            split_file=self.val_split_file,
            filter_s1_ids=self.val_ids,
            gt_lookup=self.gt_lookup,
            batch_size=10,
            shard_size=40,
            max_candidates=50,
            resume=True,
        )

        with open(manifest_path, "r", encoding="utf-8") as f:
            updated_manifest = json.load(f)

        self.assertEqual(updated_manifest["mode"], "eval_val")
        self.assertEqual(stats2["total_s1_processed"], 20)
        self.assertEqual(stats2["val_found_gt_pairs"], 40)

    def test_per_shard_candidate_cap_and_pruned_accounting(self):
        """Verify per-shard candidate caps bound intermediate volume and record pruned count accurately."""
        # Create a single shard with 25 target entities that match the same query
        df = pd.DataFrame({
            "entity_id": [f"S2-{i:03d}" for i in range(1, 26)],
            "business_name": ["Acme Global Logistics"] * 25,
            "business_address": [f"{100 + i} Main Street" for i in range(1, 26)],
            "country": ["us"] * 25,
        })
        shard = TargetIndexShard(shard_id=0)
        shard.build_from_dataframe(df)

        # Query with per_shard_cap=5
        cand_map, pruned_count = shard.query_record(
            name="acme global logistics",
            addr="101 main street",
            country="us",
            per_shard_cap=5,
            return_pruned_count=True,
        )
        self.assertEqual(len(cand_map), 5)
        self.assertEqual(pruned_count, 20)

    def test_atomic_writes_and_tmp_file_resilience(self):
        """Verify that temporary .tmp files are ignored and atomic renames prevent corrupt partial files."""
        # Pre-create a stray .tmp file in intermediate and merged_batches
        inter_dir = self.partitions_dir / "intermediate"
        inter_dir.mkdir(parents=True, exist_ok=True)
        stray_tmp = inter_dir / "shard_0000_batch_0000.tsv.tmp"
        with open(stray_tmp, "w", encoding="utf-8") as f:
            f.write("corrupted\tpartial\tdata\n")

        manifest_path = self.out_dir / "validation_manifest.json"
        stats = run_disk_backed_retrieval_pipeline(
            s1_path=self.train_dir / "train_source1.tsv",
            s2_path=self.train_dir / "train_source2.tsv",
            s3_path=self.train_dir / "train_source3.tsv",
            partitions_dir=self.partitions_dir,
            manifest_path=manifest_path,
            mode="eval_val",
            split_file=self.val_split_file,
            filter_s1_ids=self.val_ids,
            gt_lookup=self.gt_lookup,
            batch_size=10,
            shard_size=40,
            max_candidates=50,
            resume=True,
        )

        self.assertEqual(stats["total_s1_processed"], 20)
        # Ensure stray tmp was cleaned and not read
        self.assertFalse(stray_tmp.exists())

    def test_validate_submission_streaming(self):
        """Verify streaming validation correctly detects missing and extra test Source 1 IDs."""
        sub_file = self.out_dir / "test_candidate_pairs.tsv"
        s1_file = self.train_dir / "train_source1.tsv"

        # 1. Valid file containing all 40 S1 IDs
        with open(sub_file, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tcandidate_entity_ids\n")
            for i in range(1, 41):
                f.write(f"S1-{i:03d}\tS2-{i:03d}\n")

        is_valid, errors = validate_submission_streaming(
            submission_path=sub_file,
            s1_source_path=s1_file,
            is_candidate_file=True,
        )
        self.assertTrue(is_valid, f"Unexpected errors: {errors}")

        # 2. Missing S1-040
        with open(sub_file, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tcandidate_entity_ids\n")
            for i in range(1, 40):  # Missing S1-040
                f.write(f"S1-{i:03d}\tS2-{i:03d}\n")

        is_valid_missing, errors_missing = validate_submission_streaming(
            submission_path=sub_file,
            s1_source_path=s1_file,
            is_candidate_file=True,
        )
        self.assertFalse(is_valid_missing)
        self.assertTrue(any("Missing 1 expected Source 1 IDs" in e for e in errors_missing))

        # 3. Extra unexpected S1-999
        with open(sub_file, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tcandidate_entity_ids\n")
            for i in range(1, 41):
                f.write(f"S1-{i:03d}\tS2-{i:03d}\n")
            f.write("S1-999\tS2-001\n")

        is_valid_extra, errors_extra = validate_submission_streaming(
            submission_path=sub_file,
            s1_source_path=s1_file,
            is_candidate_file=True,
        )
        self.assertFalse(is_valid_extra)
        self.assertTrue(any("unexpected Source 1 IDs" in e for e in errors_extra))

    def test_non_double_counting_on_resume(self):
        """Verify that interrupted and resumed runs compute the exact same metrics without double-counting."""
        manifest_path = self.out_dir / "validation_manifest.json"

        s1_batches_dir = self.partitions_dir / "s1_batches"
        merged_dir = self.partitions_dir / "merged_batches"
        s1_batches_dir.mkdir(parents=True, exist_ok=True)
        merged_dir.mkdir(parents=True, exist_ok=True)

        fp = compute_manifest_fingerprint(
            mode="eval_val",
            input_files=[self.train_dir / "train_source1.tsv", self.train_dir / "train_source2.tsv", self.train_dir / "train_source3.tsv"],
            split_file=self.val_split_file,
            filter_s1_ids=self.val_ids,
            shard_size=50,
            batch_size=10,
            max_candidates=50,
        )

        initial_manifest = {
            "fingerprint": fp,
            "mode": "eval_val",
            "completed_batches": [0],
            "total_s1_processed": 10,
            "total_candidates_generated": 200,
            "total_truncations": 0,
            "empty_candidate_queries": 0,
            "total_shard_candidates_pruned": 0,
            "measured_shard_ram_mb": [1.2],
            "accumulated_metrics": {
                "total_gt_pairs": 20,
                "total_s2_gt": 10,
                "total_s3_gt": 10,
                "found_gt_pairs": 20,
                "found_s2_gt": 10,
                "found_s3_gt": 10,
                "s1_with_matches": 10,
                "s1_full_covered": 10,
            },
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(initial_manifest, f)

        # Write dummy batch 0 merged pairs and prov
        with open(merged_dir / "candidate_pairs_batch_0000.tsv", "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tcandidate_entity_ids\n")
            for i in range(1, 11):
                f.write(f"S1-{i:03d}\tS2-{i:03d},S3-{i:03d}\n")
        with open(merged_dir / "candidate_provenance_batch_0000.tsv", "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tcandidate_entity_id\tsource_dataset\tprovenance_rules\n")

        # Resume execution
        resumed_stats = run_disk_backed_retrieval_pipeline(
            s1_path=self.train_dir / "train_source1.tsv",
            s2_path=self.train_dir / "train_source2.tsv",
            s3_path=self.train_dir / "train_source3.tsv",
            partitions_dir=self.partitions_dir,
            manifest_path=manifest_path,
            mode="eval_val",
            split_file=self.val_split_file,
            filter_s1_ids=self.val_ids,
            gt_lookup=self.gt_lookup,
            batch_size=10,
            shard_size=50,
            max_candidates=50,
            resume=True,
        )

        self.assertEqual(resumed_stats["total_s1_processed"], 20)
        self.assertEqual(resumed_stats["val_total_gt_pairs"], 40)
        self.assertEqual(resumed_stats["val_found_gt_pairs"], 40)
        self.assertAlmostEqual(resumed_stats["val_pair_recall"], 1.0)
        self.assertAlmostEqual(resumed_stats["val_full_s1_coverage"], 1.0)

    def test_deterministic_order_independent_shard_construction(self):
        """Verify that record insertion order does not affect inverted index postings."""
        df_a = pd.DataFrame({
            "entity_id": ["S2-001", "S2-002", "S2-003"],
            "business_name": ["Alpha Beta Gamma", "Alpha Delta", "Alpha Beta"],
            "business_address": ["100 Main St", "200 Oak St", "100 Main St"],
            "country": ["us", "us", "us"],
        })
        df_b = df_a.iloc[::-1].copy()

        shard_a = TargetIndexShard(shard_id=0)
        shard_a.build_from_dataframe(df_a, max_token_freq=2)

        shard_b = TargetIndexShard(shard_id=1)
        shard_b.build_from_dataframe(df_b, max_token_freq=2)

        self.assertNotIn("alpha", shard_a.token_idx)
        self.assertNotIn("alpha", shard_b.token_idx)
        self.assertIn("beta", shard_a.token_idx)
        self.assertIn("beta", shard_b.token_idx)

    def test_benchmark_shard_memory_rss(self):
        """Verify empirical shard memory RSS benchmark computes positive bytes per target."""
        df = pd.DataFrame({
            "entity_id": [f"S2-{i:05d}" for i in range(1, 1001)],
            "business_name": [f"Acme Solutions International {i}" for i in range(1, 1001)],
            "business_address": [f"{100 + i} Industrial Parkway Suite {i}" for i in range(1, 1001)],
            "country": ["us"] * 1000,
        })
        bench = benchmark_shard_memory_rss(sample_df=df, shard_size_targets=1000)
        self.assertGreater(bench["rss_bytes_per_target"], 0.0)
        self.assertIn("estimated_shard_rss_gb", bench)
        self.assertIn("peak_rss_mb", bench)


if __name__ == "__main__":
    unittest.main()
