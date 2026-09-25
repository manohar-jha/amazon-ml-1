# Amazon Business Entity Resolution Challenge

A modular, reproducible Entity Resolution (ER) solution for matching **Source 1** business entities against candidate entities in **Source 2** and **Source 3**.

---

## Project Structure

```text
amazon-entity-resolution/
│
├── src/
│   ├── __init__.py
│   ├── config.py              # Dynamic path configuration & hyperparameters
│   ├── schemas.py             # Shared schema definitions, column constants & dataclasses
│   ├── io_utils.py            # Strict TSV reading/writing & deterministic ID list serialization
│   ├── data_loader.py         # TSV dataset loading & schema validation
│   ├── normalize.py           # Unicode-safe text, address, and country normalization
│   ├── lsh.py                 # Character 3-gram MinHash LSH with universal hashing
│   ├── sharded_index.py       # Compact TargetIndexShard with deterministic 2-pass indexing
│   ├── batch_pipeline.py      # Single-shard-resident streaming pipeline & manifest fingerprinting
│   ├── blocking.py            # High-recall multi-rule inverted index candidate blocking
│   ├── split_utils.py         # Deterministic entity-level split creation and persistence
│   ├── split.py               # CLI tool to generate and save persistent entity splits
│   ├── features.py            # [Phase 4] Pairwise feature extraction interface
│   ├── model_lgb.py           # [Phase 5] LightGBM matching model interface
│   ├── model_transformer.py   # [Phase 6] Transformer reranking interface
│   ├── fusion.py              # [Phase 7] Model fusion & threshold tuning interface
│   ├── evaluate.py            # Entity-Macro F0.5 scoring metric and evaluation utilities
│   └── submission.py          # [Phase 8] Final submission generator interface
│
├── utils/
│   └── validate_submission.py # Official competition invariant validator
│
├── data/
│   ├── train/                 # Training datasets (train_source1/2/3.tsv, train_ground_truth.tsv)
│   └── test/                  # Test datasets (test_source1/2/3.tsv)
│
├── output/                    # Model artifacts, score tables, splits, and submissions
│   ├── splits/                # Persistent S1 entity split files (train, calib, val)
│   └── candidate_pairs.tsv    # Generated candidate pairs for test
│
├── tests/
│   ├── test_foundation.py     # Synthetic contract tests (schemas, I/O, splits, F0.5 metric)
│   ├── test_normalize.py      # Unicode normalization tests (Indic scripts, French accents)
│   ├── test_blocking.py       # Blocking tests (unseen countries, candidate containment)
│   ├── test_phase1.py         # MinHash LSH, inverted index, & validation recall tests
│   ├── test_pilot.py          # Bounded pilot mode & deterministic sampling tests
│   └── test_sharded_pipeline.py # Single-shard residency, manifest rejection & resume tests
│
├── kaggle_phase1.py           # Phase 1 candidate generation, bounded pilot & validation CLI
├── requirements.txt           # Pinned/bounded dependencies
└── README.md
```

---

## Key Design Principles & Architecture

1. **Strict Single-Shard Residency**:
   - The target search space ($S2 + S3 \approx 10.3\text{M}$ records) is partitioned into bounded target files on disk (`partitions/target_shards/target_shard_XXXX.tsv`).
   - Retrieval executes in a **shard-major schedule**: exactly **one target index shard** is built and resident in RAM at any given time.
   - S1 query batches are streamed through the single resident shard, writing intermediate candidate matches directly to disk (`partitions/intermediate/shard_XXXX_batch_YYYY.tsv`).
   - The resident shard is explicitly unloaded (`del shard; gc.collect()`) before the next target shard is indexed.
2. **Disk-Backed Top-K & Deduplication Store**:
   - Intermediate candidate files across all target shards are merged on a per-batch basis.
   - Matches for each Source 1 query are deduplicated, keeping the highest rule score, sorted deterministically by `(-score, candidate_id)`, and capped at `max_candidates=250`.
   - Online validation recall metrics are accumulated incrementally without double-counting upon resume.
3. **Cryptographic Manifest Fingerprinting**:
   - Manifests are keyed by SHA-256 fingerprint over input file metadata (sizes, modification timestamps), split files, `shard_size`, `batch_size`, `max_candidates`, and blocking hyperparameters.
   - Incompatible or stale manifests are rejected automatically, wiping stale partitions and restarting cleanly to prevent corruption.
4. **Deterministic & Order-Independent Indexing**:
   - `TargetIndexShard` uses a 2-pass architecture: Pass 1 computes global document frequencies across the shard; Pass 2 builds postings deterministically filtered against frequency caps.
5. **Safe Resource Preflight Checks**:
   - Multi-platform RAM detection (via `psutil`, `/proc/meminfo`, or Windows `MEMORYSTATUSEX`) without hardcoded fallback numbers.
   - Preflight checks account for measured shard peak memory plus required headroom and disk headroom, failing safely if memory cannot be measured or is insufficient.
6. **No External Data or APIs**: The solution strictly uses the provided challenge data.
7. **Address Parsing Integrity**: Raw business addresses contain arbitrary commas and **must never be parsed by splitting on commas**. Comma splitting is strictly restricted to parsing target ID lists (`matched_entity_ids` / `candidate_entity_ids`).
8. **Open Country Support**: Countries are open-set strings (`us`, `india`, `france`, `germany`, etc.). The code does not restrict blocking to a fixed country list, allowing France (present in test) to work automatically.

---

## Installation & Testing

```bash
# Install bounded dependencies
pip install -r requirements.txt

# Run all synthetic contract and unit tests (38 tests in < 3.5s)
python -m unittest discover -s tests
```

---

## Execution on Kaggle

### 1. Bounded Phase 1 Pilot (`--mode pilot` - Verified Safe Benchmark)

To safely benchmark throughput, process memory, and plumbing without running out of RAM, use the streaming bounded pilot. It reads source TSVs in bounded chunks (`chunksize=50,000`), scans up to a configurable row window (`scan_rows=250,000`), and selects deterministic ID-hash samples (1,000 validation S1 queries, 5,000 S2 targets, 5,000 S3 targets).

```bash
python kaggle_phase1.py \
  --mode pilot \
  --data-dir /kaggle/input/datasets/manoharjha17/amazon/student_resource/dataset \
  --pilot-scan-rows 250000 \
  --pilot-chunksize 50000 \
  --pilot-seed 42 \
  --sample-s1 1000 \
  --sample-target 5000 \
  --track-memory
```

> [!NOTE]
> **Pilot Disclaimer**: Pilot mode is strictly for resource, memory, and plumbing validation. Candidate recall is **not** computed because prefix-window target sampling is not representative of full-corpus recall. Results and metrics are written to `output/pilot_summary.json`.

---

### 2. Disk-Backed Full Validation Recall Evaluation (`--mode eval_val`)

Measures true candidate recall on the saved validation split against `train_ground_truth.tsv` using the single-shard-resident pipeline. Targets (~10.3M rows) are sliced into disk shards (~2M rows each), with S1 validation queries processed in streaming batches (default 25,000 queries/batch) and online incremental recall computation:

```bash
python kaggle_phase1.py \
  --mode eval_val \
  --data-dir /kaggle/input/datasets/manoharjha17/amazon/student_resource/dataset \
  --batch-size-s1 25000 \
  --shard-size-targets 2000000 \
  --track-memory
```

- **Residency Guarantee**: At most 1 target shard is resident in RAM at any time.
- **Fault-Tolerance**: If preempted or stopped, re-running automatically resumes from the last completed batch partition using `output/validation_manifest.json`.
- **Output Artifacts**: `output/candidate_pairs.tsv`, `output/candidate_provenance.tsv`, and `output/validation_summary.json`.

---

### 3. Disk-Backed Full Test Candidate Generation (`--mode test`)

Generates `output/candidate_pairs.tsv` and pair-level `output/candidate_provenance.tsv` for all test Source 1 entities (~1.73M queries) against test targets (~9.97M rows, including France, US, and India):

```bash
python kaggle_phase1.py \
  --mode test \
  --data-dir /kaggle/input/datasets/manoharjha17/amazon/student_resource/dataset \
  --batch-size-s1 25000 \
  --shard-size-targets 2000000 \
  --track-memory
```

- **Output Artifacts**: `output/candidate_pairs.tsv`, `output/candidate_provenance.tsv`, and `output/test_summary.json`.
