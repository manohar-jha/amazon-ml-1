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
│   ├── test_phase1.py         # Phase 1 MinHash LSH, inverted index, & validation recall tests
│   └── test_pilot.py          # Bounded pilot mode & deterministic sampling tests
│
├── kaggle_phase1.py           # Phase 1 candidate generation, bounded pilot & validation CLI
├── requirements.txt           # Pinned/bounded dependencies
└── README.md
```

---

## Key Design Principles & Invariants

1. **No External Data or APIs**: The solution strictly uses the provided challenge data. No external web requests, geocoders, or company registries are permitted.
2. **Explicit Delimiters**: All datasets are tab-separated (`sep='\t'`).
3. **Address Parsing Integrity**: Raw business addresses contain arbitrary commas and **must never be parsed by splitting on commas**. Comma splitting is strictly restricted to parsing target ID lists (`matched_entity_ids` / `candidate_entity_ids`).
4. **Open Country Support**: Countries are open-set strings (`us`, `india`, `france`, `germany`, etc.). The code does not restrict blocking to a fixed country list, allowing France (present in test) to work automatically.
5. **Deterministic Serialization**: ID lists in ground truth, candidate pairs, and final submissions are sorted and deduplicated (e.g. `S2-10,S3-20`), with empty strings `""` for true singletons.
6. **Submission Invariants**:
   - `matching_results.tsv` and `candidate_pairs.tsv` contain exactly one row per test S1 ID.
   - Zero duplicate S1 IDs; zero duplicate target IDs within a row.
   - **Candidate Containment**: Every predicted match in `matching_results.tsv` **must be present** in that entity's candidate list in `candidate_pairs.tsv`.

---

## Entity-Level Validation Split Methodology

To ensure unbiased evaluation and prevent data leakage:
- **Grouping Unit**: Splitting is strictly performed at the **Source 1 entity level**. All candidate pairs for a given Source 1 entity reside in the exact same partition.
- **Three-Way Entity Split**:
  1. **Train Set (70%)**: Used to fit matching models (LightGBM, transformers).
  2. **Calibration / Fusion Set (15%)**: Used for out-of-fold calibration, score scaling, and rank fusion.
  3. **Threshold Validation Set (15%)**: Kept untouched during model training; strictly reserved for optimal decision threshold tuning and unbiased final Entity-Macro $F_{0.5}$ reporting.
- **Persistent ID Files**: Split IDs are generated once using a fixed seed (`seed=42`) and written to `output/splits/` (`train_s1_ids.txt`, `calibration_s1_ids.txt`, `validation_s1_ids.txt`). Downstream stages load these exact ID files rather than dynamically re-sampling.

---

## Installation & Testing

```bash
# Install bounded dependencies
pip install -r requirements.txt

# Run all unit tests with synthetic data (< 0.5s runtime)
python -m unittest discover -s tests
```

---

## Kaggle Execution

### 1. Bounded Phase 1 Pilot (`--mode pilot` - Recommended First Run)

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

### 2. Full Validation Recall Evaluation (`--mode eval_val`)

Measures true candidate recall on the saved validation split against `train_ground_truth.tsv`.

```bash
python kaggle_phase1.py \
  --mode eval_val \
  --data-dir /kaggle/input/datasets/manoharjha17/amazon/student_resource/dataset \
  --track-memory
```

> [!WARNING]
> **Memory Warning**: Full validation loads complete Source 1 (~2.2M rows), Source 2 (~5.0M rows), Source 3 (~5.3M rows), and ground truth into memory to build full indexes. This requires high RAM (16GB+). For quick resource and plumbing checks, always use `--mode pilot` first.

---

### 3. Full Test Candidate Generation (`--mode test`)

Generates `output/candidate_pairs.tsv` and pair-level `output/candidate_provenance.tsv` for all test Source 1 entities:

```bash
python kaggle_phase1.py \
  --mode test \
  --data-dir /kaggle/input/datasets/manoharjha17/amazon/student_resource/dataset \
  --track-memory
```

