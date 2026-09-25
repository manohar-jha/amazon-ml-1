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
│   └── test_blocking.py       # Blocking tests (unseen countries, candidate containment)
│
├── kaggle_phase3.py           # Kaggle entry point for candidate blocking
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

# Run all unit tests with synthetic data (< 0.2s runtime)
python -m unittest discover -s tests
```

---

## Kaggle Execution

In a Kaggle Notebook:
```bash
# Run fast smoke test on 5,000 samples
!python kaggle_phase3.py --mode smoke --sample-s1 5000

# Full train evaluation or test candidate generation
!python kaggle_phase3.py --mode test
```
