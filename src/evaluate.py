"""Evaluation module implementing Entity-Macro F0.5 and validation metrics.

Implements the official Amazon Entity Resolution scoring metric:
- Entity-Macro F0.5 (precision weighted more than recall: beta = 0.5).
- Macro precision, macro recall, pair-level recall, and singleton recovery.
"""

from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd

from src.io_utils import parse_id_list
from src.schemas import COL_MATCHED_IDS, COL_SOURCE1_ID


def calculate_entity_metrics(
    true_set: Set[str],
    pred_set: Set[str],
    beta: float = 0.5,
) -> Tuple[float, float, float]:
    """Calculate Precision, Recall, and F-beta for a single Source 1 entity.

    Correct handling of empty true/prediction sets (singletons):
    - True empty & Pred empty: P=1.0, R=1.0, F=1.0 (True negative singleton)
    - True empty & Pred non-empty: P=0.0, R=1.0, F=0.0 (False positive matches)
    - True non-empty & Pred empty: P=1.0, R=0.0, F=0.0 (False negative singletons)
    - True non-empty & Pred non-empty: standard P, R, F-beta.

    Args:
        true_set: Set of true matching target IDs.
        pred_set: Set of predicted matching target IDs.
        beta: F-measure beta parameter (0.5 places 2x weight on precision).

    Returns:
        Tuple[float, float, float]: (Precision, Recall, F-beta).
    """
    beta_sq = beta ** 2

    # Case 1: True singleton (no matches)
    if not true_set:
        if not pred_set:
            return 1.0, 1.0, 1.0
        else:
            return 0.0, 1.0, 0.0

    # Case 2: True non-empty, but predicted empty
    if not pred_set:
        return 1.0, 0.0, 0.0

    # Case 3: Both non-empty
    tp = len(true_set.intersection(pred_set))
    precision = tp / len(pred_set)
    recall = tp / len(true_set)

    if precision + recall == 0:
        return 0.0, 0.0, 0.0

    f_beta = ((1 + beta_sq) * precision * recall) / (beta_sq * precision + recall)
    return precision, recall, f_beta


def evaluate_predictions(
    ground_truth_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    beta: float = 0.5,
) -> Dict[str, float]:
    """Calculate Entity-Macro F0.5 and comprehensive evaluation metrics.

    Args:
        ground_truth_df: DataFrame with (source1_entity_id, matched_entity_ids).
        predictions_df: DataFrame with (source1_entity_id, matched_entity_ids).
        beta: Beta for F-score (default 0.5).

    Returns:
        Dict[str, float]: Evaluation results including entity_macro_f05, precision, recall.
    """
    # Build maps
    gt_map: Dict[str, Set[str]] = {}
    for _, row in ground_truth_df.iterrows():
        s1_id = str(row[COL_SOURCE1_ID]).strip()
        gt_map[s1_id] = set(parse_id_list(row[COL_MATCHED_IDS]))

    pred_map: Dict[str, Set[str]] = {}
    for _, row in predictions_df.iterrows():
        s1_id = str(row[COL_SOURCE1_ID]).strip()
        pred_map[s1_id] = set(parse_id_list(row[COL_MATCHED_IDS]))

    precisions: List[float] = []
    recalls: List[float] = []
    f_betas: List[float] = []

    total_tp = 0
    total_fp = 0
    total_fn = 0
    singleton_correct = 0
    singleton_total = 0

    for s1_id, true_set in gt_map.items():
        pred_set = pred_map.get(s1_id, set())
        p, r, f = calculate_entity_metrics(true_set, pred_set, beta=beta)
        precisions.append(p)
        recalls.append(r)
        f_betas.append(f)

        tp = len(true_set.intersection(pred_set))
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        if not true_set:
            singleton_total += 1
            if not pred_set:
                singleton_correct += 1

    return {
        "entity_macro_f05": float(np.mean(f_betas)),
        "entity_macro_precision": float(np.mean(precisions)),
        "entity_macro_recall": float(np.mean(recalls)),
        "pair_micro_precision": total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0,
        "pair_micro_recall": total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0,
        "singleton_accuracy": singleton_correct / singleton_total if singleton_total > 0 else 1.0,
        "total_evaluated_s1": len(gt_map),
    }
