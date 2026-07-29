"""Paratope prediction evaluation metrics.

Matches Paraplume/PECAN/Paragraph evaluation protocol:
  - PR AUC, ROC AUC  (threshold-free)
  - F1, MCC          (at 0.5 threshold)
  - All metrics averaged over proteins in the test set.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)


def per_protein_metrics(
    scores: list[float],
    labels: list[int],
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute all four metrics for a single protein."""
    y_true = np.array(labels, dtype=int)
    y_score = np.array(scores, dtype=float)
    y_pred = (y_score >= threshold).astype(int)

    # Edge case: only one class present → AUC undefined
    if len(np.unique(y_true)) < 2:
        return {"pr_auc": float("nan"), "roc_auc": float("nan"),
                "f1": float("nan"), "mcc": float("nan")}

    return {
        "pr_auc":  float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "f1":      float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc":     float(matthews_corrcoef(y_true, y_pred)),
    }


def aggregate_metrics(per_protein: list[dict[str, float]]) -> dict[str, float]:
    """Mean ± std across proteins, ignoring NaNs (proteins with one class)."""
    keys = ["pr_auc", "roc_auc", "f1", "mcc"]
    result = {}
    for k in keys:
        vals = np.array([m[k] for m in per_protein], dtype=float)
        valid = vals[~np.isnan(vals)]
        result[f"{k}_mean"] = float(valid.mean()) if len(valid) else float("nan")
        result[f"{k}_std"]  = float(valid.std())  if len(valid) else float("nan")
        result[f"{k}_n"]    = int(len(valid))
    return result


def print_table(agg: dict[str, float], label: str = "") -> None:
    if label:
        print(f"\n{'='*50}")
        print(f"  {label}")
        print(f"{'='*50}")
    print(f"  PR AUC : {agg['pr_auc_mean']:.3f}  (±{agg['pr_auc_std']:.3f}, n={agg['pr_auc_n']})")
    print(f"  ROC AUC: {agg['roc_auc_mean']:.3f}  (±{agg['roc_auc_std']:.3f})")
    print(f"  F1     : {agg['f1_mean']:.3f}  (±{agg['f1_std']:.3f})")
    print(f"  MCC    : {agg['mcc_mean']:.3f}  (±{agg['mcc_std']:.3f})")
