"""Cohort-level classification summaries and stratified bootstrap intervals."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, confusion_matrix,
    f1_score, roc_auc_score,
)

from analysis_core import LEVEL2_CLASSES


def score_summary(y_true: np.ndarray, scores: np.ndarray, task: str) -> dict[str, float]:
    """Report undefined discrimination as NaN, never as zero or a pooled substitute."""
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    if len(y) == 0 or len(s) != len(y) or not np.isfinite(s).all():
        raise ValueError("Scores must be finite and aligned with nonempty labels")
    if task == "level2":
        if s.shape != (len(y), 4):
            raise ValueError("Level 2 scores must have four columns in AML/ALL/CML/CLL order")
        per_auc, per_pr = [], []
        for i, cls in enumerate(LEVEL2_CLASSES):
            binary = (y == cls).astype(int)
            per_auc.append(roc_auc_score(binary, s[:, i]) if len(np.unique(binary)) == 2 else np.nan)
            per_pr.append(average_precision_score(binary, s[:, i]) if binary.sum() else np.nan)
        pred = LEVEL2_CLASSES[np.argmax(s, axis=1)]
        complete = set(np.unique(y)) == set(LEVEL2_CLASSES)
        return {
            "n": int(len(y)),
            "macro_auroc_ovr": float(np.mean(per_auc)) if complete else np.nan,
            "macro_auprc": float(np.mean(per_pr)) if complete else np.nan,
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)) if complete else np.nan,
            "macro_f1": float(f1_score(y, pred, labels=LEVEL2_CLASSES, average="macro", zero_division=0)) if complete else np.nan,
        }
    if task not in {"level1", "level3"} or s.ndim != 1 or not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("Binary task requires one-dimensional scores and 0/1 labels")
    pred = (s >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    both = len(np.unique(y)) == 2
    return {
        "n": int(len(y)),
        "prevalence": float(y.mean()),
        "auroc": float(roc_auc_score(y, s)) if both else np.nan,
        "auprc": float(average_precision_score(y, s)) if both else np.nan,
        "balanced_accuracy_at_0.5": float(balanced_accuracy_score(y, pred)) if both else np.nan,
        "sensitivity_at_0.5": float(tp / (tp + fn)) if tp + fn else np.nan,
        "specificity_at_0.5": float(tn / (tn + fp)) if tn + fp else np.nan,
    }


def stratified_indices(y_true: np.ndarray, n_boot: int, seed: int) -> Iterator[np.ndarray]:
    """Resample rows within each observed class, preserving cohort prevalence."""
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    y = np.asarray(y_true)
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(y == cls) for cls in np.unique(y)]
    for _ in range(n_boot):
        yield np.concatenate([rng.choice(idx, size=len(idx), replace=True) for idx in groups])


def bootstrap_ci(
    y_true: np.ndarray, scores: np.ndarray, task: str,
    n_boot: int = 500, seed: int = 20260702,
) -> dict[str, dict[str, float]]:
    """Conditional percentile 95% CIs for performance, preserving class counts."""
    y = np.asarray(y_true)
    s = np.asarray(scores)
    observed = score_summary(y, s, task)
    draws = {name: [] for name in observed}
    for idx in stratified_indices(y, n_boot, seed):
        for name, value in score_summary(y[idx], s[idx], task).items():
            if np.isfinite(value):
                draws[name].append(value)
    result = {}
    for name, point in observed.items():
        values = np.asarray(draws[name])
        descriptive = name in {"n", "prevalence"}
        result[name] = {
            "estimate": float(point),
            "ci_low": float(np.quantile(values, 0.025)) if len(values) and np.isfinite(point) and not descriptive else np.nan,
            "ci_high": float(np.quantile(values, 0.975)) if len(values) and np.isfinite(point) and not descriptive else np.nan,
        }
    return result
