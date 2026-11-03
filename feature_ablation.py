"""Training-only comparisons of prespecified CBC feature panels."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

from analysis_core import TASK_FEATURES, training_oof_scores
from metrics import score_summary, stratified_indices


def _midrank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[start:stop] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    result = np.empty_like(ranks)
    result[order] = ranks
    return result


def delong_auc_delta(y_true: np.ndarray, candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """Paired DeLong CI for candidate minus reference binary AUROC."""
    y = np.asarray(y_true, dtype=int)
    if set(np.unique(y)) != {0, 1} or min(np.bincount(y)) < 2:
        raise ValueError("DeLong comparison needs at least two rows per binary class")
    candidate = np.asarray(candidate, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if candidate.shape != y.shape or reference.shape != y.shape:
        raise ValueError("Both score arrays must align with binary labels")
    ordered = np.argsort(-y)
    paired = np.vstack((candidate[ordered], reference[ordered]))
    positives = int(y.sum())
    negatives = len(y) - positives
    tx = np.array([_midrank(row[:positives]) for row in paired])
    ty = np.array([_midrank(row[positives:]) for row in paired])
    tz = np.array([_midrank(row) for row in paired])
    aucs = tz[:, :positives].sum(axis=1) / (positives * negatives) - (positives + 1.0) / (2.0 * negatives)
    v01 = (tz[:, :positives] - tx) / negatives
    v10 = 1.0 - (tz[:, positives:] - ty) / positives
    covariance = np.cov(v01) / positives + np.cov(v10) / negatives
    delta = float(aucs[0] - aucs[1])
    variance = float(covariance[0, 0] + covariance[1, 1] - 2 * covariance[0, 1])
    se = float(np.sqrt(max(variance, 0.0)))
    if se == 0 and delta != 0:
        raise ValueError("Degenerate DeLong variance for nonidentical AUROCs")
    return {
        "delta_auroc": delta,
        "ci_low": delta - 1.96 * se,
        "ci_high": delta + 1.96 * se,
        "p_two_sided": float(2 * norm.sf(abs(delta / se))) if se else 1.0,
    }


def macro_auc_delta_bootstrap(
    y_true: np.ndarray, candidate: np.ndarray, reference: np.ndarray,
    n_boot: int = 500, seed: int = 20260702,
) -> dict[str, float]:
    """Paired, class-stratified percentile CI for four-class macro-AUROC delta."""
    y = np.asarray(y_true, dtype=int)
    a = np.asarray(candidate, dtype=float)
    b = np.asarray(reference, dtype=float)
    if a.shape != b.shape or a.shape != (len(y), 4):
        raise ValueError("Both four-class score matrices must have identical shape")
    metric = "macro_auroc_ovr"
    delta = score_summary(y, a, "level2")[metric] - score_summary(y, b, "level2")[metric]
    if not np.isfinite(delta):
        raise ValueError("All four labels must occur in the training OOF cohort")
    draws = []
    for idx in stratified_indices(y, n_boot, seed):
        da = score_summary(y[idx], a[idx], "level2")[metric]
        db = score_summary(y[idx], b[idx], "level2")[metric]
        draws.append(da - db)
    return {
        "delta_auroc": float(delta),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
    }


def compare_training_panels(
    train: pd.DataFrame, task: str, candidates: Mapping[str, Sequence[str]],
    margin: float = 0.01, n_boot: int = 500,
) -> pd.DataFrame:
    """Compare prespecified smaller panels with the locked task panel, training only.

    A candidate is flagged only if the lower 95% bound for its AUROC difference
    exceeds -margin. This is a screening illustration, not the study's complete
    SHAP-ranking and feature-selection pipeline.
    """
    if task not in TASK_FEATURES or not 0 < margin < 1:
        raise ValueError("Invalid task or noninferiority margin")
    y = train["target"].astype(int).to_numpy()
    reference = training_oof_scores(train, task)
    rows = []
    for name, features in candidates.items():
        chosen = tuple(features)
        if not chosen or len(chosen) >= len(TASK_FEATURES[task]):
            raise ValueError("Candidate panels must be smaller than the locked panel")
        scores = training_oof_scores(train, task, chosen)
        comparison = (
            macro_auc_delta_bootstrap(y, scores, reference, n_boot=n_boot)
            if task == "level2" else delong_auc_delta(y, scores, reference)
        )
        rows.append({
            "panel": name,
            "n_features": len(chosen),
            "method": "paired bootstrap" if task == "level2" else "paired DeLong",
            **comparison,
            "noninferior_at_margin": bool(comparison["ci_low"] > -margin),
        })
    return pd.DataFrame(rows).sort_values(["n_features", "panel"]).reset_index(drop=True)
