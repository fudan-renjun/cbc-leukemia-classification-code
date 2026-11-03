"""Input contract for already harmonized, authorized analysis tables."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from analysis_core import LEVEL2_CLASSES, TASK_FEATURES


@dataclass(frozen=True)
class CheckedCohort:
    data: pd.DataFrame
    input_n: int
    eligible_n: int

    @property
    def excluded_n(self) -> int:
        return self.input_n - self.eligible_n


def check_cohort(frame: pd.DataFrame, task: str, role: str) -> CheckedCohort:
    """Reject invalid labels/nonnumeric values; require complete validation predictors."""
    if task not in TASK_FEATURES or role not in {"train", "internal", "external"}:
        raise ValueError("Unknown task or cohort role")
    features = TASK_FEATURES[task]
    required = {*features, "target"} | ({"cv_fold_5"} if role == "train" else set())
    if len(frame.columns) != len(set(frame.columns)) or not required.issubset(frame.columns):
        raise ValueError(f"Missing or duplicate columns: {sorted(required - set(frame.columns))}")
    if frame.empty:
        raise ValueError("Empty cohort")
    data = frame.copy()
    for name in [*features, "target"]:
        converted = pd.to_numeric(data[name], errors="coerce")
        if (data[name].notna() & converted.isna()).any():
            raise ValueError(f"Nonnumeric value in {name}")
        if np.isinf(converted.to_numpy(dtype=float, na_value=np.nan)).any():
            raise ValueError(f"Infinite value in {name}")
        data[name] = converted
    if data["target"].isna().any():
        raise ValueError("Labeled analysis cohorts require nonmissing target values")
    valid_labels = set(LEVEL2_CLASSES.tolist()) if task == "level2" else {0, 1}
    if not set(data["target"].unique()).issubset(valid_labels):
        raise ValueError(f"Unexpected {task} label; expected {sorted(valid_labels)}")
    if role == "train":
        folds = data["cv_fold_5"]
        if folds.isna().any() or folds.nunique() != 5:
            raise ValueError("Training data require exactly five preassigned fold IDs")
        for fold in pd.unique(folds):
            held = data.loc[folds == fold]
            fitting = data.loc[folds != fold]
            if set(held["target"].unique()) != valid_labels:
                raise ValueError(f"Fold {fold} does not contain every task class")
            if fitting[list(features)].notna().sum().eq(0).any():
                raise ValueError(f"A predictor is entirely missing outside fold {fold}")
        if set(data["target"].unique()) != valid_labels:
            raise ValueError("Training data do not contain all task classes")
    else:
        data = data.loc[data[list(features)].notna().all(axis=1)].copy()
        if data.empty:
            raise ValueError(f"No complete-case {task} rows in {role} cohort")
    return CheckedCohort(data=data, input_n=len(frame), eligible_n=len(data))
