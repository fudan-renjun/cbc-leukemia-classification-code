"""Selected core methods only; no data loading or complete study pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline


TASK_FEATURES = {
    "level1": ["WBC", "RBC", "HGB", "RDWCV", "PLT", "LYM", "MONO", "NEUT"],
    "level2": [
        "WBC", "RBC", "HGB", "HCT", "MCV", "MCH", "MCHC", "RDWCV", "PLT",
        "LYM%", "LYM", "MONO%", "MONO", "NEUT%", "NEUT", "BASO%", "BASO", "EO%", "EO",
    ],
    "level3": [
        "WBC", "RBC", "HGB", "HCT", "MCV", "MCH", "MCHC", "RDWCV", "PLT", "MPV", "PCT",
        "LYM%", "LYM", "MONO%", "MONO", "NEUT%", "NEUT", "BASO%", "BASO", "EO%", "EO",
    ],
}
LEVEL2_CLASSES = np.array([0, 1, 2, 3])  # AML, ALL, CML, CLL
SEED = 20260630


def make_model(task: str):
    """Locked balanced LightGBM specification; imputation is fit within each fold."""
    if task not in TASK_FEATURES:
        raise ValueError(f"Unknown task: {task}")
    multiclass = task == "level2"
    classifier = LGBMClassifier(
        objective="multiclass" if multiclass else "binary",
        num_class=4 if multiclass else None,
        n_estimators=500 if multiclass else 520,
        learning_rate=0.035,
        num_leaves=31,
        min_child_samples=35,
        subsample=0.85,
        colsample_bytree=0.90,
        reg_alpha=0.2,
        reg_lambda=1.0,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
        verbosity=-1,
    )
    return make_pipeline(SimpleImputer(strategy="median"), classifier)


def training_oof_scores(train: pd.DataFrame, task: str, target: str = "target") -> np.ndarray:
    """Obtain five-fold out-of-fold scores using training rows only."""
    x = train[TASK_FEATURES[task]].to_numpy(dtype=np.float32)
    y = train[target].astype(int).to_numpy()
    scores = np.full((len(train), 4), np.nan) if task == "level2" else np.full(len(train), np.nan)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fit_idx, held_idx in folds.split(x, y):
        model = make_model(task)
        model.fit(x[fit_idx], y[fit_idx])
        proba = model.predict_proba(x[held_idx])
        if task == "level2":
            classes = model.named_steps["lgbmclassifier"].classes_
            for column, label in enumerate(classes):
                scores[held_idx, int(label)] = proba[:, column]
        else:
            scores[held_idx] = proba[:, 1]
    return scores


def fit_locked_model(train: pd.DataFrame, task: str, target: str = "target"):
    """Fit the already selected specification on the training cohort only."""
    model = clone(make_model(task))
    model.fit(train[TASK_FEATURES[task]].to_numpy(dtype=np.float32), train[target].astype(int))
    return model


def evaluate_cohort(model, cohort: pd.DataFrame, task: str, target: str = "target") -> dict[str, float]:
    """Apply a fitted model unchanged to an independent labeled cohort."""
    y = cohort[target].astype(int).to_numpy()
    proba = model.predict_proba(cohort[TASK_FEATURES[task]].to_numpy(dtype=np.float32))
    if task == "level2":
        if not np.array_equal(model.named_steps["lgbmclassifier"].classes_, LEVEL2_CLASSES):
            raise ValueError("Level 2 class order differs from AML/ALL/CML/CLL mapping")
        y_pred = LEVEL2_CLASSES[np.argmax(proba, axis=1)]
        one_vs_rest = [roc_auc_score(y == c, proba[:, i]) for i, c in enumerate(LEVEL2_CLASSES)]
        return {
            "macro_auroc_ovr": float(np.mean(one_vs_rest)),
            "balanced_accuracy": float(balanced_accuracy_score(y, y_pred)),
        }
    score = proba[:, 1]
    return {
        "auroc": float(roc_auc_score(y, score)),
        "auprc": float(average_precision_score(y, score)),
        "balanced_accuracy_at_0.5": float(balanced_accuracy_score(y, score >= 0.5)),
    }
