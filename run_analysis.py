"""Aggregate-only CLI for authorized, already harmonized cohort tables."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from analysis_core import TASK_FEATURES, fit_locked_model, predict_scores, training_oof_scores
from cohort_checks import check_cohort
from metrics import bootstrap_ci, score_summary


def _json_ready(value):
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def analyze(task: str, train: pd.DataFrame, internal: pd.DataFrame,
            external: pd.DataFrame, n_boot: int = 500) -> dict:
    """Keep model selection and CV inside training; apply one fit to both holdouts."""
    training = check_cohort(train, task, "train")
    internal_checked = check_cohort(internal, task, "internal")
    external_checked = check_cohort(external, task, "external")
    oof = training_oof_scores(training.data, task)
    model = fit_locked_model(training.data, task)
    result = {
        "task": task,
        "n_features": len(TASK_FEATURES[task]),
        "training_5fold_oof": score_summary(training.data["target"].to_numpy(), oof, task),
    }
    for role, checked in (("internal_validation", internal_checked), ("external_validation", external_checked)):
        y = checked.data["target"].to_numpy()
        scores = predict_scores(model, checked.data, TASK_FEATURES[task], task)
        result[role] = {
            "input_n": checked.input_n,
            "eligible_n": checked.eligible_n,
            "excluded_incomplete_n": checked.excluded_n,
            "metrics_95ci": bootstrap_ci(y, scores, task, n_boot=n_boot),
        }
    return _json_ready(result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=sorted(TASK_FEATURES))
    parser.add_argument("--train", required=True, type=Path, help="Private training CSV with target and cv_fold_5")
    parser.add_argument("--internal", required=True, type=Path, help="Private internal-validation CSV")
    parser.add_argument("--external", required=True, type=Path, help="Private external-validation CSV")
    parser.add_argument("--bootstrap", type=int, default=500)
    args = parser.parse_args()
    report = analyze(
        args.task, pd.read_csv(args.train), pd.read_csv(args.internal),
        pd.read_csv(args.external), n_boot=args.bootstrap,
    )
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
