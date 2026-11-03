"""In-memory checks only: no clinical or patient-level test data."""

import unittest
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from analysis_core import TASK_FEATURES, fit_locked_model, predict_scores, training_oof_scores
from cohort_checks import check_cohort
from feature_ablation import compare_training_panels, delong_auc_delta, macro_auc_delta_bootstrap
from metrics import bootstrap_ci, score_summary
from run_analysis import analyze


def random_frame(task: str, n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    frame = pd.DataFrame(rng.normal(size=(n, len(TASK_FEATURES[task]))), columns=TASK_FEATURES[task])
    frame["target"] = np.arange(n) % (4 if task == "level2" else 2)
    frame["cv_fold_5"] = np.arange(n) // (n // 5)
    return frame


class AnalysisTests(unittest.TestCase):
    def test_fold_scores_and_locked_evaluation(self):
        for task in ("level1", "level2", "level3"):
            with self.subTest(task=task), warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="X does not have valid feature names")
                frame = check_cohort(random_frame(task), task, "train").data
                oof = training_oof_scores(frame, task)
                self.assertTrue(np.isfinite(oof).all())
                model = fit_locked_model(frame, task)
                scores = predict_scores(model, frame, TASK_FEATURES[task], task)
                summary = score_summary(frame["target"].to_numpy(), scores, task)
                self.assertEqual(summary["n"], len(frame))
                self.assertEqual(oof.shape, scores.shape)

    def test_complete_case_validation_and_ci(self):
        frame = random_frame("level1")
        frame.loc[0, "WBC"] = np.nan
        checked = check_cohort(frame, "level1", "external")
        self.assertEqual(checked.excluded_n, 1)
        y = np.array([0, 0, 1, 1])
        s = np.array([0.1, 0.2, 0.8, 0.9])
        interval = bootstrap_ci(y, s, "level1", n_boot=30)
        self.assertEqual(interval["auroc"]["estimate"], 1.0)
        self.assertEqual(interval["auroc"]["ci_low"], 1.0)

    def test_paired_comparisons(self):
        y = np.array([0, 0, 0, 1, 1, 1])
        a = np.array([0.1, 0.3, 0.2, 0.8, 0.9, 0.7])
        b = np.array([0.2, 0.4, 0.1, 0.7, 0.6, 0.9])
        comparison = delong_auc_delta(y, a, b)
        self.assertAlmostEqual(comparison["delta_auroc"], roc_auc_score(y, a) - roc_auc_score(y, b))
        y4 = np.tile(np.arange(4), 5)
        p4 = np.eye(4)[y4] * 0.7 + 0.075
        delta = macro_auc_delta_bootstrap(y4, p4, p4, n_boot=30)
        self.assertEqual(delta["delta_auroc"], 0.0)

    def test_training_only_ablation_and_invalid_labels(self):
        frame = random_frame("level1")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names")
            result = compare_training_panels(
                frame, "level1", {"smaller": TASK_FEATURES["level1"][:6]}, n_boot=10,
            )
        self.assertEqual(result.loc[0, "method"], "paired DeLong")
        self.assertEqual(result.loc[0, "n_features"], 6)
        frame.loc[0, "target"] = 9
        with self.assertRaises(ValueError):
            check_cohort(frame, "level1", "train")

    def test_aggregate_entry_point(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names")
            frame = random_frame("level1")
            report = analyze("level1", frame, frame.copy(), frame.copy(), n_boot=10)
        self.assertIn("training_5fold_oof", report)
        self.assertIn("metrics_95ci", report["external_validation"])
        self.assertNotIn("scores", report)


if __name__ == "__main__":
    unittest.main()
