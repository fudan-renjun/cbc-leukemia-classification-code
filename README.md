# CBC-based leukemia classification: core analysis code

This repository contains a **concise code excerpt** illustrating the main modeling approach used in the study: task-specific CBC feature panels, five-fold training-set cross-validation, balanced LightGBM classifiers, and evaluation of a fitted model on a separate cohort. It is not the complete analysis pipeline and does not reproduce the published results by itself.

The three tasks shown are leukemia versus control (Level 1), AML/ALL/CML/CLL four-class classification (Level 2), and AML-M3/APL versus other AML (Level 3). Class labels for Level 2 are AML=0, ALL=1, CML=2, and CLL=3. The Level 3 model applies only within AML. The scores do **not** form a validated automatic diagnostic cascade.

## Files

- `analysis_core.py`: a short, data-agnostic excerpt of the principal modeling and metric calculations.
- `requirements.txt`: packages needed for the excerpt.
- `.gitignore`: guards against accidentally committing common clinical-data formats and model binaries.

No patient-level data, cohort assignment tables, report images, predictions, fitted model weights, or other derived datasets are included. The excerpt omits data cleaning, center harmonization, cohort construction, feature-selection searches, threshold selection, confidence intervals, calibration, decision-curve analysis, and supplementary analyses. Those steps should not be inferred from this small code sample.

## Data availability

The clinical data are not publicly deposited because they contain sensitive health information. Access to de-identified data may be requested from the corresponding author and is subject to institutional approval and applicable data-use agreements. This repository does not grant access to the data.

## Use

The functions accept in-memory pandas DataFrames supplied by an authorized analyst. There are no bundled data files or local paths. Use Python 3.11 or newer and install dependencies with `pip install -r requirements.txt`. The code is for methodological transparency only, not clinical diagnosis or individual risk prediction.
