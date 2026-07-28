# Rice Variety Classification

[Türkçe açıklama](README.tr.md)

A reproducible machine-learning study that classifies **Cammeo** and
**Osmancik** rice varieties from seven morphological measurements.

The repository focuses on sound evaluation rather than a single optimistic
score: the split is stratified and deterministic, five-fold cross-validation
runs only on the training partition, and the final metrics come from an
untouched 20% holdout set.

![Confusion matrix and feature importance](docs/evaluation-summary.png)

## Results

| Evaluation | Metric | Result |
|---|---|---:|
| Training partition, 5-fold CV | Accuracy | 92.45% ± 1.10% |
| Training partition, 5-fold CV | Macro F1 | 92.29% ± 1.11% |
| Untouched holdout set (762 rows) | Accuracy | 92.13% |
| Untouched holdout set (762 rows) | Macro F1 | 91.90% |

The holdout confusion matrix, with rows as actual classes and columns as
predictions, is:

|  | Predicted Cammeo | Predicted Osmancik |
|---|---:|---:|
| Actual Cammeo | 287 | 39 |
| Actual Osmancik | 21 | 415 |

These figures are produced by the fixed configuration in `rice_model.py`; they
are not copied from an earlier notebook run. Full machine-readable results are
available in [`docs/metrics.json`](docs/metrics.json).

## Method

- **Data:** 3,810 rows, seven numerical features, no missing or duplicate rows
- **Classes:** 1,630 Cammeo and 2,180 Osmancik samples
- **Model:** random forest with 300 trees and `min_samples_leaf=2`
- **Split:** stratified 80/20 holdout, `random_state=42`
- **Validation:** five-fold stratified cross-validation on the training
  partition only
- **Metrics:** accuracy, balanced accuracy, macro precision, macro recall and
  macro F1

Random forests do not require feature scaling, so no scaler is fitted. No
preprocessing step sees the holdout labels or measurements before the final
evaluation.

## Run locally

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python train.py
```

The command writes `metrics.json` and `evaluation-summary.png` under
`artifacts/`. To run the checks:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
```

The cleaned notebook in [`notebooks/rice_exploration.ipynb`](notebooks/rice_exploration.ipynb)
uses the same tested functions as the command-line workflow.

## Project structure

```text
.
├── data/                         # Source workbook
├── docs/                         # Reproduced evaluation results
├── notebooks/                    # Output-free exploration notebook
├── tests/                        # Data and training checks
├── rice_model.py                 # Loading, validation, training and reporting
└── train.py                      # Command-line entry point
```

## Data source and scope

The workbook is an exact row-and-value match to the official UCI
**Rice (Cammeo and Osmancik)** dataset:

- Ilkay Cinar and Murat Koklu (2019)
- [UCI dataset page](https://archive.ics.uci.edu/dataset/545/rice+cammeo+and+osmancik)
- DOI: [10.24432/C5MW4Z](https://doi.org/10.24432/C5MW4Z)
- Dataset license: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

This project classifies already-extracted tabular measurements; it does **not**
classify raw rice images. The dataset license permits redistribution with
attribution. A separate license for the repository's source code has not yet
been selected.
