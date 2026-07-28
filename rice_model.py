"""Reproducible training and evaluation for the UCI rice dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

FEATURE_COLUMNS = [
    "Area",
    "Perimeter",
    "Major_Axis_Length",
    "Minor_Axis_Length",
    "Eccentricity",
    "Convex_Area",
    "Extent",
]
TARGET_COLUMN = "Class"
CLASS_NAMES = ["Cammeo", "Osmancik"]
RANDOM_STATE = 42


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load the workbook and fail early if its schema or contents drift."""

    data_path = Path(path)
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    data = pd.read_excel(data_path)
    expected_columns = FEATURE_COLUMNS + [TARGET_COLUMN]

    if list(data.columns) != expected_columns:
        raise ValueError(
            "Unexpected dataset columns. "
            f"Expected {expected_columns}, received {list(data.columns)}."
        )
    if data.empty:
        raise ValueError("Dataset is empty.")
    if data.isna().any().any():
        raise ValueError("Dataset contains missing values.")
    if data.duplicated().any():
        raise ValueError("Dataset contains duplicate rows.")
    if set(data[TARGET_COLUMN].unique()) != set(CLASS_NAMES):
        raise ValueError(
            f"Unexpected classes: {sorted(data[TARGET_COLUMN].unique().tolist())}"
        )
    if not all(pd.api.types.is_numeric_dtype(data[column]) for column in FEATURE_COLUMNS):
        raise ValueError("All feature columns must be numeric.")

    return data


def split_dataset(
    data: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a deterministic holdout split while preserving class ratios."""

    features = data[FEATURE_COLUMNS]
    target = data[TARGET_COLUMN]
    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )


def build_model(
    *,
    n_estimators: int = 300,
    min_samples_leaf: int = 2,
    random_state: int = RANDOM_STATE,
) -> RandomForestClassifier:
    """Return the fixed random-forest configuration used in the report."""

    return RandomForestClassifier(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1,
    )


def _mean_and_std(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


def _serializable_report(report: dict[str, Any]) -> dict[str, Any]:
    serializable: dict[str, Any] = {}
    for key, value in report.items():
        if isinstance(value, dict):
            serializable[key] = {
                nested_key: float(nested_value)
                for nested_key, nested_value in value.items()
            }
        else:
            serializable[key] = float(value)
    return serializable


def run_experiment(
    data_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    test_size: float = 0.2,
    cv_splits: int = 5,
    n_estimators: int = 300,
    min_samples_leaf: int = 2,
    random_state: int = RANDOM_STATE,
) -> tuple[dict[str, Any], RandomForestClassifier]:
    """Validate, train and evaluate the model without touching the holdout early."""

    data = load_dataset(data_path)
    x_train, x_test, y_train, y_test = split_dataset(
        data,
        test_size=test_size,
        random_state=random_state,
    )

    model = build_model(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    folds = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=random_state,
    )
    cv_scores = cross_validate(
        model,
        x_train,
        y_train,
        cv=folds,
        scoring={
            "accuracy": "accuracy",
            "balanced_accuracy": "balanced_accuracy",
            "macro_precision": "precision_macro",
            "macro_recall": "recall_macro",
            "macro_f1": "f1_macro",
        },
        n_jobs=-1,
    )

    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    matrix = confusion_matrix(y_test, predictions, labels=CLASS_NAMES)
    report = classification_report(
        y_test,
        predictions,
        labels=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    results: dict[str, Any] = {
        "dataset": {
            "rows": int(len(data)),
            "features": int(len(FEATURE_COLUMNS)),
            "missing_values": int(data.isna().sum().sum()),
            "duplicate_rows": int(data.duplicated().sum()),
            "class_distribution": {
                class_name: int((data[TARGET_COLUMN] == class_name).sum())
                for class_name in CLASS_NAMES
            },
        },
        "configuration": {
            "model": "RandomForestClassifier",
            "n_estimators": n_estimators,
            "min_samples_leaf": min_samples_leaf,
            "test_size": test_size,
            "cv_splits": cv_splits,
            "random_state": random_state,
        },
        "split": {
            "training_rows": int(len(x_train)),
            "holdout_rows": int(len(x_test)),
            "stratified": True,
        },
        "cross_validation_on_training_partition": {
            metric: _mean_and_std(cv_scores[f"test_{metric}"])
            for metric in (
                "accuracy",
                "balanced_accuracy",
                "macro_precision",
                "macro_recall",
                "macro_f1",
            )
        },
        "holdout": {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "balanced_accuracy": float(
                balanced_accuracy_score(y_test, predictions)
            ),
            "macro_precision": float(
                precision_score(y_test, predictions, average="macro")
            ),
            "macro_recall": float(recall_score(y_test, predictions, average="macro")),
            "macro_f1": float(f1_score(y_test, predictions, average="macro")),
            "label_order": CLASS_NAMES,
            "confusion_matrix": matrix.tolist(),
            "classification_report": _serializable_report(report),
        },
        "feature_importance": {
            feature: float(importance)
            for feature, importance in sorted(
                zip(FEATURE_COLUMNS, model.feature_importances_, strict=True),
                key=lambda item: item[1],
                reverse=True,
            )
        },
    }

    if output_dir is not None:
        write_artifacts(results, matrix, output_dir)

    return results, model


def write_artifacts(
    results: dict[str, Any],
    matrix: np.ndarray,
    output_dir: str | Path,
) -> None:
    """Write a machine-readable report and a compact portfolio visual."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    metrics_path = destination / "metrics.json"
    metrics_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    feature_importance = results["feature_importance"]
    feature_names = list(reversed(feature_importance.keys()))
    importance_values = list(reversed(feature_importance.values()))

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))

    image = axes[0].imshow(matrix, cmap="Blues")
    axes[0].set_title("Holdout confusion matrix")
    axes[0].set_xlabel("Predicted class")
    axes[0].set_ylabel("Actual class")
    axes[0].set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    axes[0].set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    threshold = float(matrix.max()) / 2
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axes[0].text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
                fontsize=12,
                fontweight="bold",
            )
    figure.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)

    axes[1].barh(feature_names, importance_values, color="#3b82f6")
    axes[1].set_title("Random-forest feature importance")
    axes[1].set_xlabel("Mean decrease in impurity")
    axes[1].grid(axis="x", alpha=0.25)

    holdout = results["holdout"]
    figure.suptitle(
        "Rice variety classification"
        f" — holdout accuracy {holdout['accuracy']:.2%},"
        f" macro F1 {holdout['macro_f1']:.2%}",
        fontsize=14,
    )
    figure.tight_layout()
    figure.savefig(
        destination / "evaluation-summary.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)
